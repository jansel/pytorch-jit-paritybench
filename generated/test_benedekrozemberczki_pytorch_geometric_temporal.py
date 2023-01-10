import sys
_module = sys.modules[__name__]
del sys
conf = _module
a3tgcn2_example = _module
a3tgcn_example = _module
agcrn_example = _module
dcrnn_example = _module
dygrencoder_example = _module
evolvegcnh_example = _module
evolvegcno_example = _module
gclstm_example = _module
gconvgru_example = _module
gconvlstm_example = _module
lightning_example = _module
lrgcn_example = _module
mpnnlstm_example = _module
tgcn_example = _module
setup = _module
attention_test = _module
batch_test = _module
dataset_test = _module
heterogeneous_test = _module
recurrent_test = _module
torch_geometric_temporal = _module
dataset = _module
chickenpox = _module
encovid = _module
metr_la = _module
montevideo_bus = _module
mtm = _module
pedalme = _module
pems_bay = _module
twitter_tennis = _module
wikimath = _module
windmilllarge = _module
windmillmedium = _module
windmillsmall = _module
nn = _module
attention = _module
astgcn = _module
dnntsp = _module
gman = _module
mstgcn = _module
mtgnn = _module
stgcn = _module
tsagcn = _module
hetero = _module
heterogclstm = _module
recurrent = _module
agcrn = _module
attentiontemporalgcn = _module
dcrnn = _module
dygrae = _module
evolvegcnh = _module
evolvegcno = _module
gc_lstm = _module
gconv_gru = _module
gconv_lstm = _module
lrgcn = _module
mpnn_lstm = _module
temporalgcn = _module
signal = _module
dynamic_graph_static_signal = _module
dynamic_graph_static_signal_batch = _module
dynamic_graph_temporal_signal = _module
dynamic_graph_temporal_signal_batch = _module
dynamic_hetero_graph_static_signal = _module
dynamic_hetero_graph_static_signal_batch = _module
dynamic_hetero_graph_temporal_signal = _module
dynamic_hetero_graph_temporal_signal_batch = _module
static_graph_temporal_signal = _module
static_graph_temporal_signal_batch = _module
static_hetero_graph_temporal_signal = _module
static_hetero_graph_temporal_signal_batch = _module
train_test_split = _module

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


import numpy as np


import matplotlib.pyplot as plt


import torch


import torch.nn.functional as F


from torch.nn import functional as F


import math


from typing import Optional


from typing import List


from typing import Union


import torch.nn as nn


from torch.nn import Parameter


from typing import Callable


import numbers


from torch.nn import init


from torch.autograd import Variable


from torch.nn import LSTM


from torch.nn import GRU


from typing import Tuple


from torch import Tensor


from typing import Dict


class TGCN2(torch.nn.Module):
    """An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, batch_size: int, improved: bool=False, cached: bool=False, add_self_loops: bool=True):
        super(TGCN2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=2)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=2)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=2)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class A3TGCN2(torch.nn.Module):
    """An implementation THAT SUPPORTS BATCHES of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(self, in_channels: int, out_channels: int, periods: int, batch_size: int, improved: bool=False, cached: bool=False, add_self_loops: bool=True):
        super(A3TGCN2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(in_channels=self.in_channels, out_channels=self.out_channels, batch_size=self.batch_size, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(X[:, :, :, period], edge_index, edge_weight, H)
        return H_accum


class TemporalGNN(torch.nn.Module):

    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods, batch_size=batch_size)
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


class TGCN(torch.nn.Module):
    """An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, improved: bool=False, cached: bool=False, add_self_loops: bool=True):
        super(TGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class RecurrentGCN(torch.nn.Module):

    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, 32)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class Conv2D(nn.Module):
    """An implementation of the 2D-convolution block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.relu.
        bn_decay (float, optional): Batch normalization momentum, default is None.
    """

    def __init__(self, input_dims: int, output_dims: int, kernel_size: Union[tuple, list], stride: Union[tuple, list]=(1, 1), use_bias: bool=True, activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]]=F.relu, bn_decay: Optional[float]=None):
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride, padding=0, bias=use_bias)
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self._conv2d.weight)
        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the 2D-convolution block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, input_dims).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, output_dims).
        """
        X = X.permute(0, 3, 2, 1)
        X = self._conv2d(X)
        X = self._batch_norm(X)
        if self._activation is not None:
            X = self._activation(X)
        return X.permute(0, 3, 2, 1)


class FullyConnected(nn.Module):
    """An implementation of the fully-connected layer.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        input_dims (int or list): Dimension(s) of input.
        units (int or list): Dimension(s) of outputs in each 2D convolution block.
        activations (Callable or list): Activation function(s).
        bn_decay (float, optional): Batch normalization momentum, default is None.
        use_bias (bool, optional): Whether to use bias, default is True.
    """

    def __init__(self, input_dims: Union[int, list], units: Union[int, list], activations: Union[Callable[[torch.FloatTensor], torch.FloatTensor], list], bn_decay: float, use_bias: bool=True):
        super(FullyConnected, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        assert type(units) == list
        self._conv2ds = nn.ModuleList([Conv2D(input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1], use_bias=use_bias, activation=activation, bn_decay=bn_decay) for input_dim, num_unit, activation in zip(input_dims, units, activations)])

    def forward(self, X: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the fully-connected layer.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, 1).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        for conv in self._conv2ds:
            X = conv(X)
        return X


class SpatialAttention(nn.Module):
    """An implementation of the spatial attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
    """

    def __init__(self, K: int, d: int, bn_decay: float):
        super(SpatialAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._fully_connected_q = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected_k = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected_v = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected = FullyConnected(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the spatial attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Spatial attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class TemporalAttention(nn.Module):
    """An implementation of the temporal attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
        mask (bool): Whether to mask attention score.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool):
        super(TemporalAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._mask = mask
        self._fully_connected_q = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected_k = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected_v = FullyConnected(input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected = FullyConnected(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the temporal attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        if self._mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)
            mask = mask
            condition = torch.FloatTensor([-2 ** 15 + 1])
            attention = torch.where(mask, attention, condition)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class ASTGCNBlock(nn.Module):
    """An implementation of the Attention Based Spatial-Temporal Graph Convolutional Block.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int, time_strides: int, num_of_vertices: int, num_of_timesteps: int, normalization: Optional[str]=None, bias: bool=True):
        super(ASTGCNBlock, self).__init__()
        self._temporal_attention = TemporalAttention(in_channels, num_of_vertices, num_of_timesteps)
        self._spatial_attention = SpatialAttention(in_channels, num_of_vertices, num_of_timesteps)
        self._chebconv_attention = ChebConvAttention(in_channels, nb_chev_filter, K, normalization, bias)
        self._time_convolution = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self._residual_convolution = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self._normalization = normalization
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, edge_index: Union[torch.LongTensor, List[torch.LongTensor]]) ->torch.FloatTensor:
        """
        Making a forward pass with the ASTGCN block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = X.shape
        X_tilde = self._temporal_attention(X)
        X_tilde = torch.matmul(X.reshape(batch_size, -1, num_of_timesteps), X_tilde)
        X_tilde = X_tilde.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        X_tilde = self._spatial_attention(X_tilde)
        if not isinstance(edge_index, list):
            data = Data(edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices)
            if self._normalization != 'sym':
                lambda_max = LaplacianLambdaMax()(data).lambda_max
            else:
                lambda_max = None
            X_hat = []
            for t in range(num_of_timesteps):
                X_hat.append(torch.unsqueeze(self._chebconv_attention(X[:, :, :, t], edge_index, X_tilde, lambda_max=lambda_max), -1))
            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        else:
            X_hat = []
            for t in range(num_of_timesteps):
                data = Data(edge_index=edge_index[t], edge_attr=None, num_nodes=num_of_vertices)
                if self._normalization != 'sym':
                    lambda_max = LaplacianLambdaMax()(data).lambda_max
                else:
                    lambda_max = None
                X_hat.append(torch.unsqueeze(self._chebconv_attention(X[:, :, :, t], edge_index[t], X_tilde, lambda_max=lambda_max), -1))
            X_hat = F.relu(torch.cat(X_hat, dim=-1))
        X_hat = self._time_convolution(X_hat.permute(0, 2, 1, 3))
        X = self._residual_convolution(X.permute(0, 2, 1, 3))
        X = self._layer_norm(F.relu(X + X_hat).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X


class ASTGCN(nn.Module):
    """An implementation of the Attention Based Spatial-Temporal Graph Convolutional Cell.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        edge_index (array): edge indices.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
        num_of_vertices (int): Number of vertices in the graph.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):
            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`
            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`
            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, nb_block: int, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int, time_strides: int, num_for_predict: int, len_input: int, num_of_vertices: int, normalization: Optional[str]=None, bias: bool=True):
        super(ASTGCN, self).__init__()
        self._blocklist = nn.ModuleList([ASTGCNBlock(in_channels, K, nb_chev_filter, nb_time_filter, time_strides, num_of_vertices, len_input, normalization, bias)])
        self._blocklist.extend([ASTGCNBlock(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, num_of_vertices, len_input // time_strides, normalization, bias) for _ in range(nb_block - 1)])
        self._final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor) ->torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * **edge_index** (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * **X** (PyTorch FloatTensor)* - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        for block in self._blocklist:
            X = block(X, edge_index)
        X = self._final_conv(X.permute(0, 3, 1, 2))
        X = X[:, :, :, -1]
        X = X.permute(0, 2, 1)
        return X


class MaskedSelfAttention(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads, attention_aggregate='mean'):
        super(MaskedSelfAttention, self).__init__()
        self.attention_aggregate = attention_aggregate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        if attention_aggregate == 'concat':
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim // n_heads
        elif attention_aggregate == 'mean':
            self.per_head_dim = self.dq = self.dk = self.dv = output_dim
        else:
            raise ValueError(f'wrong value for aggregate {attention_aggregate}')
        self.Wq = nn.Linear(input_dim, n_heads * self.dq, bias=False)
        self.Wk = nn.Linear(input_dim, n_heads * self.dk, bias=False)
        self.Wv = nn.Linear(input_dim, n_heads * self.dv, bias=False)

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1]
        Q = self.Wq(input_tensor)
        K = self.Wk(input_tensor)
        V = self.Wv(input_tensor)
        Q = Q.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dq).transpose(1, 2)
        K = K.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dk).permute(0, 2, 3, 1)
        V = V.reshape(input_tensor.shape[0], input_tensor.shape[1], self.n_heads, self.dv).transpose(1, 2)
        attention_score = Q.matmul(K) / np.sqrt(self.per_head_dim)
        attention_mask = torch.zeros(seq_length, seq_length).masked_fill(torch.tril(torch.ones(seq_length, seq_length)) == 0, -np.inf)
        attention_score = attention_score + attention_mask
        attention_score = torch.softmax(attention_score, dim=-1)
        multi_head_result = attention_score.matmul(V)
        if self.attention_aggregate == 'concat':
            output = multi_head_result.transpose(1, 2).reshape(input_tensor.shape[0], seq_length, self.n_heads * self.per_head_dim)
        elif self.attention_aggregate == 'mean':
            output = multi_head_result.transpose(1, 2).mean(dim=2)
        else:
            raise ValueError(f'wrong value for aggregate {self.attention_aggregate}')
        None
        return output


class GlobalGatedUpdater(nn.Module):

    def __init__(self, items_total, item_embedding):
        super(GlobalGatedUpdater, self).__init__()
        self.items_total = items_total
        self.item_embedding = item_embedding
        self.alpha = nn.Parameter(torch.rand(items_total, 1), requires_grad=True)

    def forward(self, nodes_output):
        batch_size = nodes_output.shape[0] // self.items_total
        id = 0
        num_nodes = self.items_total
        items_embedding = self.item_embedding(torch.tensor([i for i in range(self.items_total)]))
        batch_embedding = []
        for _ in range(batch_size):
            output_node_features = nodes_output[id:id + num_nodes, :]
            embed = (1 - self.alpha) * items_embedding
            embed = embed + self.alpha * output_node_features
            batch_embedding.append(embed)
            id += num_nodes
        batch_embedding = torch.stack(batch_embedding)
        return batch_embedding


class AggregateTemporalNodeFeatures(nn.Module):

    def __init__(self, item_embed_dim):
        super(AggregateTemporalNodeFeatures, self).__init__()
        self.Wq = nn.Linear(item_embed_dim, item_embed_dim, bias=False)

    def forward(self, nodes_output):
        aggregated_features = []
        for l in range(nodes_output.shape[0]):
            output_node_features = nodes_output[l, :, :]
            weights = self.Wq(output_node_features)
            aggregated_features.append(weights)
        aggregated_features = torch.cat(aggregated_features, dim=0)
        None
        return aggregated_features


class WeightedGCNBlock(nn.Module):

    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int):
        super(WeightedGCNBlock, self).__init__()
        gcns, relus, bns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        input_size = in_features
        for hidden_size in hidden_sizes:
            gcns.append(GCNConv(input_size, hidden_size))
            relus.append(nn.ReLU())
            bns.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        gcns.append(GCNConv(hidden_sizes[-1], out_features))
        relus.append(nn.ReLU())
        bns.append(nn.BatchNorm1d(out_features))
        self.gcns, self.relus, self.bns = gcns, relus, bns

    def forward(self, node_features: torch.FloatTensor, edge_index: torch.LongTensor, edges_weight: torch.LongTensor):
        h = node_features
        for gcn, relu, bn in zip(self.gcns, self.relus, self.bns):
            h = gcn(h, edge_index, edges_weight)
            h = bn(h.transpose(1, -1)).transpose(1, -1)
            h = relu(h)
        return h


class DNNTSP(nn.Module):
    """An implementation of the Deep Neural Network for Temporal Set Prediction.
    For details see: `"Predicting Temporal Sets with Deep Neural Networks" <https://dl.acm.org/doi/abs/10.1145/3394486.3403152>`_

    Args:
        items_total (int): Total number of items in the sets. Cardinality of the union.
        item_embedding_dim (int): Item embedding dimensions.
        n_heads (int): Number of attention heads.
    """

    def __init__(self, items_total: int, item_embedding_dim: int, n_heads: int):
        super(DNNTSP, self).__init__()
        self.item_embedding = nn.Embedding(items_total, item_embedding_dim)
        self.item_embedding_dim = item_embedding_dim
        self.items_total = items_total
        self.stacked_gcn = WeightedGCNBlock(item_embedding_dim, [item_embedding_dim], item_embedding_dim)
        self.masked_self_attention = MaskedSelfAttention(input_dim=item_embedding_dim, output_dim=item_embedding_dim, n_heads=n_heads)
        self.aggregate_nodes_temporal_feature = AggregateTemporalNodeFeatures(item_embed_dim=item_embedding_dim)
        self.global_gated_updater = GlobalGatedUpdater(items_total=items_total, item_embedding=self.item_embedding)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None):
        """Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self.stacked_gcn(X, edge_index, edge_weight)
        H = H.view(-1, self.items_total, self.item_embedding_dim)
        H = self.masked_self_attention(H)
        H = self.aggregate_nodes_temporal_feature(H)
        H = self.global_gated_updater(H)
        return H


class SpatioTemporalEmbedding(nn.Module):
    """An implementation of the spatial-temporal embedding block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : Dimension of output.
        bn_decay (float): Batch normalization momentum.
        steps_per_day (int): Steps to take for a day.
        use_bias (bool, optional): Whether to use bias in Fully Connected layers, default is True.
    """

    def __init__(self, D: int, bn_decay: float, steps_per_day: int, use_bias: bool=True):
        super(SpatioTemporalEmbedding, self).__init__()
        self._fully_connected_se = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay, use_bias=use_bias)
        self._fully_connected_te = FullyConnected(input_dims=[steps_per_day + 7, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay, use_bias=use_bias)

    def forward(self, SE: torch.FloatTensor, TE: torch.FloatTensor, T: int) ->torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.

        Arg types:
            * **SE** (PyTorch Float Tensor) - Spatial embedding, with shape (num_nodes, D).
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, 2).(dayofweek, timeofday)
            * **T** (int) - Number of time steps in one day.

        Return types:
            * **output** (PyTorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_his + num_pred, num_nodes, D).
        """
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self._fully_connected_se(SE)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i] % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j] % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self._fully_connected_te(TE)
        del dayofweek, timeofday
        return SE + TE


class GatedFusion(nn.Module):
    """An implementation of the gated fusion mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : dimension of output.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, D: int, bn_decay: float):
        super(GatedFusion, self).__init__()
        self._fully_connected_xs = FullyConnected(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=False)
        self._fully_connected_xt = FullyConnected(input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=True)
        self._fully_connected_h = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, HS: torch.FloatTensor, HT: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the gated fusion mechanism.

        Arg types:
            * **HS** (PyTorch Float Tensor) - Spatial attention scores, with shape (batch_size, num_step, num_nodes, D).
            * **HT** (Pytorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, D).

        Return types:
            * **H** (PyTorch Float Tensor) - Spatial-temporal attention scores, with shape (batch_size, num_step, num_nodes, D).
        """
        XS = self._fully_connected_xs(HS)
        XT = self._fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._fully_connected_h(H)
        del XS, XT, z
        return H


class SpatioTemporalAttention(nn.Module):
    """An implementation of the spatial-temporal attention block, with spatial attention and temporal attention
    followed by gated fusion. For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
        mask (bool): Whether to mask attention score in temporal attention.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool):
        super(SpatioTemporalAttention, self).__init__()
        self._spatial_attention = SpatialAttention(K, d, bn_decay)
        self._temporal_attention = TemporalAttention(K, d, bn_decay, mask=mask)
        self._gated_fusion = GatedFusion(K * d, bn_decay)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal attention block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        HS = self._spatial_attention(X, STE)
        HT = self._temporal_attention(X, STE)
        H = self._gated_fusion(HS, HT)
        del HS, HT
        X = torch.add(X, H)
        return X


class TransformAttention(nn.Module):
    """An implementation of the tranform attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
    """

    def __init__(self, K: int, d: int, bn_decay: float):
        super(TransformAttention, self).__init__()
        D = K * d
        self._K = K
        self._d = d
        self._fully_connected_q = FullyConnected(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected_k = FullyConnected(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected_v = FullyConnected(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
        self._fully_connected = FullyConnected(input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, STE_his: torch.FloatTensor, STE_pred: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the transform attention layer.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_his, num_nodes, K*d).
            * **STE_his** (Pytorch Float Tensor) - Spatial-temporal embedding for history,
            with shape (batch_size, num_his, num_nodes, K*d).
            * **STE_pred** (Pytorch Float Tensor) - Spatial-temporal embedding for prediction,
            with shape (batch_size, num_pred, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Output sequence for prediction, with shape (batch_size, num_pred, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        query = self._fully_connected_q(STE_pred)
        key = self._fully_connected_k(STE_his)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    """An implementation of GMAN.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        L (int) : Number of STAtt blocks in the encoder/decoder.
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        num_his (int): Number of history steps.
        bn_decay (float): Batch normalization momentum.
        steps_per_day (int): Number of steps in a day.
        use_bias (bool): Whether to use bias in Fully Connected layers.
        mask (bool): Whether to mask attention score in temporal attention.
    """

    def __init__(self, L: int, K: int, d: int, num_his: int, bn_decay: float, steps_per_day: int, use_bias: bool, mask: bool):
        super(GMAN, self).__init__()
        D = K * d
        self._num_his = num_his
        self._steps_per_day = steps_per_day
        self._st_embedding = SpatioTemporalEmbedding(D, bn_decay, steps_per_day, use_bias)
        self._st_att_block1 = nn.ModuleList([SpatioTemporalAttention(K, d, bn_decay, mask) for _ in range(L)])
        self._st_att_block2 = nn.ModuleList([SpatioTemporalAttention(K, d, bn_decay, mask) for _ in range(L)])
        self._transform_attention = TransformAttention(K, d, bn_decay)
        self._fully_connected_1 = FullyConnected(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self._fully_connected_2 = FullyConnected(input_dims=[D, D], units=[D, 1], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor, SE: torch.FloatTensor, TE: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of GMAN.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_hist, num of nodes).
            * **SE** (Pytorch Float Tensor) - Spatial embedding, with shape (numbed of nodes, K * d).
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, 2).

        Return types:
            * **X** (PyTorch Float Tensor) - Output sequence for prediction, with shape (batch_size, num_pred, num of nodes).
        """
        X = torch.unsqueeze(X, -1)
        X = self._fully_connected_1(X)
        STE = self._st_embedding(SE, TE, self._steps_per_day)
        STE_his = STE[:, :self._num_his]
        STE_pred = STE[:, self._num_his:]
        for net in self._st_att_block1:
            X = net(X, STE_his)
        X = self._transform_attention(X, STE_his, STE_pred)
        for net in self._st_att_block2:
            X = net(X, STE_pred)
        X = torch.squeeze(self._fully_connected_2(X), 3)
        return X


class MSTGCNBlock(nn.Module):
    """An implementation of the Multi-Component Spatial-Temporal Graph
    Convolution block from this paper: `"Attention Based Spatial-Temporal
    Graph Convolutional Networks for Traffic Flow Forecasting."
    <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filters (int): Number of Chebyshev filters.
        nb_time_filters (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
    """

    def __init__(self, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int, time_strides: int):
        super(MSTGCNBlock, self).__init__()
        self._cheb_conv = ChebConv(in_channels, nb_chev_filter, K, normalization=None)
        self._time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self._residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        self.nb_time_filter = nb_time_filter
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor) ->torch.FloatTensor:
        """
        Making a forward pass with a single MSTGCN block.

        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, nb_time_filter, T_out).
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = X.shape
        if not isinstance(edge_index, list):
            lambda_max = LaplacianLambdaMax()(Data(edge_index=edge_index, edge_attr=None, num_nodes=num_of_vertices)).lambda_max
            X_tilde = X.permute(2, 0, 1, 3)
            X_tilde = X_tilde.reshape(num_of_vertices, in_channels, num_of_timesteps * batch_size)
            X_tilde = X_tilde.permute(2, 0, 1)
            X_tilde = F.relu(self._cheb_conv(x=X_tilde, edge_index=edge_index, lambda_max=lambda_max))
            X_tilde = X_tilde.permute(1, 2, 0)
            X_tilde = X_tilde.reshape(num_of_vertices, self.nb_time_filter, batch_size, num_of_timesteps)
            X_tilde = X_tilde.permute(2, 0, 1, 3)
        else:
            X_tilde = []
            for t in range(num_of_timesteps):
                lambda_max = LaplacianLambdaMax()(Data(edge_index=edge_index[t], edge_attr=None, num_nodes=num_of_vertices)).lambda_max
                X_tilde.append(torch.unsqueeze(self._cheb_conv(X[:, :, :, t], edge_index[t], lambda_max=lambda_max), -1))
            X_tilde = F.relu(torch.cat(X_tilde, dim=-1))
        X_tilde = self._time_conv(X_tilde.permute(0, 2, 1, 3))
        X = self._residual_conv(X.permute(0, 2, 1, 3))
        X = self._layer_norm(F.relu(X + X_tilde).permute(0, 3, 2, 1))
        X = X.permute(0, 2, 3, 1)
        return X


class MSTGCN(nn.Module):
    """An implementation of the Multi-Component Spatial-Temporal Graph Convolution Networks, a degraded version of ASTGCN.
    For details see this paper: `"Attention Based Spatial-Temporal Graph Convolutional
    Networks for Traffic Flow Forecasting." <https://ojs.aaai.org/index.php/AAAI/article/view/3881>`_

    Args:

        nb_block (int): Number of ASTGCN blocks in the model.
        in_channels (int): Number of input features.
        K (int): Order of Chebyshev polynomials. Degree is K-1.
        nb_chev_filter (int): Number of Chebyshev filters.
        nb_time_filter (int): Number of time filters.
        time_strides (int): Time strides during temporal convolution.
        num_for_predict (int): Number of predictions to make in the future.
        len_input (int): Length of the input sequence.
    """

    def __init__(self, nb_block: int, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int, time_strides: int, num_for_predict: int, len_input: int):
        super(MSTGCN, self).__init__()
        self._blocklist = nn.ModuleList([MSTGCNBlock(in_channels, K, nb_chev_filter, nb_time_filter, time_strides)])
        self._blocklist.extend([MSTGCNBlock(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1) for _ in range(nb_block - 1)])
        self._final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resetting the model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor) ->torch.FloatTensor:
        """Making a forward pass. This module takes a likst of MSTGCN blocks and use a final convolution to serve as a multi-component fusion.
        B is the batch size. N_nodes is the number of nodes in the graph. F_in is the dimension of input features.
        T_in is the length of input sequence in time. T_out is the length of output sequence in time.

        Arg types:
            * X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
            * edge_index (PyTorch LongTensor): Edge indices, can be an array of a list of Tensor arrays, depending on whether edges change over time.

        Return types:
            * X (PyTorch FloatTensor) - Hidden state tensor for all nodes, with shape (B, N_nodes, T_out).
        """
        for block in self._blocklist:
            X = block(X, edge_index)
        X = self._final_conv(X.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return X


class Linear(nn.Module):
    """An implementation of the linear layer, conducting 2D convolution.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool, optional): Whether to have bias. Default: True.
    """

    def __init__(self, c_in: int, c_out: int, bias: bool=True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of the linear layer.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        return self._mlp(X)


class MixProp(nn.Module):
    """An implementation of the dynatic mix-hop propagation layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        A = A + torch.eye(A.size(0))
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdep):
            H = self._alpha * X + (1 - self._alpha) * torch.einsum('ncwl,vw->ncvl', (H, A))
            H_0 = torch.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    """An implementation of the dilated inception layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor.
    """

    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor)))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3):]
        X = torch.cat(X, dim=1)
        return X


class GraphConstructor(nn.Module):
    """An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(self, nnodes: int, k: int, dim: int, alpha: float, xd: Optional[int]=None):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)
        self._k = k
        self._alpha = alpha
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor]=None) ->torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
            * **FE** (Pytorch Float Tensor, optional) - Static feature, default None.
        Return types:
            * **A** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """
        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0))
        mask.fill_(float('0'))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A


class LayerNormalization(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    """An implementation of the layer normalization layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." 
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        normalized_shape (int): Input shape from an expected input of size.
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine (bool, optional): Whether to conduct elementwise affine transformation or not. Default: True.
    """

    def __init__(self, normalized_shape: int, eps: float=1e-05, elementwise_affine: bool=True):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('_weight', None)
            self.register_parameter('_bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) ->torch.FloatTensor:
        """
        Making a forward pass of layer normalization.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
            * **idx** (Pytorch Long Tensor) - Input indices.

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
        """
        if self._elementwise_affine:
            return F.layer_norm(X, tuple(X.shape[1:]), self._weight[:, idx, :], self._bias[:, idx, :], self._eps)
        else:
            return F.layer_norm(X, tuple(X.shape[1:]), self._weight, self._bias, self._eps)


class MTGNNLayer(nn.Module):
    """An implementation of the MTGNN layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        dilation_exponential (int): Dilation exponential.
        rf_size_i (int): Size of receptive field.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        j (int): Iteration index.
        residual_channels (int): Residual channels.
        conv_channels (int): Convolution channels.
        skip_channels (int): Skip channels.
        kernel_set (list of int): List of kernel sizes.
        new_dilation (int): Dilation.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        gcn_true (bool): Whether to add graph convolution layer.
        seq_length (int): Length of input sequence.
        receptive_field (int): Receptive field.
        dropout (float): Droupout rate.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.

    """

    def __init__(self, dilation_exponential: int, rf_size_i: int, kernel_size: int, j: int, residual_channels: int, conv_channels: int, skip_channels: int, kernel_set: list, new_dilation: int, layer_norm_affline: bool, gcn_true: bool, seq_length: int, receptive_field: int, dropout: float, gcn_depth: int, num_nodes: int, propalpha: float):
        super(MTGNNLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true
        if dilation_exponential > 1:
            rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)
        self._filter_conv = DilatedInception(residual_channels, conv_channels, kernel_set=kernel_set, dilation_factor=new_dilation)
        self._gate_conv = DilatedInception(residual_channels, conv_channels, kernel_set=kernel_set, dilation_factor=new_dilation)
        self._residual_conv = nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1))
        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, seq_length - rf_size_j + 1))
        else:
            self._skip_conv = nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, receptive_field - rf_size_j + 1))
        if gcn_true:
            self._mixprop_conv1 = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
            self._mixprop_conv2 = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
        if seq_length > receptive_field:
            self._normalization = LayerNormalization((residual_channels, num_nodes, seq_length - rf_size_j + 1), elementwise_affine=layer_norm_affline)
        else:
            self._normalization = LayerNormalization((residual_channels, num_nodes, receptive_field - rf_size_j + 1), elementwise_affine=layer_norm_affline)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, X_skip: torch.FloatTensor, A_tilde: Optional[torch.FloatTensor], idx: torch.LongTensor, training: bool) ->torch.FloatTensor:
        """
        Making a forward pass of MTGNN layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Input feature tensor,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Input feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor or None) - Predefined adjacency matrix.
            * **idx** (Pytorch LongTensor) - Input indices.
            * **training** (bool) - Whether in traning mode.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence tensor,
                with shape (batch_size, seq_len, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Output feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
        """
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(X, A_tilde.transpose(1, 0))
        else:
            X = self._residual_conv(X)
        X = X + X_residual[:, :, :, -X.size(3):]
        X = self._normalization(X, idx)
        return X, X_skip


class MTGNN(nn.Module):
    """An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        gcn_true (bool): Whether to add graph convolution layer.
        build_adj (bool): Whether to construct adaptive adjacency matrix.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        dropout (float): Droupout rate.
        subgraph_size (int): Size of subgraph.
        node_dim (int): Dimension of nodes.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels.
        residual_channels (int): Residual channels.
        skip_channels (int): Skip channels.
        end_channels (int): End channels.
        seq_length (int): Length of input sequence.
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of layers.
        propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.
        tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(self, gcn_true: bool, build_adj: bool, gcn_depth: int, num_nodes: int, kernel_set: list, kernel_size: int, dropout: float, subgraph_size: int, node_dim: int, dilation_exponential: int, conv_channels: int, residual_channels: int, skip_channels: int, end_channels: int, seq_length: int, in_dim: int, out_dim: int, layers: int, propalpha: float, tanhalpha: float, layer_norm_affline: bool, xd: Optional[int]=None):
        super(MTGNN, self).__init__()
        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)
        self._mtgnn_layers = nn.ModuleList()
        self._graph_constructor = GraphConstructor(num_nodes, subgraph_size, node_dim, alpha=tanhalpha, xd=xd)
        self._set_receptive_field(dilation_exponential, kernel_size, layers)
        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(MTGNNLayer(dilation_exponential=dilation_exponential, rf_size_i=1, kernel_size=kernel_size, j=j, residual_channels=residual_channels, conv_channels=conv_channels, skip_channels=skip_channels, kernel_set=kernel_set, new_dilation=new_dilation, layer_norm_affline=layer_norm_affline, gcn_true=gcn_true, seq_length=seq_length, receptive_field=self._receptive_field, dropout=dropout, gcn_depth=gcn_depth, num_nodes=num_nodes, propalpha=propalpha))
            new_dilation *= dilation_exponential
        self._setup_conv(in_dim, skip_channels, end_channels, residual_channels, out_dim)
        self._reset_parameters()

    def _setup_conv(self, in_dim, skip_channels, end_channels, residual_channels, out_dim):
        self._start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        if self._seq_length > self._receptive_field:
            self._skip_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self._seq_length), bias=True)
            self._skip_conv_E = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self._seq_length - self._receptive_field + 1), bias=True)
        else:
            self._skip_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self._receptive_field), bias=True)
            self._skip_conv_E = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self._end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self._end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(self, X_in: torch.FloatTensor, A_tilde: Optional[torch.FloatTensor]=None, idx: Optional[torch.LongTensor]=None, FE: Optional[torch.FloatTensor]=None) ->torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor, optional) - Predefined adjacency matrix, default None.
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE** (Pytorch FloatTensor, optional) - Static feature, default None.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        assert seq_len == self._seq_length, 'Input sequence length not equal to preset sequence length.'
        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(X_in, (self._receptive_field - self._seq_length, 0, 0, 0))
        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx, FE=FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE=FE)
        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(F.dropout(X_in, self._dropout, training=self.training))
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, self._idx, self.training)
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training)
        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        return X


class TemporalConv(nn.Module):
    """Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X: torch.FloatTensor) ->torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STConv(nn.Module):
    """Spatio-temporal convolution block using ChebConv Graph Convolutions.
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting"
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv)
    with kernel size k. Hence for an input sequence of length m,
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, K: int, normalization: str='sym', bias: bool=True):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._temporal_conv1 = TemporalConv(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self._graph_conv = ChebConv(in_channels=hidden_channels, out_channels=hidden_channels, K=K, normalization=normalization, bias=bias)
        self._temporal_conv2 = TemporalConv(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size)
        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None) ->torch.FloatTensor:
        """Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        T_0 = self._temporal_conv1(X)
        T = torch.zeros_like(T_0)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)
        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)
        return T


class UnitTCN(nn.Module):
    """
    Temporal Convolutional Block applied to nodes in the Two-Stream Adaptive Graph
    Convolutional Network as originally implemented in the
    `Github Repo <https://github.com/lshiwjx/2s-AGCN>`. For implementational details
    see https://arxiv.org/abs/1805.07694
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size. (default: :obj:`9`)
        stride (int): Temporal Convolutional kernel stride. (default: :obj:`1`)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=9, stride: int=1):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._conv_init(self.conv)
        self._bn_init(self.bn, 1)

    def _bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def _conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class UnitGCN(nn.Module):
    """
    Graph Convolutional Block applied to nodes in the Two-Stream Adaptive Graph Convolutional
    Network as originally implemented in the `Github Repo <https://github.com/lshiwjx/2s-AGCN>`.
    For implementational details see https://arxiv.org/abs/1805.07694.
    Temporal attention, spatial attention and channel-wise attention will be applied.
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        A (Tensor array): Adaptive Graph.
        coff_embedding (int, optional): Coefficient Embeddings. (default: :int:`4`)
        num_subset (int, optional): Subsets for adaptive graphs, see
        :math:`\\mathbf{A}, \\mathbf{B}, \\mathbf{C}` in https://arxiv.org/abs/1805.07694
        for details. (default: :int:`3`)
        adaptive (bool, optional): Apply Adaptive Graph Convolutions. (default: :obj:`True`)
        attention (bool, optional): Apply Attention. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.FloatTensor, coff_embedding: int=4, num_subset: int=3, adaptive: bool=True, attention: bool=True):
        super(UnitGCN, self).__init__()
        self.inter_c = out_channels // coff_embedding
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        self.A = A
        self.num_jpts = A.shape[-1]
        self.attention = attention
        self.adaptive = adaptive
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        if self.adaptive:
            self._init_adaptive_layers()
        else:
            self.A = Variable(self.A, requires_grad=False)
        if self.attention:
            self._init_attention_layers()
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self._init_conv_bn()

    def _bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def _conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        nn.init.constant_(conv.bias, 0)

    def _conv_branch_init(self, conv, branches):
        weight = conv.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
        nn.init.constant_(conv.bias, 0)

    def _init_conv_bn(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                self._bn_init(m, 1)
        self._bn_init(self.bn, 1e-06)
        for i in range(self.num_subset):
            self._conv_branch_init(self.conv_d[i], self.num_subset)

    def _init_attention_layers(self):
        self.conv_ta = nn.Conv1d(self.out_c, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        ker_jpt = self.num_jpts - 1 if not self.num_jpts % 2 else self.num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(self.out_c, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        rr = 2
        self.fc1c = nn.Linear(self.out_c, self.out_c // rr)
        self.fc2c = nn.Linear(self.out_c // rr, self.out_c)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def _init_adaptive_layers(self):
        self.PA = nn.Parameter(self.A)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(self.in_c, self.inter_c, 1))
            self.conv_b.append(nn.Conv2d(self.in_c, self.inter_c, 1))

    def _attentive_forward(self, y):
        se = y.mean(-2)
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y

    def _adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        A = self.PA
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))
            A1 = A[i] + A1 * self.alpha
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y

    def _non_adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        for i in range(self.num_subset):
            A1 = self.A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        if self.adaptive:
            y = self._adaptive_forward(x, y)
        else:
            y = self._non_adaptive_forward(x, y)
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        if self.attention:
            y = self._attentive_forward(y)
        return y


class GraphAAGCN:
    """
    Defining the Graph for the Two-Stream Adaptive Graph Convolutional Network.
    It's composed of the normalized inward-links, outward-links and
    self-links between the nodes as originally defined in the
    `authors repo  <https://github.com/lshiwjx/2s-AGCN/blob/master/graph/tools.py>`
    resulting in the shape of (3, num_nodes, num_nodes).
    Args:
        edge_index (Tensor array): Edge indices
        num_nodes (int): Number of nodes
    Return types:
            * **A** (PyTorch Float Tensor) - Three layer normalized adjacency matrix
    """

    def __init__(self, edge_index: list, num_nodes: int):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.A = self.get_spatial_graph(self.num_nodes)

    def get_spatial_graph(self, num_nodes):
        self_mat = torch.eye(num_nodes)
        inward_mat = torch.squeeze(to_dense_adj(self.edge_index))
        inward_mat_norm = F.normalize(inward_mat, dim=0, p=1)
        outward_mat = inward_mat.transpose(0, 1)
        outward_mat_norm = F.normalize(outward_mat, dim=0, p=1)
        adj_mat = torch.stack((self_mat, inward_mat_norm, outward_mat_norm))
        return adj_mat


class AAGCN(nn.Module):
    """Two-Stream Adaptive Graph Convolutional Network.

    For details see this paper: `"Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition." <https://arxiv.org/abs/1805.07694>`_.
    This implementation is based on the authors Github Repo https://github.com/lshiwjx/2s-AGCN.
    It's used by the author for classifying actions from sequences of 3D body joint coordinates.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        edge_index (PyTorch LongTensor): Graph edge indices.
        num_nodes (int): Number of nodes in the network.
        stride (int, optional): Time strides during temporal convolution. (default: :obj:`1`)
        residual (bool, optional): Applying residual connection. (default: :obj:`True`)
        adaptive (bool, optional): Adaptive node connection weights. (default: :obj:`True`)
        attention (bool, optional): Applying spatial-temporal-channel-attention.
        (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, edge_index: torch.LongTensor, num_nodes: int, stride: int=1, residual: bool=True, adaptive: bool=True, attention: bool=True):
        super(AAGCN, self).__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.graph = GraphAAGCN(self.edge_index, self.num_nodes)
        self.A = self.graph.A
        self.gcn1 = UnitGCN(in_channels, out_channels, self.A, adaptive=adaptive, attention=attention)
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        Making a forward pass.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods,
            with shape (B, F_in, T_in, N_nodes).

        Return types:
            * **X** (PyTorch FloatTensor)* - Sequence of node features,
            with shape (B, out_channels, T_in//stride, N_nodes).
        """
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class HeteroGCLSTM(torch.nn.Module):
    """An implementation similar to the Integrated Graph Convolutional Long Short Term
        Memory Cell for heterogeneous Graphs.

        Args:
            in_channels_dict (dict of keys=str and values=int): Dimension of each node's input features.
            out_channels (int): Number of output features.
            metadata (tuple): Metadata on node types and edge types in the graphs. Can be generated via PyG method
                :obj:`snapshot.metadata()` where snapshot is a single HeteroData object.
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels_dict: dict, out_channels: int, metadata: tuple, bias: bool=True):
        super(HeteroGCLSTM, self).__init__()
        self.in_channels_dict = in_channels_dict
        self.out_channels = out_channels
        self.metadata = metadata
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_i = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1), out_channels=self.out_channels, bias=self.bias) for edge_type in self.metadata[1]})
        self.W_i = nn.ParameterDict({node_type: Parameter(torch.Tensor(in_channels, self.out_channels)) for node_type, in_channels in self.in_channels_dict.items()})
        self.b_i = nn.ParameterDict({node_type: Parameter(torch.Tensor(1, self.out_channels)) for node_type in self.in_channels_dict})

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_f = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1), out_channels=self.out_channels, bias=self.bias) for edge_type in self.metadata[1]})
        self.W_f = nn.ParameterDict({node_type: Parameter(torch.Tensor(in_channels, self.out_channels)) for node_type, in_channels in self.in_channels_dict.items()})
        self.b_f = nn.ParameterDict({node_type: Parameter(torch.Tensor(1, self.out_channels)) for node_type in self.in_channels_dict})

    def _create_cell_state_parameters_and_layers(self):
        self.conv_c = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1), out_channels=self.out_channels, bias=self.bias) for edge_type in self.metadata[1]})
        self.W_c = nn.ParameterDict({node_type: Parameter(torch.Tensor(in_channels, self.out_channels)) for node_type, in_channels in self.in_channels_dict.items()})
        self.b_c = nn.ParameterDict({node_type: Parameter(torch.Tensor(1, self.out_channels)) for node_type in self.in_channels_dict})

    def _create_output_gate_parameters_and_layers(self):
        self.conv_o = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1), out_channels=self.out_channels, bias=self.bias) for edge_type in self.metadata[1]})
        self.W_o = nn.ParameterDict({node_type: Parameter(torch.Tensor(in_channels, self.out_channels)) for node_type, in_channels in self.in_channels_dict.items()})
        self.b_o = nn.ParameterDict({node_type: Parameter(torch.Tensor(1, self.out_channels)) for node_type in self.in_channels_dict})

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        for key in self.W_i:
            glorot(self.W_i[key])
        for key in self.W_f:
            glorot(self.W_f[key])
        for key in self.W_c:
            glorot(self.W_c[key])
        for key in self.W_o:
            glorot(self.W_o[key])
        for key in self.b_i:
            glorot(self.b_i[key])
        for key in self.b_f:
            glorot(self.b_f[key])
        for key in self.b_c:
            glorot(self.b_c[key])
        for key in self.b_o:
            glorot(self.b_o[key])

    def _set_hidden_state(self, x_dict, h_dict):
        if h_dict is None:
            h_dict = {node_type: torch.zeros(X.shape[0], self.out_channels) for node_type, X in x_dict.items()}
        return h_dict

    def _set_cell_state(self, x_dict, c_dict):
        if c_dict is None:
            c_dict = {node_type: torch.zeros(X.shape[0], self.out_channels) for node_type, X in x_dict.items()}
        return c_dict

    def _calculate_input_gate(self, x_dict, edge_index_dict, h_dict, c_dict):
        i_dict = {node_type: torch.matmul(X, self.W_i[node_type]) for node_type, X in x_dict.items()}
        conv_i = self.conv_i(h_dict, edge_index_dict)
        i_dict = {node_type: (I + conv_i[node_type]) for node_type, I in i_dict.items()}
        i_dict = {node_type: (I + self.b_i[node_type]) for node_type, I in i_dict.items()}
        i_dict = {node_type: torch.sigmoid(I) for node_type, I in i_dict.items()}
        return i_dict

    def _calculate_forget_gate(self, x_dict, edge_index_dict, h_dict, c_dict):
        f_dict = {node_type: torch.matmul(X, self.W_f[node_type]) for node_type, X in x_dict.items()}
        conv_f = self.conv_f(h_dict, edge_index_dict)
        f_dict = {node_type: (F + conv_f[node_type]) for node_type, F in f_dict.items()}
        f_dict = {node_type: (F + self.b_f[node_type]) for node_type, F in f_dict.items()}
        f_dict = {node_type: torch.sigmoid(F) for node_type, F in f_dict.items()}
        return f_dict

    def _calculate_cell_state(self, x_dict, edge_index_dict, h_dict, c_dict, i_dict, f_dict):
        t_dict = {node_type: torch.matmul(X, self.W_c[node_type]) for node_type, X in x_dict.items()}
        conv_c = self.conv_c(h_dict, edge_index_dict)
        t_dict = {node_type: (T + conv_c[node_type]) for node_type, T in t_dict.items()}
        t_dict = {node_type: (T + self.b_c[node_type]) for node_type, T in t_dict.items()}
        t_dict = {node_type: torch.tanh(T) for node_type, T in t_dict.items()}
        c_dict = {node_type: (f_dict[node_type] * C + i_dict[node_type] * t_dict[node_type]) for node_type, C in c_dict.items()}
        return c_dict

    def _calculate_output_gate(self, x_dict, edge_index_dict, h_dict, c_dict):
        o_dict = {node_type: torch.matmul(X, self.W_o[node_type]) for node_type, X in x_dict.items()}
        conv_o = self.conv_o(h_dict, edge_index_dict)
        o_dict = {node_type: (O + conv_o[node_type]) for node_type, O in o_dict.items()}
        o_dict = {node_type: (O + self.b_o[node_type]) for node_type, O in o_dict.items()}
        o_dict = {node_type: torch.sigmoid(O) for node_type, O in o_dict.items()}
        return o_dict

    def _calculate_hidden_state(self, o_dict, c_dict):
        h_dict = {node_type: (o_dict[node_type] * torch.tanh(C)) for node_type, C in c_dict.items()}
        return h_dict

    def forward(self, x_dict, edge_index_dict, h_dict=None, c_dict=None):
        """
        Making a forward pass. If the hidden state and cell state
        matrix dicts are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **x_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensors)* - Node features dicts. Can
                be obtained via PyG method :obj:`snapshot.x_dict` where snapshot is a single HeteroData object.
            * **edge_index_dict** *(Dictionary where keys=Tuples and values=PyTorch Long Tensors)* - Graph edge type
                and index dicts. Can be obtained via PyG method :obj:`snapshot.edge_index_dict`.
            * **h_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor, optional)* - Node type and
                hidden state matrix dict for all nodes.
            * **c_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor, optional)* - Node type and
                cell state matrix dict for all nodes.

        Return types:
            * **h_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor)* - Node type and
                hidden state matrix dict for all nodes.
            * **c_dict** *(Dictionary where keys=Strings and values=PyTorch Float Tensor)* - Node type and
                cell state matrix dict for all nodes.
        """
        h_dict = self._set_hidden_state(x_dict, h_dict)
        c_dict = self._set_cell_state(x_dict, c_dict)
        i_dict = self._calculate_input_gate(x_dict, edge_index_dict, h_dict, c_dict)
        f_dict = self._calculate_forget_gate(x_dict, edge_index_dict, h_dict, c_dict)
        c_dict = self._calculate_cell_state(x_dict, edge_index_dict, h_dict, c_dict, i_dict, f_dict)
        o_dict = self._calculate_output_gate(x_dict, edge_index_dict, h_dict, c_dict)
        h_dict = self._calculate_hidden_state(o_dict, c_dict)
        return h_dict, c_dict


class AVWGCN(nn.Module):
    """An implementation of the Node Adaptive Graph Convolution Layer.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, embedding_dimensions: int):
        super(AVWGCN, self).__init__()
        self.K = K
        self.weights_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, K, in_channels, out_channels))
        self.bias_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, out_channels))
        glorot(self.weights_pool)
        zeros(self.bias_pool)

    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor) ->torch.FloatTensor:
        """Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        number_of_nodes = E.shape[0]
        supports = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(number_of_nodes), supports]
        for _ in range(2, self.K):
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)
        supports = torch.stack(support_set, dim=0)
        W = torch.einsum('nd,dkio->nkio', E, self.weights_pool)
        bias = torch.matmul(E, self.bias_pool)
        X_G = torch.einsum('knm,bmc->bknc', supports, X)
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum('bnki,nkio->bno', X_G, W) + bias
        return X_G


class AGCRN(nn.Module):
    """An implementation of the Adaptive Graph Convolutional Recurrent Unit.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        number_of_nodes (int): Number of vertices.
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """

    def __init__(self, number_of_nodes: int, in_channels: int, out_channels: int, K: int, embedding_dimensions: int):
        super(AGCRN, self).__init__()
        self.number_of_nodes = number_of_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.embedding_dimensions = embedding_dimensions
        self._setup_layers()

    def _setup_layers(self):
        self._gate = AVWGCN(in_channels=self.in_channels + self.out_channels, out_channels=2 * self.out_channels, K=self.K, embedding_dimensions=self.embedding_dimensions)
        self._update = AVWGCN(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, embedding_dimensions=self.embedding_dimensions)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels)
        return H

    def forward(self, X: torch.FloatTensor, E: torch.FloatTensor, H: torch.FloatTensor=None) ->torch.FloatTensor:
        """Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node feature matrix.
            * **E** (PyTorch Float Tensor) - Node embedding matrix.
            * **H** (PyTorch Float Tensor) - Node hidden state matrix. Default is None.
        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        X_H = torch.cat((X, H), dim=-1)
        Z_R = torch.sigmoid(self._gate(X_H, E))
        Z, R = torch.split(Z_R, self.out_channels, dim=-1)
        C = torch.cat((X, Z * H), dim=-1)
        HC = torch.tanh(self._update(C, E))
        H = R * H + (1 - R) * HC
        return H


class A3TGCN(torch.nn.Module):
    """An implementation of the Attention Temporal Graph Convolutional Cell.
    For details see this paper: `"A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting." <https://arxiv.org/abs/2006.11583>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(self, in_channels: int, out_channels: int, periods: int, improved: bool=False, cached: bool=False, add_self_loops: bool=True):
        super(A3TGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved, cached=self.cached, add_self_loops=self.add_self_loops)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(X[:, :, period], edge_index, edge_weight, H)
        return H_accum


class DCRNN(torch.nn.Module):
    """An implementation of the Diffusion Convolutional Gated Recurrent Unit.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`)

    """

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool=True):
        super(DCRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = DConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = DConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = DConv(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels, K=self.K, bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([X, H], dim=1)
        Z = self.conv_x_z(Z, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([X, H], dim=1)
        R = self.conv_x_r(R, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([X, H * R], dim=1)
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) ->torch.FloatTensor:
        """Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **edge_index** (PyTorch Long Tensor) - Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional) - Edge weight vector.
            * **H** (PyTorch Float Tensor, optional) - Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class DyGrEncoder(torch.nn.Module):
    """An implementation of the integrated Gated Graph Convolution Long Short
    Term Memory Layer. For details see this paper: `"Predictive Temporal Embedding
    of Dynamic Graphs." <https://ieeexplore.ieee.org/document/9073186>`_

    Args:
        conv_out_channels (int): Number of output channels for the GGCN.
        conv_num_layers (int): Number of Gated Graph Convolutions.
        conv_aggr (str): Aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        lstm_out_channels (int): Number of LSTM channels.
        lstm_num_layers (int): Number of neurons in LSTM.
    """

    def __init__(self, conv_out_channels: int, conv_num_layers: int, conv_aggr: str, lstm_out_channels: int, lstm_num_layers: int):
        super(DyGrEncoder, self).__init__()
        assert conv_aggr in ['mean', 'add', 'max'], 'Wrong aggregator.'
        self.conv_out_channels = conv_out_channels
        self.conv_num_layers = conv_num_layers
        self.conv_aggr = conv_aggr
        self.lstm_out_channels = lstm_out_channels
        self.lstm_num_layers = lstm_num_layers
        self._create_layers()

    def _create_layers(self):
        self.conv_layer = GatedGraphConv(out_channels=self.conv_out_channels, num_layers=self.conv_num_layers, aggr=self.conv_aggr, bias=True)
        self.recurrent_layer = LSTM(input_size=self.conv_out_channels, hidden_size=self.lstm_out_channels, num_layers=self.lstm_num_layers)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, C: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If the hidden state and cell state matrices are
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H_tilde** *(PyTorch Float Tensor)* - Output matrix for all nodes.
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H_tilde = self.conv_layer(X, edge_index, edge_weight)
        H_tilde = H_tilde[None, :, :]
        if H is None and C is None:
            H_tilde, (H, C) = self.recurrent_layer(H_tilde)
        elif H is not None and C is not None:
            H = H[None, :, :]
            C = C[None, :, :]
            H_tilde, (H, C) = self.recurrent_layer(H_tilde, (H, C))
        else:
            raise ValueError('Invalid hidden state and cell matrices.')
        H_tilde = H_tilde.squeeze()
        H = H.squeeze()
        C = C.squeeze()
        return H_tilde, H, C


class EvolveGCNH(torch.nn.Module):
    """An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\\mathbf{\\hat{A}}` as :math:`\\mathbf{A} + 2\\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\\mathbf{\\hat{D}}^{-1/2} \\mathbf{\\hat{A}}
            \\mathbf{\\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(self, num_of_nodes: int, in_channels: int, improved: bool=False, cached: bool=False, normalize: bool=True, add_self_loops: bool=True):
        super(EvolveGCNH, self).__init__()
        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.initial_weight)

    def _create_layers(self):
        self.ratio = self.in_channels / self.num_of_nodes
        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)
        self.recurrent_layer = GRU(input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1)
        self.conv_layer = GCNConv_Fixed_W(in_channels=self.in_channels, out_channels=self.in_channels, improved=self.improved, cached=self.cached, normalize=self.normalize, add_self_loops=self.add_self_loops)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        X_tilde = X_tilde[0][None, :, :]
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[None, :, :]
        X_tilde, W = self.recurrent_layer(X_tilde, W)
        X = self.conv_layer(W.squeeze(dim=0), X, edge_index, edge_weight)
        return X


class EvolveGCNO(torch.nn.Module):
    """An implementation of the Evolving Graph Convolutional without Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_
    Args:
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\\mathbf{\\hat{A}}` as :math:`\\mathbf{A} + 2\\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\\mathbf{\\hat{D}}^{-1/2} \\mathbf{\\hat{A}}
            \\mathbf{\\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, improved: bool=False, cached: bool=False, normalize: bool=True, add_self_loops: bool=True):
        super(EvolveGCNO, self).__init__()
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.initial_weight)

    def _create_layers(self):
        self.recurrent_layer = GRU(input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1)
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()
        self.conv_layer = GCNConv_Fixed_W(in_channels=self.in_channels, out_channels=self.in_channels, improved=self.improved, cached=self.cached, normalize=self.normalize, add_self_loops=self.add_self_loops)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[None, :, :]
        _, W = self.recurrent_layer(W, W)
        X = self.conv_layer(W.squeeze(dim=0), X, edge_index, edge_weight)
        return X


class GCLSTM(torch.nn.Module):
    """An implementation of the the Integrated Graph Convolutional Long Short Term
    Memory Cell. For details see this paper: `"GC-LSTM: Graph Convolution Embedded LSTM
    for Dynamic Link Prediction." <https://arxiv.org/abs/1812.04206>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: str='sym', bias: bool=True):
        super(GCLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_i = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_i = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_f = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_f = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_c = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_c = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_o = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.W_o = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.W_i)
        glorot(self.W_f)
        glorot(self.W_c)
        glorot(self.W_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = torch.matmul(X, self.W_i)
        I = I + self.conv_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = torch.matmul(X, self.W_f)
        F = F + self.conv_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = torch.matmul(X, self.W_c)
        T = T + self.conv_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = torch.matmul(X, self.W_o)
        O = O + self.conv_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, C: torch.FloatTensor=None, lambda_max: torch.Tensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C


class GConvGRU(torch.nn.Module):
    """An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: str='sym', bias: bool=True):
        super(GConvGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_z = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_r = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_h = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, lambda_max: torch.Tensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.


        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class GConvLSTM(torch.nn.Module):
    """An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2} \\mathbf{A}
            \\mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1} \\mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: str='sym', bias: bool=True):
        super(GConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_i = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_f = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_c = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = ChebConv(in_channels=self.in_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.conv_h_o = ChebConv(in_channels=self.out_channels, out_channels=self.out_channels, K=self.K, normalization=self.normalization, bias=self.bias)
        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.w_c_i * C
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.w_c_f * C
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.w_c_o * C
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, C: torch.FloatTensor=None, lambda_max: torch.Tensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C


class LRGCN(torch.nn.Module):
    """An implementation of the Long Short Term Memory Relational
    Graph Convolution Layer. For details see this paper: `"Predicting Path
    Failure In Time-Evolving Graphs." <https://arxiv.org/abs/1905.03994>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases.
    """

    def __init__(self, in_channels: int, out_channels: int, num_relations: int, num_bases: int):
        super(LRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self._create_layers()

    def _create_input_gate_layers(self):
        self.conv_x_i = RGCNConv(in_channels=self.in_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)
        self.conv_h_i = RGCNConv(in_channels=self.out_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)

    def _create_forget_gate_layers(self):
        self.conv_x_f = RGCNConv(in_channels=self.in_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)
        self.conv_h_f = RGCNConv(in_channels=self.out_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)

    def _create_cell_state_layers(self):
        self.conv_x_c = RGCNConv(in_channels=self.in_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)
        self.conv_h_c = RGCNConv(in_channels=self.out_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)

    def _create_output_gate_layers(self):
        self.conv_x_o = RGCNConv(in_channels=self.in_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)
        self.conv_h_o = RGCNConv(in_channels=self.out_channels, out_channels=self.out_channels, num_relations=self.num_relations, num_bases=self.num_bases)

    def _create_layers(self):
        self._create_input_gate_layers()
        self._create_forget_gate_layers()
        self._create_cell_state_layers()
        self._create_output_gate_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_type, H, C):
        I = self.conv_x_i(X, edge_index, edge_type)
        I = I + self.conv_h_i(H, edge_index, edge_type)
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_type, H, C):
        F = self.conv_x_f(X, edge_index, edge_type)
        F = F + self.conv_h_f(H, edge_index, edge_type)
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_type, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_type)
        T = T + self.conv_h_c(H, edge_index, edge_type)
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_type, H, C):
        O = self.conv_x_o(X, edge_index, edge_type)
        O = O + self.conv_h_o(H, edge_index, edge_type)
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor, H: torch.FloatTensor=None, C: torch.FloatTensor=None) ->torch.FloatTensor:
        """
        Making a forward pass. If the hidden state and cell state matrices are
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_type** *(PyTorch Long Tensor)* - Edge type vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_type, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_type, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_type, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_type, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C


class MPNNLSTM(nn.Module):
    """An implementation of the Message Passing Neural Network with Long Short Term Memory.
    For details see this paper: `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_

    Args:
        in_channels (int): Number of input features.
        hidden_size (int): Dimension of hidden representations.
        num_nodes (int): Number of nodes in the network.
        window (int): Number of past samples included in the input.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels: int, hidden_size: int, num_nodes: int, window: int, dropout: float):
        super(MPNNLSTM, self).__init__()
        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels
        self._create_parameters_and_layers()

    def _create_parameters_and_layers(self):
        self._convolution_1 = GCNConv(self.in_channels, self.hidden_size)
        self._convolution_2 = GCNConv(self.hidden_size, self.hidden_size)
        self._batch_norm_1 = nn.BatchNorm1d(self.hidden_size)
        self._batch_norm_2 = nn.BatchNorm1d(self.hidden_size)
        self._recurrent_1 = nn.LSTM(2 * self.hidden_size, self.hidden_size, 1)
        self._recurrent_2 = nn.LSTM(self.hidden_size, self.hidden_size, 1)

    def _graph_convolution_1(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_1(X, edge_index, edge_weight))
        X = self._batch_norm_1(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def _graph_convolution_2(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_2(X, edge_index, edge_weight))
        X = self._batch_norm_2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor) ->torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.

        Arg types:
            * **X** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.

        Return types:
            *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
        """
        R = list()
        S = X.view(-1, self.window, self.num_nodes, self.in_channels)
        S = torch.transpose(S, 1, 2)
        S = S.reshape(-1, self.window, self.in_channels)
        O = [S[:, 0, :]]
        for l in range(1, self.window):
            O.append(S[:, l, self.in_channels - 1].unsqueeze(1))
        S = torch.cat(O, dim=1)
        X = self._graph_convolution_1(X, edge_index, edge_weight)
        R.append(X)
        X = self._graph_convolution_2(X, edge_index, edge_weight)
        R.append(X)
        X = torch.cat(R, dim=1)
        X = X.view(-1, self.window, self.num_nodes, X.size(1))
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.size(3))
        X, (H_1, _) = self._recurrent_1(X)
        X, (H_2, _) = self._recurrent_2(X)
        H = torch.cat([H_1[0, :, :], H_2[0, :, :], S], dim=1)
        return H


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AggregateTemporalNodeFeatures,
     lambda: ([], {'item_embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2D,
     lambda: ([], {'input_dims': 4, 'output_dims': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DilatedInception,
     lambda: ([], {'c_in': 4, 'c_out': 4, 'kernel_set': [4, 4], 'dilation_factor': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GatedFusion,
     lambda: ([], {'D': 4, 'bn_decay': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalGatedUpdater,
     lambda: ([], {'items_total': 4, 'item_embedding': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MTGNN,
     lambda: ([], {'gcn_true': 4, 'build_adj': 4, 'gcn_depth': 1, 'num_nodes': 4, 'kernel_set': [4, 4], 'kernel_size': 4, 'dropout': 0.5, 'subgraph_size': 4, 'node_dim': 4, 'dilation_exponential': 1, 'conv_channels': 4, 'residual_channels': 4, 'skip_channels': 4, 'end_channels': 4, 'seq_length': 4, 'in_dim': 4, 'out_dim': 4, 'layers': 1, 'propalpha': 4, 'tanhalpha': 4, 'layer_norm_affline': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedSelfAttention,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (TemporalConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformAttention,
     lambda: ([], {'K': 4, 'd': 4, 'bn_decay': 4}),
     lambda: ([torch.rand([4, 16, 16, 16]), torch.rand([4, 16, 16, 16]), torch.rand([4, 16, 16, 16])], {}),
     False),
    (UnitGCN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'A': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UnitTCN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_benedekrozemberczki_pytorch_geometric_temporal(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

