import sys
_module = sys.modules[__name__]
del sys
main = _module
time = _module
type = _module
utils = _module
conf = _module
intro = _module
pygod = _module
generator = _module
outlier_generator = _module
metrics = _module
models = _module
adone = _module
anemone = _module
anomalous = _module
anomalydae = _module
base = _module
basic_nn = _module
cola = _module
conad = _module
dominant = _module
done = _module
gaan = _module
gcnae = _module
guide = _module
mlpae = _module
ocgnn = _module
one = _module
radar = _module
scan = _module
test_adone = _module
test_anemone = _module
test_anomalous = _module
test_anomalydae = _module
test_base = _module
test_basic_nn = _module
test_cola = _module
test_conad = _module
test_dominant = _module
test_done = _module
test_gaan = _module
test_gcnae = _module
test_guide = _module
test_metrics = _module
test_mlpae = _module
test_ocgnn = _module
test_one = _module
test_outlier_generator = _module
test_radar = _module
test_utility = _module
dataset = _module
utility = _module
version = _module
setup = _module

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


import warnings


import time


import numpy as np


import math


import random


import torch.nn as nn


import torch.nn.functional as F


from sklearn.utils.validation import check_is_fitted


import scipy.sparse as sp


from torch import nn


import copy


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


from typing import Callable


from torch import Tensor


from torch.nn import BatchNorm1d


from torch.nn import Identity


from torch.nn import Linear


from torch.nn import ModuleList


from copy import deepcopy


from torch.utils.data import DataLoader


from numpy.testing import assert_equal


from numpy.testing import assert_raises


from numpy.testing import assert_allclose


import copy as cp


from torch.utils.data import Dataset


import numbers


class MLP(torch.nn.Module):
    """Multilayer Perceptron (MLP) model.
    Adapted from PyG for upward compatibility
    There exists two ways to instantiate an :class:`MLP`:
    1. By specifying explicit channel sizes, *e.g.*,
       .. code-block:: python
          mlp = MLP([16, 32, 64, 128])
       creates a three-layer MLP with **differently** sized hidden layers.
    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,
       .. code-block:: python
          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)
       creates a three-layer MLP with **equally** sized hidden layers.
    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        batch_norm_kwargs (Dict[str, Any], optional): Arguments passed to
            :class:`torch.nn.BatchNorm1d` in case :obj:`batch_norm == True`.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the module will not
            learn additive biases. (default: :obj:`True`)
        relu_first (bool, optional): Deprecated in favor of :obj:`act_first`.
            (default: :obj:`False`)
    """

    def __init__(self, channel_list: Optional[Union[List[int], int]]=None, *, in_channels: Optional[int]=None, hidden_channels: Optional[int]=None, out_channels: Optional[int]=None, num_layers: Optional[int]=None, dropout: float=0.0, act: Callable=F.relu, batch_norm: bool=True, act_first: bool=False, batch_norm_kwargs: Optional[Dict[str, Any]]=None, bias: bool=True, relu_first: bool=False):
        super().__init__()
        act_first = act_first or relu_first
        batch_norm_kwargs = batch_norm_kwargs or {}
        if isinstance(channel_list, int):
            in_channels = channel_list
        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]
        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.act = act
        self.act_first = act_first
        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))
        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)
        self.reset_parameters()

    @property
    def in_channels(self) ->int:
        """Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) ->int:
        """Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) ->int:
        """The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) ->Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.act_first:
                x = self.act(x)
            x = norm(x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


class AdONE_Base(nn.Module):

    def __init__(self, x_dim, s_dim, hid_dim, num_layers, dropout, act):
        super(AdONE_Base, self).__init__()
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers
        self.attr_encoder = MLP(in_channels=x_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act)
        self.attr_decoder = MLP(in_channels=hid_dim, hidden_channels=hid_dim, out_channels=x_dim, num_layers=decoder_layers, dropout=dropout, act=act)
        self.struct_encoder = MLP(in_channels=s_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act)
        self.struct_decoder = MLP(in_channels=hid_dim, hidden_channels=hid_dim, out_channels=s_dim, num_layers=decoder_layers, dropout=dropout, act=act)
        self.neigh_diff = NeighDiff()
        self.discriminator = MLP(in_channels=hid_dim, hidden_channels=int(hid_dim / 2), out_channels=1, num_layers=2, dropout=dropout, act=torch.tanh)

    def forward(self, x, s, edge_index):
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()
        dis_a = torch.sigmoid(self.discriminator(h_a))
        dis_s = torch.sigmoid(self.discriminator(h_s))
        return x_, s_, h_a, h_s, dna, dns, dis_a, dis_s


class AvgReadout(nn.Module):

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class Contextual_Discriminator(nn.Module):

    def __init__(self, n_h, negsamp_round):
        super(Contextual_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


class MaxReadout(nn.Module):

    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):

    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class Patch_Discriminator(nn.Module):

    def __init__(self, n_h, negsamp_round):
        super(Patch_Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_ano, h_unano, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_unano, h_ano))
        h_mi = h_ano
        for _ in range(self.negsamp_round):
            h_mi = torch.cat((h_mi[-2:-1, :], h_mi[:-1, :]), 0)
            scs.append(self.f_k(h_unano, h_mi))
        logits = torch.cat(tuple(scs))
        return logits


class WSReadout(nn.Module):

    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class ANEMONE_Base(nn.Module):

    def __init__(self, n_in, n_h, activation, negsamp_round_patch, negsamp_round_context, readout):
        super(ANEMONE_Base, self).__init__()
        self.read_mode = readout
        self.gcn_context = GCN(n_in, n_h, activation)
        self.gcn_patch = GCN(n_in, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
        self.c_disc = Contextual_Discriminator(n_h, negsamp_round_context)
        self.p_disc = Patch_Discriminator(n_h, negsamp_round_patch)

    def forward(self, seq1, adj, sparse=False, samp_bias1=None, samp_bias2=None):
        h_1 = self.gcn_context(seq1, adj, sparse)
        h_2 = self.gcn_patch(seq1, adj, sparse)
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, :-1, :])
            h_mv = h_1[:, -1, :]
            h_unano = h_2[:, -1, :]
            h_ano = h_2[:, -2, :]
        else:
            c = self.read(h_1[:, :-1, :], h_1[:, -2:-1, :])
            h_mv = h_1[:, -1, :]
            h_unano = h_2[:, -1, :]
            h_ano = h_2[:, -2, :]
        ret1 = self.c_disc(c, h_mv, samp_bias1, samp_bias2)
        ret2 = self.p_disc(h_ano, h_unano, samp_bias1, samp_bias2)
        return ret1, ret2


class Discriminator(nn.Module):

    def __init__(self, n_h, negsamp_ratio):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_ratio = negsamp_ratio

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_ratio):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


class ANOMALOUS_Base(nn.Module):

    def __init__(self, w, r):
        super(ANOMALOUS_Base, self).__init__()
        self.w = nn.Parameter(w)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return x @ self.w @ x, self.r


class AttributeAE(nn.Module):
    """
    Attribute Autoencoder in AnomalyDAE model: the encoder
    employs two non-linear feature transform to the node attribute
    x. The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to
    reconstruct the original node attribute.

    Parameters
    ----------
    in_dim:  int
        dimension of the input number of nodes
    embed_dim: int
        the latent representation dimension of node
        (after the first linear layer)
    out_dim:  int
        the output dim after two linear layers
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self, in_dim, embed_dim, out_dim, dropout, act):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, h):
        x = self.act(self.dense1(x.T))
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout)
        x = h @ x.T
        return x


class StructureAE(nn.Module):
    """
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent
    representation with the linear layer, and a graph attention
    layer produces an embedding with weight importance of node
    neighbors. Finally, the decoder reconstructs the final embedding
    to the original.

    Parameters
    ----------
    in_dim: int
        input dimension of node data
    embed_dim: int
        the latent representation dimension of node
       (after the first linear layer)
    out_dim: int
        the output dim after the graph attention layer
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    embed_x : torch.Tensor
        Embed nodes after the attention layer
    """

    def __init__(self, in_dim, embed_dim, out_dim, dropout, act):
        super(StructureAE, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim)
        self.attention_layer = GATConv(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index):
        x = self.act(self.dense(x))
        x = F.dropout(x, self.dropout)
        h = self.attention_layer(x, edge_index)
        s_ = torch.sigmoid(h @ h.T)
        return s_, h


class AnomalyDAE_Base(nn.Module):
    """
    AnomalyDAE_Base is an anomaly detector consisting of a structure
    autoencoder and an attribute reconstruction autoencoder.

    Parameters
    ----------
    in_node_dim : int
         Dimension of input feature
    in_num_dim: int
         Dimension of the input number of nodes
    embed_dim:: int
         Dimension of the embedding after the first reduced linear
         layer (D1)
    out_dim : int
         Dimension of final representation
    dropout : float, optional
        Dropout rate of the model
        Default: 0
    act: F, optional
         Choice of activation function
    """

    def __init__(self, in_node_dim, in_num_dim, embed_dim, out_dim, dropout, act):
        super(AnomalyDAE_Base, self).__init__()
        self.num_center_nodes = in_num_dim
        self.structure_ae = StructureAE(in_node_dim, embed_dim, out_dim, dropout, act)
        self.attribute_ae = AttributeAE(self.num_center_nodes, embed_dim, out_dim, dropout, act)

    def forward(self, x, edge_index, batch_size):
        s_, h = self.structure_ae(x, edge_index)
        if batch_size < self.num_center_nodes:
            x = F.pad(x, (0, 0, 0, self.num_center_nodes - batch_size))
        x_ = self.attribute_ae(x[:self.num_center_nodes], h)
        return x_, s_


class Vanilla_GCN(torch.nn.Module):

    def __init__(self, in_ft, out_ft, act, bias=True):
        super(Vanilla_GCN, self).__init__()
        self.fc = torch.nn.Linear(in_ft, out_ft, bias=False)
        self.act = torch.nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class CoLA_Base(nn.Module):

    def __init__(self, n_in, n_h, activation, negsamp_round, readout, subgraph_size, device):
        super(CoLA_Base, self).__init__()
        self.n_in = n_in
        self.subgraph_size = subgraph_size
        self.device = device
        self.readout = readout
        self.gcn = GCN(n_in, n_h, activation)
        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, x, adj, idx, subgraphs, batch_size, sparse=False):
        batch_adj = []
        batch_feature = []
        added_adj_zero_row = torch.zeros((batch_size, 1, self.subgraph_size))
        added_adj_zero_col = torch.zeros((batch_size, self.subgraph_size + 1, 1))
        added_adj_zero_col[:, -1, :] = 1.0
        added_feat_zero_row = torch.zeros((batch_size, 1, self.n_in))
        for i in idx:
            cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
            cur_feat = x[:, subgraphs[i], :]
            batch_adj.append(cur_adj)
            batch_feature.append(cur_feat)
        batch_adj = torch.cat(batch_adj)
        batch_adj = torch.cat((batch_adj, added_adj_zero_row), dim=1)
        batch_adj = torch.cat((batch_adj, added_adj_zero_col), dim=2)
        batch_feature = torch.cat(batch_feature)
        batch_feature = torch.cat((batch_feature[:, :-1, :], added_feat_zero_row, batch_feature[:, -1:, :]), dim=1)
        h_1 = self.gcn(batch_feature, batch_adj, sparse)
        if self.readout == 'max':
            h_mv = h_1[:, -1, :]
            c = torch.max(h_1[:, :-1, :], 1).values
        elif self.readout == 'min':
            h_mv = h_1[:, -1, :]
            c = torch.min(h_1[:, :-1, :], 1).values
        elif self.readout == 'avg':
            h_mv = h_1[:, -1, :]
            c = torch.mean(h_1[:, :-1, :], 1)
        elif self.readout == 'weighted_sum':
            seq, query = h_1[:, :-1, :], h_1[:, -2:-1, :]
            query = query.permute(0, 2, 1)
            sim = torch.matmul(seq, query)
            sim = F.softmax(sim, dim=1)
            sim = sim.repeat(1, 1, 64)
            out = torch.mul(seq, sim)
            c = torch.sum(out, 1)
            h_mv = h_1[:, -1, :]
        ret = self.disc(c, h_mv)
        return ret


class CONAD_Base(nn.Module):

    def __init__(self, in_dim, hid_dim, num_layers, dropout, act):
        super(CONAD_Base, self).__init__()
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers
        self.shared_encoder = GCN(in_channels=in_dim, hidden_channels=hid_dim, num_layers=encoder_layers, out_channels=hid_dim, dropout=dropout, act=act)
        self.attr_decoder = GCN(in_channels=hid_dim, hidden_channels=hid_dim, num_layers=decoder_layers, out_channels=in_dim, dropout=dropout, act=act)
        self.struct_decoder = GCN(in_channels=hid_dim, hidden_channels=hid_dim, num_layers=decoder_layers - 1, out_channels=in_dim, dropout=dropout, act=act)

    def embed(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        return h

    def reconstruct(self, h, edge_index):
        x_ = self.attr_decoder(h, edge_index)
        h_ = self.struct_decoder(h, edge_index)
        s_ = h_ @ h_.T
        return x_, s_

    def forward(self, x, edge_index):
        h = self.embed(x, edge_index)
        x_, s_ = self.reconstruct(h, edge_index)
        return x_, s_


class DOMINANT_Base(nn.Module):

    def __init__(self, in_dim, hid_dim, num_layers, dropout, act):
        super(DOMINANT_Base, self).__init__()
        encoder_layers = int(num_layers / 2)
        decoder_layers = num_layers - encoder_layers
        self.shared_encoder = GCN(in_channels=in_dim, hidden_channels=hid_dim, num_layers=encoder_layers, out_channels=hid_dim, dropout=dropout, act=act)
        self.attr_decoder = GCN(in_channels=hid_dim, hidden_channels=hid_dim, num_layers=decoder_layers, out_channels=in_dim, dropout=dropout, act=act)
        self.struct_decoder = GCN(in_channels=hid_dim, hidden_channels=hid_dim, num_layers=decoder_layers - 1, out_channels=in_dim, dropout=dropout, act=act)

    def forward(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        x_ = self.attr_decoder(h, edge_index)
        h_ = self.struct_decoder(h, edge_index)
        s_ = h_ @ h_.T
        return x_, s_


class DONE_Base(nn.Module):

    def __init__(self, x_dim, s_dim, hid_dim, num_layers, dropout, act):
        super(DONE_Base, self).__init__()
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers
        self.attr_encoder = MLP(in_channels=x_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act)
        self.attr_decoder = MLP(in_channels=hid_dim, hidden_channels=hid_dim, out_channels=x_dim, num_layers=decoder_layers, dropout=dropout, act=act)
        self.struct_encoder = MLP(in_channels=s_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act)
        self.struct_decoder = MLP(in_channels=hid_dim, hidden_channels=hid_dim, out_channels=s_dim, num_layers=decoder_layers, dropout=dropout, act=act)
        self.neigh_diff = NeighDiff()

    def forward(self, x, s, edge_index):
        h_a = self.attr_encoder(x)
        x_ = self.attr_decoder(h_a)
        dna = self.neigh_diff(h_a, edge_index).squeeze()
        h_s = self.struct_encoder(s)
        s_ = self.struct_decoder(h_s)
        dns = self.neigh_diff(h_s, edge_index).squeeze()
        return x_, s_, h_a, h_s, dna, dns


class GAAN_Base(nn.Module):

    def __init__(self, in_dim, noise_dim, hid_dim, generator_layers, encoder_layers, dropout, act):
        super(GAAN_Base, self).__init__()
        self.generator = MLP(in_channels=noise_dim, hidden_channels=hid_dim, out_channels=in_dim, num_layers=generator_layers, dropout=dropout, act=act)
        self.encoder = MLP(in_channels=in_dim, hidden_channels=hid_dim, out_channels=hid_dim, num_layers=encoder_layers, dropout=dropout, act=act)

    def forward(self, x, noise, edge_index):
        x_ = self.generator(noise)
        z = self.encoder(x)
        z_ = self.encoder(x_)
        a = torch.sigmoid(z @ z.T)
        a_ = torch.sigmoid(z_ @ z_.T)
        return x_, a, a_


class GNA(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, act):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels, hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            s = self.act(s)
        return s


class GUIDE_Base(nn.Module):

    def __init__(self, a_dim, s_dim, a_hid, s_hid, num_layers, dropout, act):
        super(GUIDE_Base, self).__init__()
        self.attr_ae = GCN(in_channels=a_dim, hidden_channels=a_hid, num_layers=num_layers, out_channels=a_dim, dropout=dropout, act=act)
        self.struct_ae = GNA(in_channels=s_dim, hidden_channels=s_hid, num_layers=num_layers, out_channels=s_dim, dropout=dropout, act=act)

    def forward(self, x, s, edge_index):
        x_ = self.attr_ae(x, edge_index)
        s_ = self.struct_ae(s, edge_index)
        return x_, s_


class GCN_base(nn.Module):
    """
    Describe: Backbone GCN module.
    """

    def __init__(self, in_feats, n_hidden, n_layers, dropout, act):
        super(GCN_base, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_feats, n_hidden, bias=False))
        for i in range(n_layers):
            self.layers.append(GCNConv(n_hidden, n_hidden, bias=False))
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class Radar_Base(nn.Module):

    def __init__(self, w, r):
        super(Radar_Base, self).__init__()
        self.w = nn.Parameter(w)
        self.r = nn.Parameter(r)

    def forward(self, x):
        return self.w @ x, self.r


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttributeAE,
     lambda: ([], {'in_dim': 4, 'embed_dim': 4, 'out_dim': 4, 'dropout': 0.5, 'act': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AvgReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Contextual_Discriminator,
     lambda: ([], {'n_h': 4, 'negsamp_round': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Discriminator,
     lambda: ([], {'n_h': 4, 'negsamp_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GAAN_Base,
     lambda: ([], {'in_dim': 4, 'noise_dim': 4, 'hid_dim': 4, 'generator_layers': 1, 'encoder_layers': 1, 'dropout': 0.5, 'act': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MinReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Patch_Discriminator,
     lambda: ([], {'n_h': 4, 'negsamp_round': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Vanilla_GCN,
     lambda: ([], {'in_ft': 4, 'out_ft': 4, 'act': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_pygod_team_pygod(_paritybench_base):
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

