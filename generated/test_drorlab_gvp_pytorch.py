import sys
_module = sys.modules[__name__]
del sys
gvp = _module
atom3d = _module
data = _module
models = _module
run_atom3d = _module
run_cpd = _module
setup = _module
test_equivariance = _module

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


import functools


from torch import nn


import torch.nn.functional as F


import random


import scipy


import math


import torch.nn as nn


import pandas as pd


import numpy as np


from torch.utils.data import IterableDataset


import torch.utils.data as data


from torch.distributions import Categorical


from functools import partial


import time


import sklearn.metrics as sk_metrics


from collections import defaultdict


import scipy.stats as stats


from sklearn.metrics import confusion_matrix


from scipy.spatial.transform import Rotation


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-08, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(self, in_dims, out_dims, h_dim=None, activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        """
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


def _merge(s, v):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


def _split(x, nv):
    """
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def tuple_cat(*args, dim=-1):
    """
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    """
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def tuple_sum(*args):
    """
    Sums any number of tuples (s, V) elementwise.
    """
    return tuple(map(sum, zip(*args)))


class GVPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(self, node_dims, edge_dims, n_message=3, n_feedforward=2, drop_rate=0.1, autoregressive=False, activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message, aggr='add' if autoregressive else 'mean', activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            dh = tuple_sum(self.conv(x, edge_index_forward, edge_attr_forward), self.conv(autoregressive_x, edge_index_backward, edge_attr_backward))
            count = scatter_add(torch.ones_like(dst), dst, dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)
        else:
            dh = self.conv(x, edge_index, edge_attr)
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


_DEFAULT_E_DIM = 32, 1


_DEFAULT_V_DIM = 100, 16


_NUM_ATOM_TYPES = 9


class BaseModel(nn.Module):
    """
    A base 5-layer GVP-GNN for all ATOM3D tasks, using GVPs with 
    vector gating as described in the manuscript. Takes in atomic-level
    structure graphs of type `torch_geometric.data.Batch`
    and returns a single scalar.
    
    This class should not be used directly. Instead, please use the
    task-specific models which extend BaseModel. (Some of these classes
    may be aliases of BaseModel.)
    
    :param num_rbf: number of radial bases to use in the edge embedding
    """

    def __init__(self, num_rbf=16):
        super().__init__()
        activations = F.relu, None
        self.embed = nn.Embedding(_NUM_ATOM_TYPES, _NUM_ATOM_TYPES)
        self.W_e = nn.Sequential(LayerNorm((num_rbf, 1)), GVP((num_rbf, 1), _DEFAULT_E_DIM, activations=(None, None), vector_gate=True))
        self.W_v = nn.Sequential(LayerNorm((_NUM_ATOM_TYPES, 0)), GVP((_NUM_ATOM_TYPES, 0), _DEFAULT_V_DIM, activations=(None, None), vector_gate=True))
        self.layers = nn.ModuleList(GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, activations=activations, vector_gate=True) for _ in range(5))
        ns, _ = _DEFAULT_V_DIM
        self.W_out = nn.Sequential(LayerNorm(_DEFAULT_V_DIM), GVP(_DEFAULT_V_DIM, (ns, 0), activations=activations, vector_gate=True))
        self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * ns, 1))

    def forward(self, batch, scatter_mean=True, dense=True):
        """
        Forward pass which can be adjusted based on task formulation.
        
        :param batch: `torch_geometric.data.Batch` with data attributes
                      as returned from a BaseTransform
        :param scatter_mean: if `True`, returns mean of final node embeddings
                             (for each graph), else, returns embeddings seperately
        :param dense: if `True`, applies final dense layer to reduce embedding
                      to a single scalar; else, returns the embedding
        """
        h_V = self.embed(batch.atoms)
        h_E = batch.edge_s, batch.edge_v
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        batch_id = batch.batch
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)
        out = self.W_out(h_V)
        if scatter_mean:
            out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense:
            out = self.dense(out).squeeze(-1)
        return out


class PPIModel(BaseModel):
    """
    GVP-GNN for the PPI task.
    
    Extends BaseModel to accept a tuple (batch1, batch2)
    of `torch_geometric.data.Batch` graphs, where each graph
    index in a batch is paired with the same graph index in the
    other batch.
    
    As noted in the manuscript, PPIModel uses the final alpha
    carbon embeddings instead of the graph mean embedding.
    
    Returns a single scalar for each graph pair which can be used as
    a logit in binary classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(2 * ns, 4 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(4 * ns, 1))

    def forward(self, batch):
        graph1, graph2 = batch
        out1, out2 = map(self._gnn_forward, (graph1, graph2))
        out = torch.cat([out1, out2], dim=-1)
        out = self.dense(out)
        return torch.sigmoid(out).squeeze(-1)

    def _gnn_forward(self, graph):
        out = super().forward(graph, scatter_mean=False, dense=False)
        return out[graph.ca_idx + graph.ptr[:-1]]


class LEPModel(BaseModel):
    """
    GVP-GNN for the LEP task.
    
    Extends BaseModel to accept a tuple (batch1, batch2)
    of `torch_geometric.data.Batch` graphs, where each graph
    index in a batch is paired with the same graph index in the
    other batch.
    
    Returns a single scalar for each graph pair which can be used as
    a logit in binary classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(2 * ns, 4 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(4 * ns, 1))

    def forward(self, batch):
        out1, out2 = map(self._gnn_forward, batch)
        out = torch.cat([out1, out2], dim=-1)
        out = self.dense(out)
        return torch.sigmoid(out).squeeze(-1)

    def _gnn_forward(self, graph):
        return super().forward(graph, dense=False)


class MSPModel(BaseModel):
    """
    GVP-GNN for the MSP task.
    
    Extends BaseModel to accept a tuple (batch1, batch2)
    of `torch_geometric.data.Batch` graphs, where each graph
    index in a batch is paired with the same graph index in the
    other batch.
    
    As noted in the manuscript, MSPModel uses the final embeddings
    averaged over the residue of interest instead of the entire graph.
    
    Returns a single scalar for each graph pair which can be used as
    a logit in binary classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(2 * ns, 4 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(4 * ns, 1))

    def forward(self, batch):
        out1, out2 = map(self._gnn_forward, batch)
        out = torch.cat([out1, out2], dim=-1)
        out = self.dense(out)
        return torch.sigmoid(out).squeeze(-1)

    def _gnn_forward(self, graph):
        out = super().forward(graph, scatter_mean=False, dense=False)
        out = out * graph.node_mask.unsqueeze(-1)
        out = torch_scatter.scatter_add(out, graph.batch, dim=0)
        count = torch_scatter.scatter_add(graph.node_mask, graph.batch)
        return out / count.unsqueeze(-1)


class RESModel(BaseModel):
    """
    GVP-GNN for the RES task.
    
    Extends BaseModel to output a 20-dim vector instead of a single
    scalar for each graph, which can be used as logits in 20-way
    classification.
    
    As noted in the manuscript, RESModel uses the final alpha
    carbon embeddings instead of the graph mean embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * ns, 20))

    def forward(self, batch):
        out = super().forward(batch, scatter_mean=False)
        return out[batch.ca_idx + batch.ptr[:-1]]


class CPDModel(torch.nn.Module):
    """
    GVP-GNN for structure-conditioned autoregressive 
    protein design as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 20 amino acids at each position in a `torch.Tensor` of 
    shape [n_nodes, 20].
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    """

    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, num_layers=3, drop_rate=0.1):
        super(CPDModel, self).__init__()
        self.W_v = nn.Sequential(GVP(node_in_dim, node_h_dim, activations=(None, None)), LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(GVP(edge_in_dim, edge_h_dim, activations=(None, None)), LayerNorm(edge_h_dim))
        self.encoder_layers = nn.ModuleList(GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers))
        self.W_s = nn.Embedding(20, 20)
        edge_h_dim = edge_h_dim[0] + 20, edge_h_dim[1]
        self.decoder_layers = nn.ModuleList(GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate, autoregressive=True) for _ in range(num_layers))
        self.W_out = GVP(node_h_dim, (20, 0), activations=(None, None))

    def forward(self, h_V, edge_index, h_E, seq):
        """
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        """
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        encoder_embeddings = h_V
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = torch.cat([h_E[0], h_S], dim=-1), h_E[1]
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
        logits = self.W_out(h_V)
        return logits

    def sample(self, h_V, edge_index, h_E, n_samples, temperature=0.1):
        """
        Samples sequences autoregressively from the distribution
        learned by the model.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param n_samples: number of samples
        :param temperature: temperature to use in softmax 
                            over the categorical distribution
        
        :return: int `torch.Tensor` of shape [n_samples, n_nodes] based on the
                 residue-to-int mapping of the original training data
        """
        with torch.no_grad():
            device = edge_index.device
            L = h_V[0].shape[0]
            h_V = self.W_v(h_V)
            h_E = self.W_e(h_E)
            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)
            h_V = h_V[0].repeat(n_samples, 1), h_V[1].repeat(n_samples, 1, 1)
            h_E = h_E[0].repeat(n_samples, 1), h_E[1].repeat(n_samples, 1, 1)
            edge_index = edge_index.expand(n_samples, -1, -1)
            offset = L * torch.arange(n_samples, device=device).view(-1, 1, 1)
            edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
            seq = torch.zeros(n_samples * L, device=device, dtype=torch.int)
            h_S = torch.zeros(n_samples * L, 20, device=device)
            h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
            for i in range(L):
                h_S_ = h_S[edge_index[0]]
                h_S_[edge_index[0] >= edge_index[1]] = 0
                h_E_ = torch.cat([h_E[0], h_S_], dim=-1), h_E[1]
                edge_mask = edge_index[1] % L == i
                edge_index_ = edge_index[:, edge_mask]
                h_E_ = tuple_index(h_E_, edge_mask)
                node_mask = torch.zeros(n_samples * L, device=device, dtype=torch.bool)
                node_mask[i::L] = True
                for j, layer in enumerate(self.decoder_layers):
                    out = layer(h_V_cache[j], edge_index_, h_E_, autoregressive_x=h_V_cache[0], node_mask=node_mask)
                    out = tuple_index(out, node_mask)
                    if j < len(self.decoder_layers) - 1:
                        h_V_cache[j + 1][0][i::L] = out[0]
                        h_V_cache[j + 1][1][i::L] = out[1]
                logits = self.W_out(out)
                seq[i::L] = Categorical(logits=logits / temperature).sample()
                h_S[i::L] = self.W_s(seq[i::L])
            return seq.view(n_samples, L)


class MQAModel(nn.Module):
    """
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    """

    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, seq_in=False, num_layers=3, drop_rate=0.1):
        super(MQAModel, self).__init__()
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = node_in_dim[0] + 20, node_in_dim[1]
        self.W_v = nn.Sequential(LayerNorm(node_in_dim), GVP(node_in_dim, node_h_dim, activations=(None, None)))
        self.W_e = nn.Sequential(LayerNorm(edge_in_dim), GVP(edge_in_dim, edge_h_dim, activations=(None, None)))
        self.layers = nn.ModuleList(GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers))
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0)))
        self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=drop_rate), nn.Linear(2 * ns, 1))

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        """
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        """
        if seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V[0], seq], dim=-1), h_V[1]
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        if batch is None:
            out = out.mean(dim=0, keepdims=True)
        else:
            out = scatter_mean(out, batch, dim=0)
        return self.dense(out).squeeze(-1) + 0.5


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Dropout,
     lambda: ([], {'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GVP,
     lambda: ([], {'in_dims': [4, 4], 'out_dims': [4, 4]}),
     lambda: ([(torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (LayerNorm,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (_VDropout,
     lambda: ([], {'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_drorlab_gvp_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

