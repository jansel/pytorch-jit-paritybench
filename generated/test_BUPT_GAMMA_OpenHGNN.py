import sys
_module = sys.modules[__name__]
del sys
conf = _module
run_experiment = _module
custom_link_prediction_dataset = _module
custom_model = _module
custom_node_classification_dataset = _module
hpo = _module
main = _module
openhgnn = _module
auto = _module
hpo_space = _module
config = _module
LinkPredictionDataset = _module
NodeClassificationDataset = _module
RecommendationDataset = _module
dataset = _module
academic_graph = _module
adapter = _module
base_dataset = _module
gtn_dataset = _module
hgb_dataset = _module
knowledge_graph = _module
multigraph = _module
ohgb_dataset = _module
utils = _module
test = _module
experiment = _module
EmbedLayer = _module
GeneralGNNLayer = _module
GeneralHGNNLayer = _module
HeteroGraphConv = _module
HeteroLinear = _module
MetapathConv = _module
SkipConnection = _module
layers = _module
ATTConv = _module
SemanticConv = _module
macro_layer = _module
CompConv = _module
HGConv = _module
LSTM_conv = _module
micro_layer = _module
CompGCN = _module
DMGI = _module
GATNE = _module
GTN_sparse = _module
HAN = _module
HDE = _module
HGAT = _module
HGNN_AC = _module
HGSL = _module
HGT = _module
HGT_hetero = _module
HPN = _module
HeCo = _module
HeGAN = _module
HetGNN = _module
HetSANN = _module
HeteroMLP = _module
KGCN = _module
MAGNN = _module
MHNF = _module
Micro_layer = _module
Multi_level = _module
NARS = _module
NEW_model = _module
NSHE = _module
RGAT = _module
RGCN = _module
RHGNN = _module
RSHN = _module
SLiCE = _module
SimpleHGN = _module
SkipGram = _module
TransD = _module
TransE = _module
TransH = _module
TransR = _module
models = _module
base_model = _module
HeRec = _module
SkipGram = _module
emb = _module
fastGTN = _module
general_HGNN = _module
homo_GNN = _module
ieHGCN = _module
GATNE_sampler = _module
HGT_sampler = _module
HetGNN_sampler = _module
MAGNN_sampler = _module
RSHN_sampler = _module
SLiCE_sampler = _module
TransX_sampler = _module
sampler = _module
hde_sampler = _module
negative_sampler = _module
random_walk_sampler = _module
test_MAGNN_sampler = _module
test_mp2vec_sampler = _module
tasks = _module
base_task = _module
demo = _module
link_prediction = _module
node_classification = _module
recommendation = _module
DMGI_trainer = _module
DeepwalkTrainer = _module
GATNE_trainer = _module
HeCo_trainer = _module
HeGAN_trainer = _module
TransX_trainer = _module
trainerflow = _module
base_flow = _module
demo = _module
hde_trainer = _module
herec_trainer = _module
hetgnn_trainer = _module
hgt_trainer = _module
kgcn_trainer = _module
link_prediction = _module
mp2vec_trainer = _module
networkschema = _module
node_classification = _module
node_classification_ac = _module
nshe_trainer = _module
recommendation = _module
slice_trainer = _module
activation = _module
best_config = _module
dgl_graph = _module
evaluator = _module
logger = _module
sampler = _module
trainer = _module
utils = _module
visualize = _module
setup = _module
space4hgnn = _module
distribution = _module
rank = _module
generate_yaml = _module
gather_all_Csv = _module
utils = _module
test_academic_dataset = _module
test_hgb_dataset = _module
test_industrial_dataset = _module
test_ohgb_dataset = _module
test_utils = _module
run_experiments = _module
test_experiment = _module
test_visualize = _module

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


import torch as th


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import math


import random


import scipy.sparse as sp


from scipy import sparse


from scipy import io as sio


import torch


import pandas as pd


from torch import nn


from torch.nn import Parameter


from torch.nn.parameter import Parameter


from collections import OrderedDict


from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score


from sklearn.svm import LinearSVC


from torch.autograd import Variable


import numpy


import torch.sparse as sparse


from torch.utils.data import DataLoader


from abc import ABCMeta


from torch.nn import init


from functools import partial


from functools import reduce


import time


import scipy.sparse as ssp


from torch.utils.data import IterableDataset


from torch.serialization import save


from itertools import combinations


from torch.utils.data import Dataset


import warnings


import copy


from numpy import random


from sklearn.metrics import roc_auc_score


from abc import ABC


from abc import abstractmethod


from collections.abc import Mapping


import torch.optim as optim


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.sampler import BatchSampler


from torch.utils.data import TensorDataset


from typing import List


from sklearn.metrics import accuracy_score


from sklearn.metrics import auc


from sklearn.metrics import mean_squared_error


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import roc_curve


from sklearn.cluster import KMeans


from sklearn.metrics import normalized_mutual_info_score


from sklearn.metrics import adjusted_rand_score


from sklearn.metrics import ndcg_score


from sklearn.linear_model import LogisticRegression


import sklearn.metrics as Metric


from scipy.sparse import coo_matrix


from collections import Counter


class MyRGCN(nn.Module):

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroEmbedLayer(nn.Module):
    """
    Embedding layer for featureless heterograph.

    Parameters
    -----------
    n_nodes_dict : dict[str, int]
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size : int
        Dimension of embedding,
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    """

    def __init__(self, n_nodes_dict, embed_size, embed_name='embed', activation=None, dropout=0.0):
        super(HeteroEmbedLayer, self).__init__()
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.ParameterDict()
        for ntype, nodes in n_nodes_dict.items():
            embed = nn.Parameter(th.FloatTensor(nodes, self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self):
        """
        Returns
        -------
        The output embeddings.
        """
        out_feature = {}
        for key, embed in self.embeds.items():
            out_feature[key] = embed
        return out_feature

    def forward_nodes(self, nodes_dict):
        """

        Parameters
        ----------
        nodes_dict : dict[str, th.Tensor]
            Key of dict means node type, value of dict means idx of nodes.

        Returns
        -------
        out_feature : dict[str, th.Tensor]
            Output feature.
        """
        out_feature = {}
        for key, nid in nodes_dict.items():
            out_feature[key] = self.embeds[key][nid]
        return out_feature


class multi_Linear(nn.Module):

    def __init__(self, linear_list, bias=False):
        super(multi_Linear, self).__init__()
        self.encoder = nn.ModuleDict({})
        for linear in linear_list:
            self.encoder[linear[0]] = nn.Linear(in_features=linear[1], out_features=linear[2], bias=bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder.weight)

    def forward(self, name_linear, h):
        h = self.encoder[name_linear](h)
        return h


class multi_2Linear(nn.Module):

    def __init__(self, linear_list, bias=False):
        super(multi_2Linear, self).__init__()
        hidden_dim = 16
        self.hidden_layer = nn.ModuleDict({})
        self.output_layer = nn.ModuleDict({})
        for linear in linear_list:
            self.hidden_layer[linear[0]] = nn.Linear(in_features=linear[1], out_features=hidden_dim, bias=bias)
            self.output_layer[linear[0]] = nn.Linear(in_features=hidden_dim, out_features=linear[2], bias=bias)

    def forward(self, name_linear, h):
        h = F.relu(self.hidden_layer[name_linear](h))
        h = self.output_layer[name_linear](h)
        return h


class hetero_linear(nn.Module):

    def __init__(self, linear_list, bias=False):
        super(hetero_linear, self).__init__()
        self.encoder = multi_Linear(linear_list, bias)

    def forward(self, h_dict):
        h_out = {}
        for ntype, h in h_dict.items():
            h = self.encoder(ntype, h)
            h_out[ntype] = h
        return h_out


def Aggr_max(z):
    z = torch.stack(z, dim=1)
    return z.max(1)[0]


def Aggr_mean(z):
    z = torch.stack(z, dim=1)
    return z.mean(1)


def Aggr_sum(z):
    z = torch.stack(z, dim=1)
    return z.sum(1)


class MetapathConv(nn.Module):
    """
    MetapathConv is an aggregation function based on meta-path, which is similar with `dgl.nn.pytorch.HeteroGraphConv`.
    We could choose Attention/ APPNP or any GraphConvLayer to aggregate node features.
    After that we will get embeddings based on different meta-paths and fusion them.

    .. math::
        \\mathbf{Z}=\\mathcal{F}(Z^{\\Phi_1},Z^{\\Phi_2},...,Z^{\\Phi_p})=\\mathcal{F}(f(H,\\Phi_1),f(H,\\Phi_2),...,f(H,\\Phi_p))

    where :math:`\\mathcal{F}` denotes semantic fusion function, such as semantic-attention. :math:`\\Phi_i` denotes meta-path and
    :math:`f` denotes the aggregation function, such as GAT, APPNP.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[tuple(meta-path)]]
        contain multiple meta-paths.
    mods : nn.ModuleDict
        aggregation function
    macro_func : callable aggregation func
        A semantic aggregation way, e.g. 'mean', 'max', 'sum' or 'attention'

    """

    def __init__(self, meta_paths_dict, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        self.mods = mods
        self.meta_paths_dict = meta_paths_dict
        self.SemanticConv = macro_func

    def forward(self, g_list, h_dict):
        """
        Parameters
        -----------
        g_list : list[dgl.DGLGraph]
            A list of DGLGraph extracted by metapaths.
        h : dict[str: torch.Tensor]
            The input features

        Returns
        --------
        h : dict[str: torch.Tensor]
            The output features dict
        """
        outputs = {g.dsttypes[0]: [] for s, g in g_list.items()}
        for mp, meta_path in self.meta_paths_dict.items():
            new_g = g_list[mp]
            h = h_dict[new_g.srctypes[0]]
            outputs[new_g.dsttypes[0]].append(self.mods[mp](new_g, h).flatten(1))
        rsts = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) != 0:
                rsts[ntype] = self.SemanticConv(ntype_outputs)
        return rsts


class SemanticAttention(nn.Module):

    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z, nty=None):
        if len(z) == 0:
            return None
        z = torch.stack(z, dim=1)
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class APPNPConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(APPNPConv, self).__init__()
        self.model = dgl.nn.pytorch.APPNPConv(k=3, alpha=0.5)
        self.lin = nn.Linear(dim_in, dim_out, bias)

    def forward(self, g, h):
        h = self.model(g, h)
        h = self.lin(h)
        return h


class GATConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = dgl.nn.pytorch.GATConv(dim_in, dim_out, num_heads=kwargs['num_heads'], bias=bias, allow_zero_in_degree=True)

    def forward(self, g, h):
        h = self.model(g, h).mean(1)
        return h


class GCNConv(nn.Module):

    def __init__(self):
        super(GCNConv, self).__init__()

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
                graph.srcdata['h'] = feat
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
        return rst


class GINConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        lin = nn.Sequential(nn.Linear(dim_in, dim_out, bias), nn.ReLU(), nn.Linear(dim_out, dim_out))
        self.model = dgl.nn.pytorch.GINConv(lin, 'max')

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class HgtConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(HgtConv, self).__init__()
        self.model = HGTConv(dim_in, dim_out, n_heads=kwargs['num_heads'], n_etypes=kwargs['num_etypes'], n_ntypes=kwargs['num_ntypes'])

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class SAGEConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = dgl.nn.pytorch.SAGEConv(dim_in, dim_out, aggregator_type='mean', bias=bias)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class SimpleHGNConv(nn.Module):
    """
    The SimpleHGN convolution layer.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """

    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0, negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes
        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)
        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))
        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)
        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer('residual', None)
        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0
        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, self.num_heads, self.edge_dim)
        row = g.edges()[0]
        col = g.edges()[1]
        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)
        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)
        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)
        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'), Fn.sum('m', 'emb'))
            h_output = g.ndata['emb'].view(-1, self.out_dim * self.num_heads)
        g.edata['alpha'] = edge_attention
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)
        return h_output


class SimpleConv(nn.Module):

    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SimpleConv, self).__init__()
        self.model = SimpleHGNConv(dim_in, dim_in, int(dim_out / kwargs['num_heads']), kwargs['num_heads'], kwargs['num_etypes'], beta=0.0)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


homo_layer_dict = {'gcnconv': GCNConv, 'sageconv': SAGEConv, 'gatconv': GATConv, 'ginconv': GINConv, 'simpleconv': SimpleConv, 'hgtconv': HgtConv, 'appnpconv': APPNPConv}


class MPConv(nn.Module):

    def __init__(self, name, dim_in, dim_out, bias=False, **kwargs):
        super(MPConv, self).__init__()
        macro_func = kwargs['macro_func']
        meta_paths = kwargs['meta_paths']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        elif macro_func == 'sum':
            macro_func = Aggr_sum
        elif macro_func == 'mean':
            macro_func = Aggr_mean
        elif macro_func == 'max':
            macro_func = Aggr_max
        self.model = MetapathConv(meta_paths, [homo_layer_dict[name](dim_in, dim_out, bias=bias, **kwargs) for _ in meta_paths], macro_func)
        self.meta_paths = meta_paths

    def forward(self, mp_g_list, h):
        h = self.model(mp_g_list, h)
        return h


class GeneralLayer(nn.Module):
    """General wrapper for layers"""

    def __init__(self, name, dim_in, dim_out, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        if kwargs.get('meta_paths') is not None:
            self.layer = MPConv(name, dim_in, dim_out, bias=not has_bn, **kwargs)
        else:
            self.layer = homo_layer_dict[name](dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, homo_g, h):
        h = self.layer(homo_g, h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class Linear(nn.Module):

    def __init__(self, dim_in, dim_out, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(Linear, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = nn.Linear(dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, h):
        h = self.layer(h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class MultiLinearLayer(nn.Module):

    def __init__(self, linear_list, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(MultiLinearLayer, self).__init__()
        for i in range(len(linear_list) - 1):
            d_in = linear_list[i]
            d_out = linear_list[i + 1]
            layer = Linear(d_in, d_out, dropout, act, has_bn, has_l2norm)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, h):
        for layer in self.children():
            h = layer(h)
        return h


class BatchNorm1dNode(nn.Module):
    """General wrapper for layers"""

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in)

    def forward(self, h):
        h = self.bn(h)
        return h


class RelationConv(nn.Module):

    def __init__(self, name, rel_names, dim_in, dim_out, bias=False, **kwargs):
        super(RelationConv, self).__init__()
        macro_func = kwargs['macro_func']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        self.model = HeteroGraphConv({rel: homo_layer_dict[name](dim_in, dim_out, bias=bias, **kwargs) for rel in rel_names}, aggregate=macro_func)

    def forward(self, g, h_dict):
        h_dict = self.model(g, h_dict)
        return h_dict


class HeteroGeneralLayer(nn.Module):
    """
    General wrapper for layers
    """

    def __init__(self, name, rel_names, dim_in, dim_out, dropout, act=None, has_bn=True, has_l2norm=False, **kwargs):
        super(HeteroGeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = RelationConv(name, rel_names, dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, g, h_dict):
        h_dict = self.layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(self.post_layer(batch_h), p=2, dim=-1)
        return h_dict


class GeneralLinear(nn.Module):
    """
    General Linear, combined with activation, normalization(batch and L2), dropout and so on.

    Parameters
    ------------
    in_features : int
        size of each input sample, which is fed into nn.Linear
    out_features : int
        size of each output sample, which is fed into nn.Linear
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    has_l2norm : bool
        If True, applies torch.nn.functional.normalize to the node features at last of forward(). Default: ``True``
    has_bn : bool
        If True, applies torch.nn.BatchNorm1d to the node features after applying nn.Linear.

    """

    def __init__(self, in_features, out_features, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(GeneralLinear, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = nn.Linear(in_features, out_features, bias=not has_bn)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(out_features))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch_h: torch.Tensor) ->torch.Tensor:
        """
        Apply Linear, BatchNorm1d, Dropout and normalize(if need).
        """
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h


class HeteroLinearLayer(nn.Module):
    """
    Transform feature with nn.Linear. In general, heterogeneous feature has different dimension as input.
    Even though they may have same dimension, they may have different semantic in every dimension.
    So we use a linear layer for each node type to map all node features to a shared feature space.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input dimension and output dimension.

    Examples
    ----------

    >>> import torch as th
    >>> linear_dict = {}
    >>> linear_dict['author'] = [110, 64]
    >>> linear_dict['paper'] = [128,64]
    >>> h_dict = {}
    >>> h_dict['author'] = th.tensor(10, 110)
    >>> h_dict['paper'] = th.tensor(5, 128)
    >>> layer = HeteroLinearLayer(linear_dict)
    >>> out_dict = layer(h_dict)

    """

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(HeteroLinearLayer, self).__init__()
        self.layer = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            self.layer[name] = GeneralLinear(in_features=linear_dim[0], out_features=linear_dim[1], act=act, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

    def forward(self, dict_h: dict) ->dict:
        """
        Parameters
        ----------
        dict_h : dict
            A dict of heterogeneous feature

        return dict_h
        """
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layer[name](batch_h)
        return new_h


class HeteroMLPLayer(nn.Module):
    """
    HeteroMLPLayer contains multiple GeneralLinears, different with HeteroLinearLayer.
    The latter contains only one layer.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input, hidden and output dimension.

    """

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, final_act=False, **kwargs):
        super(HeteroMLPLayer, self).__init__()
        self.layers = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            nn_list = []
            n_layer = len(linear_dim) - 1
            for i in range(n_layer):
                in_dim = linear_dim[i]
                out_dim = linear_dim[i + 1]
                if i == n_layer - 1:
                    if final_act:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                    else:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=None, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                else:
                    layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act, dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                nn_list.append(layer)
            self.layers[name] = nn.Sequential(*nn_list)

    def forward(self, dict_h):
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layers[name](batch_h)
        return new_h


class HeteroFeature(nn.Module):
    """
    This is a feature preprocessing component which is dealt with various heterogeneous feature situation.

    In general, we will face the following three situations.

        1. The dataset has not feature at all.

        2. The dataset has features in every node type.

        3. The dataset has features of a part of node types.

    To deal with that, we implement the HeteroFeature.In every situation, we can see that

        1. We will build embeddings for all node types.

        2. We will build linear layer for all node types.

        3. We will build embeddings for parts of node types and linear layer for parts of node types which have original feature.

    Parameters
    ----------
    h_dict: dict
        Input heterogeneous feature dict,
        key of dict means node type,
        value of dict means corresponding feature of the node type.
        It can be None if the dataset has no feature.
    n_nodes_dict: dict
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size: int
        Dimension of embedding, and used to assign to the output dimension of Linear which transform the original feature.
    need_trans: bool, optional
        A flag to control whether to transform original feature linearly. Default is ``True``.

    Attributes
    -----------
    embed_dict : nn.ParameterDict
        store the embeddings

    hetero_linear : HeteroLinearLayer
        A heterogeneous linear layer to transform original feature.
    """

    def __init__(self, h_dict, n_nodes_dict, embed_size, act=None, need_trans=True, all_feats=True):
        """

        @param h_dict:
        @param n_dict:
        @param embed_size:
        @param need_trans:
        @param all_feats:
        """
        super(HeteroFeature, self).__init__()
        self.n_nodes_dict = n_nodes_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans
        self.embed_dict = nn.ParameterDict()
        linear_dict = {}
        for ntype, n_nodes in self.n_nodes_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed = nn.Parameter(torch.FloatTensor(n_nodes, self.embed_size))
                    nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
                    self.embed_dict[ntype] = embed
            else:
                linear_dict[ntype] = [h.shape[1], self.embed_size]
        if need_trans:
            self.hetero_linear = HeteroLinearLayer(linear_dict, act=act)

    def forward(self):
        """
        return feature.

        Returns
        -------
        dict [str, th.Tensor]
            The output feature dictionary of feature.
        """
        out_dict = {}
        for ntype, _ in self.n_nodes_dict.items():
            if self.h_dict.get(ntype) is None:
                out_dict[ntype] = self.embed_dict[ntype]
        if self.need_trans:
            out_dict.update(self.hetero_linear(self.h_dict))
        else:
            out_dict.update(self.h_dict)
        return out_dict

    def forward_nodes(self, nodes_dict):
        out_feature = {}
        for ntype, nid in nodes_dict.items():
            if self.h_dict.get(ntype) is None:
                out_feature[ntype] = self.embed_dict[ntype][nid]
            elif self.need_trans:
                out_feature[ntype] = self.hetero_linear(self.h_dict)[ntype][nid]
            else:
                out_feature[ntype] = self.h_dict[ntype][nid]
        return out_feature


def HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return HeteroGeneralLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


class HGNNSkipBlock(nn.Module):
    """Skip block for HGNN"""

    def __init__(self, gnn_type, rel_names, dim_in, dim_out, num_layers, stage_type, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        self.act = act
        if stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, g, h):
        h_0 = h
        for layer in self.f:
            h = layer(g, h)
        out_h = {}
        for key, value in h_0.items():
            if self.stage_type == 'skipsum':
                out_h[key] = self.act(h[key] + h_0[key])
            elif self.stage_type == 'skipconcat':
                out_h[key] = self.act(torch.cat((h[key], h_0[key]), 1))
            else:
                raise ValueError('stage_type must in [skipsum, skipconcat]')
        return out_h


class HGNNStackStage(nn.Module):
    """Simple Stage that stack GNN layers"""

    def __init__(self, gnn_type, rel_names, stage_type, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h_dict):
        for layer in self.children():
            h_dict = layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(batch_h, p=2, dim=-1)
        return h_dict


class HGNNSkipStage(nn.Module):
    """ Stage with skip connections"""

    def __init__(self, gnn_type, rel_names, stage_type, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, 'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = HGNNSkipBlock(gnn_type, rel_names, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('block{}'.format(i), block)
        if stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h_dict):
        for layer in self.children():
            h_dict = layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(batch_h, p=2, dim=-1)
        return h_dict


def GNNLayer(gnn_type, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return GeneralLayer(gnn_type, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


class GNNSkipBlock(nn.Module):
    """
    Skip block for HGNN
    """

    def __init__(self, gnn_type, dim_in, dim_out, num_layers, stage_type, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(GNNLayer(gnn_type, dim_in, dim_out, dropout, None, has_bn, has_l2norm, **kwargs))
        else:
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, None, has_bn, has_l2norm, **kwargs))
        self.act = act
        if stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, g, h):
        h_0 = h
        for layer in self.f:
            h = layer(g, h)
        if self.stage_type == 'skipsum':
            h = h + h_0
        elif self.stage_type == 'skipconcat':
            h = torch.cat((h_0, h), 1)
        else:
            raise ValueError('stage_type must in [skipsum, skipconcat]')
        h = self.act(h)
        return h


class GNNStackStage(nn.Module):
    """Simple Stage that stack GNN layers"""

    def __init__(self, gnn_type, dim_in, dim_out, num_layers, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h):
        for layer in self.children():
            h = layer(g, h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=-1)
        return h


class GNNSkipStage(nn.Module):
    """ Stage with skip connections"""

    def __init__(self, gnn_type, stage_type, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, 'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(gnn_type, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('block{}'.format(i), block)
        if stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h):
        for layer in self.children():
            h = layer(g, h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=-1)
        return h


class ATTConv(nn.Module):
    """
    It is macro_layer of the models [HetGNN].
    It presents in the 3.3.2 Types Combination of the paper.
    
    In this framework, to make embedding dimension consistent and models tuning easy,
    we use the same dimension d for content embedding in Section 3.2,
    aggregated content embedding in Section 3.3, and output node embedding in Section 3.3.
        
    So just give one dim parameter.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    ntypes : list
        Node types.

    Note:
        We don't implement multi-heads version.

        atten_w is specific to the center node type, agnostic to the neighbor node type.
    """

    def __init__(self, ntypes, dim):
        super(ATTConv, self).__init__()
        self.ntypes = ntypes
        self.activation = nn.LeakyReLU()
        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hg, h_neigh, h_center):
        with hg.local_scope():
            if hg.is_block:
                h_dst = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_center.items()}
            else:
                h_dst = h_center
            n_types = len(self.ntypes) + 1
            outputs = {}
            for n in self.ntypes:
                h = h_dst[n]
                batch_size = h.shape[0]
                concat_h = []
                concat_emd = []
                for i in range(len(h_neigh[n])):
                    concat_h.append(th.cat((h, h_neigh[n][i]), 1))
                    concat_emd.append(h_neigh[n][i])
                concat_h.append(th.cat((h, h), 1))
                concat_emd.append(h)
                concat_h = th.hstack(concat_h).view(batch_size * n_types, self.dim * 2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, n_types)
                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)
                concat_emd = th.hstack(concat_emd).view(batch_size, n_types, self.dim)
                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                outputs[n] = weight_agg_batch
            return outputs


class MacroConv(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout : float, optional
        Dropout rate, defaults: ``0``.
    """

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float=0.0, negative_slope: float=0.2):
        super(MacroConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, input_dst: dict, relation_features: dict, edge_type_transformation_weight: nn.ParameterDict, central_node_transformation_weight: nn.ParameterDict, edge_types_attention_weight: nn.Parameter):
        """
        :param graph: dgl.DGLHeteroGraph
        :param input_dst: dict: {ntype: features}
        :param relation_features: dict: {(stype, etype, dtype): features}
        :param edge_type_transformation_weight: ParameterDict {etype: (n_heads * hidden_dim, n_heads * hidden_dim)}
        :param central_node_transformation_weight:  ParameterDict {ntype: (input_central_node_dim, n_heads * hidden_dim)}
        :param edge_types_attention_weight: Parameter (n_heads, 2 * hidden_dim)
        :return: output_features: dict, {"type": features}
        """
        output_features = {}
        for ntype in input_dst:
            if graph.number_of_dst_nodes(ntype) != 0:
                central_node_feature = input_dst[ntype]
                central_node_feature = torch.matmul(central_node_feature, central_node_transformation_weight[ntype]).view(-1, self._num_heads, self._out_feats)
                types_features = []
                for relation_tuple in relation_features:
                    stype, etype, dtype = relation_tuple
                    if dtype == ntype:
                        types_features.append(torch.matmul(relation_features[relation_tuple], edge_type_transformation_weight[etype]))
                types_features = torch.stack(types_features, dim=0)
                if types_features.shape[0] == 1:
                    output_features[ntype] = types_features.squeeze(dim=0)
                else:
                    types_features = types_features.view(types_features.shape[0], -1, self._num_heads, self._out_feats)
                    stacked_central_features = torch.stack([central_node_feature for _ in range(types_features.shape[0])], dim=0)
                    concat_features = torch.cat((stacked_central_features, types_features), dim=-1)
                    attention_scores = (edge_types_attention_weight * concat_features).sum(dim=-1, keepdim=True)
                    attention_scores = self.leaky_relu(attention_scores)
                    attention_scores = F.softmax(attention_scores, dim=0)
                    output_feature = (attention_scores * types_features).sum(dim=0)
                    output_feature = self.dropout(output_feature)
                    output_feature = output_feature.reshape(-1, self._num_heads * self._out_feats)
                    output_features[ntype] = output_feature
        return output_features


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return th.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    try:
        from torch import irfft
        from torch import rfft
    except ImportError:
        from torch.fft import irfft2
        from torch.fft import rfft2

        def rfft(x, d):
            t = rfft2(x, dim=-d)
            return th.stack((t.real, t.imag), -1)

        def irfft(x, d, signal_sizes):
            return irfft2(th.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=-d)
    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class CompConv(nn.Module):
    """
    Composition-based convolution was introduced in `Composition-based Multi-Relational Graph Convolutional Networks
    <https://arxiv.org/abs/1911.03082>`__
    and mathematically is defined as follows:

    Parameters
    ----------
    comp_fn : str, one of 'sub', 'mul', 'ccorr'
    """

    def __init__(self, comp_fn, norm='right', linear=False, in_feats=None, out_feats=None, bias=False, activation=None, _allow_zero_in_degree=False):
        super(CompConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right". But got "{}".'.format(norm))
        self._norm = norm
        self.comp_fn = comp_fn
        if self.comp_fn == 'sub':
            self.aggregate = fn.u_sub_e('h', '_edge_weight', out='comp_h')
        elif self.comp_fn == 'mul':
            self.aggregate = fn.u_mul_e('h', '_edge_weight', out='comp_h')
        elif self.comp_fn == 'ccorr':
            self.aggregate = lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['_edge_weight'])}
        else:
            raise Exception('Only supports sub, mul, and ccorr')
        if linear:
            if in_feats is None or out_feats is None:
                raise DGLError('linear is True, so you must specify the in/out feats')
            else:
                self.Linear = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('Linear', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self._activation = activation
        self._allow_zero_in_degree = _allow_zero_in_degree

    def forward(self, graph, feat, h_e, Linear=None):
        """
        Compute Composition-based  convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        Linear : a Linear nn.Module, optional
            Optional external weight tensor.
        h_e : torch.Tensor
            :math:`(1, D_{in})`
            means the edge type feature.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        The h_e is a tensor of size `(1, D_{in})`
        
        * Input shape: :math:`(N, *, \\text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \\text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Linear shape: :math:`(\\text{in_feats}, \\text{out_feats})`.
                """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, causing silent performance regression. Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run.')
            graph.edata['_edge_weight'] = h_e.expand(graph.num_edges(), -1)
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.apply_edges(self.aggregate)
            if Linear is not None:
                if self.Linear is not None:
                    raise DGLError('External Linear is provided while at the same time the module has defined its own Linear module. Please create the module with flag Linear=False.')
            else:
                Linear = self.Linear
            graph.edata['comp_h'] = Linear(graph.edata['comp_h'])
            graph.update_all(fn.copy_e('comp_h', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata['h']
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
            if self.bias is not None:
                rst = rst + self.bias
            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class LSTMConv(nn.Module):
    """
    Aggregate the neighbors with LSTM
    """

    def __init__(self, dim):
        super(LSTMConv, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.

        Note
        ----
        The LSTM module is using xavier initialization method for its weights.
        """
        self.lstm.reset_parameters()

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh


class CompGraphConvLayer(nn.Module):
    """One layer of simplified CompGCN."""

    def __init__(self, in_dim, out_dim, rel_names, comp_fn='sub', activation=None, batchnorm=False, dropout=0):
        super(CompGraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = activation
        self.batchnorm = batchnorm
        self.rel_names = rel_names
        self.dropout = nn.Dropout(dropout)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)
        self.W_R = nn.Linear(self.in_dim, self.out_dim)
        self.conv = dglnn.HeteroGraphConv({rel: CompConv(comp_fn=comp_fn, norm='right', _allow_zero_in_degree=True) for rel in rel_names})

    def forward(self, hg, n_in_feats, r_feats):
        """
        Compute one layer of composition transfer for one relation only in a
        homogeneous graph with bidirectional edges.
        """
        with hg.local_scope():
            wdict = {}
            for i, etype in enumerate(self.rel_names):
                if etype[:4] == 'rev-' or etype[-4:] == '-rev':
                    W = self.W_I
                else:
                    W = self.W_O
                wdict[etype] = {'Linear': W, 'h_e': r_feats[i + 1]}
            if hg.is_block:
                inputs_src = n_in_feats
                inputs_dst = {k: v[:hg.number_of_dst_nodes(k)] for k, v in n_in_feats.items()}
            else:
                inputs_src = inputs_dst = n_in_feats
            outputs = self.conv(hg, inputs_src, mod_kwargs=wdict)
            for n, emd in outputs.items():
                if self.comp_fn == 'sub':
                    h_self = self.W_S(inputs_dst[n] - r_feats[-1])
                elif self.comp_fn == 'mul':
                    h_self = self.W_S(inputs_dst[n] * r_feats[-1])
                elif self.comp_fn == 'ccorr':
                    h_self = self.W_S(ccorr(inputs_dst[n], r_feats[-1]))
                else:
                    raise Exception('Only supports sub, mul, and ccorr')
                h_self.add_(emd)
                if self.batchnorm:
                    if h_self.shape[0] > 1:
                        h_self = self.bn(h_self)
                n_out_feats = self.dropout(h_self)
                if self.actvation is not None:
                    n_out_feats = self.actvation(n_out_feats)
                outputs[n] = n_out_feats
        r_out_feats = self.W_R(r_feats)
        r_out_feats = self.dropout(r_out_feats)
        if self.actvation is not None:
            r_out_feats = self.actvation(r_out_feats)
        return outputs, r_out_feats


class Attention(nn.Module):

    def __init__(self, hidden_dim, num_mps, num_ndoes):
        super(Attention, self).__init__()
        self.num_mps = num_mps
        self.hidden_dim = hidden_dim
        self.num_nodes = num_ndoes
        self.A = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_mps)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.num_mps):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos, feat_neg, summary):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        summary, summary_attn = self.attn_summary(summary)
        return feat_pos, feat_neg, summary

    def attn_feature(self, features):
        features_attn = []
        for i in range(self.num_mps):
            features_attn.append(self.A[i](features[i].squeeze()))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features, 1).squeeze(0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.num_mps, self.num_nodes, self.hid_units).sum(0).unsqueeze(0)
        return features, features_attn

    def attn_summary(self, features):
        features_attn = []
        for i in range(self.num_mps):
            features_attn.append(self.A[i](features[i].squeeze()))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1).unsqueeze(1)
        features = torch.stack(features, 0)
        features_attn_expanded = features_attn.expand_as(features)
        features = (features * features_attn_expanded).sum(0).unsqueeze(0)
        return features, features_attn


class Discriminator(nn.Module):
    """
    A generator :math:`G` samples fake node embeddings from a continuous distribution. The distribution is Gaussian distribution:

    .. math::
        \\mathcal{N}(\\mathbf{e}_u^{G^T} \\mathbf{M}_r^G, \\mathbf{\\sigma}^2 \\mathbf{I})

    where :math:`e_u^G \\in \\mathbb{R}^{d \\times 1}` and :math:`M_r^G \\in \\mathbb{R}^{d \\times d}` denote the node embedding of :math:`u \\in \\mathcal{V}` and the relation matrix of :math:`r \\in \\mathcal{R}` for the generator.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\\mathbf{u}, \\mathbf{r}; \\mathbf{\\theta}^G) = f(\\mathbf{W_2}f(\\mathbf{W}_1 \\mathbf{e} + \\mathbf{b}_1) + \\mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is:

    .. math::
        L_1^D = \\mathbb{E}_{\\langle u,v,r\\rangle \\sim P_G} = -\\log D(e_v^u|u,r))

        L_2^D = \\mathbb{E}_{\\langle u,v\\rangle \\sim P_G, r' \\sim P_{R'}} = -\\log (1-D(e_v^u|u,r')))

        L_3^D = \\mathbb{E}_{\\langle u,v\\rangle \\sim P_G, e'_v \\sim G(u,r;\\theta^G)} = -\\log (1-D(e_v'|u,r)))

        L_G = L_1^D + L_2^D + L_2^D + \\lambda^D || \\theta^D ||_2^2

    where :math:`\\theta^D` denote all the learnable parameters in Discriminator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """

    def __init__(self, emb_size, hg):
        super().__init__()
        self.n_relation = len(hg.etypes)
        self.node_emb_dim = emb_size
        self.nodes_embedding = nn.ParameterDict()
        for nodes_type, nodes_emb in hg.ndata['h'].items():
            self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)
        self.relation_matrix = nn.ParameterDict()
        for et in hg.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)

    def forward(self, pos_hg, neg_hg1, neg_hg2, generate_neighbor_emb):
        """
        Parameters
        ----------
        pos_hg:
            sampled postive graph.
        neg_hg1:
            sampled negative graph with wrong relation.
        neg_hg2:
            sampled negative graph wtih wrong node.
        generate_neighbor_emb:
            generator node embeddings.
        """
        self.assign_node_data(pos_hg)
        self.assign_node_data(neg_hg1)
        self.assign_node_data(neg_hg2, generate_neighbor_emb)
        self.assign_edge_data(pos_hg)
        self.assign_edge_data(neg_hg1)
        self.assign_edge_data(neg_hg2)
        pos_score = self.score_pred(pos_hg)
        neg_score1 = self.score_pred(neg_hg1)
        neg_score2 = self.score_pred(neg_hg2)
        return pos_score, neg_score1, neg_score2

    def get_parameters(self):
        """
        return discriminator node embeddings and relation embeddings.
        """
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}, {k: self.relation_matrix[k] for k in self.relation_matrix.keys()}

    def score_pred(self, hg):
        """
        predict the discriminator score for sampled heterogeneous graph.
        """
        score_list = []
        with hg.local_scope():
            for et in hg.canonical_etypes:
                hg.apply_edges(lambda edges: {'s': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).reshape(hg.num_edges(et), 64)}, etype=et)
                if len(hg.edata['f']) == 0:
                    hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.dst['h'])}, etype=et)
                else:
                    hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['f'])}, etype=et)
                score = torch.sum(hg.edata['score'].pop(et), dim=1)
                score_list.append(score)
        return torch.cat(score_list)

    def assign_edge_data(self, hg):
        d = {}
        for et in hg.canonical_etypes:
            e = self.relation_matrix[et[1]]
            n = hg.num_edges(et)
            d[et] = e.expand(n, -1, -1)
        hg.edata['e'] = d

    def assign_node_data(self, hg, generate_neighbor_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if generate_neighbor_emb:
            hg.edata['f'] = generate_neighbor_emb


class AvgReadout(nn.Module):
    """
    Considering the efficiency of the method, we simply employ average pooling, computing the average of the set of embedding matrices

    .. math::
      \\begin{equation}
        \\mathbf{H}=\\mathcal{Q}\\left(\\left\\{\\mathbf{H}^{(r)} \\mid r \\in \\mathcal{R}\\right\\}\\right)=\\frac{1}{|\\mathcal{R}|} \\sum_{r \\in \\mathcal{R}} \\mathbf{H}^{(r)}
      \\end{equation}
    """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class LogReg(nn.Module):
    """
    Parameters
    ----------
    ft_in : int
        Size of hid_units
    nb_class : int
        The number of category's types
    """

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class NSLoss(nn.Module):

    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(torch.Tensor([((math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)) for k in range(num_nodes)]), dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1)))
        negs = torch.multinomial(self.sample_weights, self.num_sampled * n, replacement=True).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1).squeeze()
        loss = log_target + sum_log_sampled
        return -loss.sum() / n


class GTConv(nn.Module):
    """
        We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

        .. math::
            A_{1} = conv\\left(A ; W_{c}\\right)=\\sum_{R_{i} \\in R} w_{R_{i}} A_{R_{i}}

        where :math:`R_i \\subseteq \\mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results


class GTLayer(nn.Module):
    """
        CTLayer multiply each combination adjacency matrix :math:`l` times to a :math:`l-length`
        meta-paths adjacency matrix.

        The method to generate :math:`l-length` meta-path adjacency matrix can be described as:

        .. math::
            A_{(l)}=\\Pi_{i=1}^{l} A_{i}

        where :math:`A_{i}` is the combination adjacency matrix generated by GT conv.

        Parameters
        ----------
            in_channels: int
                The input dimension of GTConv which is numerically equal to the number of relations.
            out_channels: int
                The input dimension of GTConv which is numerically equal to the number of channel in GTN.
            first: bool
                If true, the first combination adjacency matrix multiply the combination adjacency matrix.

    """

    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [F.softmax(self.conv1.weight, dim=1).detach(), F.softmax(self.conv2.weight, dim=1).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [F.softmax(self.conv1.weight, dim=1).detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W


class HANLayer(nn.Module):
    """
    HAN layer.

    Parameters
    -----------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Attributes
    ------------
    _cached_graph : dgl.DGLHeteroGraph
        a cached graph
    _cached_coalesced_graph : list
        _cached_coalesced_graph list generated by *dgl.metapath_reachable_graph()*
    """

    def __init__(self, meta_paths_dict, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        self.gat_layers = nn.ModuleList()
        semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        mods = nn.ModuleDict({mp: GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu, allow_zero_in_degree=True) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        """
        Parameters
        -----------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, mp_value)
        h = self.model(self._cached_coalesced_graph, h)
        return h


class GNN(nn.Module):
    """
    Aggregate 2-hop neighbor.
    """

    def __init__(self, input_dim, output_dim, num_neighbor, use_bias=True):
        super(GNN, self).__init__()
        self.input_dim = int(input_dim)
        self.num_fea = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_neighbor = num_neighbor
        self.use_bias = use_bias
        self.linear1 = nn.Linear(self.input_dim * 2, 64)
        self.linear2 = nn.Linear(64 + self.num_fea, 64)
        self.linear3 = nn.Linear(64, self.output_dim)

    def forward(self, fea):
        node = fea[:, :self.num_fea]
        neigh1 = fea[:, self.num_fea:self.num_fea * (self.num_neighbor + 1)]
        neigh1 = torch.reshape(neigh1, [-1, self.num_neighbor, self.num_fea])
        neigh2 = fea[:, self.num_fea * (self.num_neighbor + 1):]
        neigh2 = torch.reshape(neigh2, [-1, self.num_neighbor, self.num_neighbor, self.num_fea])
        neigh2_agg = torch.mean(neigh2, dim=2)
        tmp = torch.cat([neigh1, neigh2_agg], dim=2)
        tmp = F.relu(self.linear1(tmp))
        emb = torch.cat([node, torch.mean(tmp, dim=1)], dim=1)
        emb = F.relu(self.linear2(emb))
        emb = F.relu(self.linear3(emb))
        return emb


class TypeAttention(nn.Module):
    """
    The type-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    ntypes: list
        the list of the node type in the graph
    slope: float
        the negative slope used in the LeakyReLU
    """

    def __init__(self, in_dim, ntypes, slope):
        super(TypeAttention, self).__init__()
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = in_dim
        self.mu_l = HeteroLinear(attn_vector, in_dim)
        self.mu_r = HeteroLinear(attn_vector, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, hg, h_dict):
        """
        The forward part of the TypeAttention.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        h_t = {}
        attention = {}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                with rel_graph.local_scope():
                    degs = rel_graph.out_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    feat_src = h_dict[srctype]
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    feat_src = feat_src * norm
                    rel_graph.srcdata['h'] = feat_src
                    rel_graph.update_all(Fn.copy_src('h', 'm'), Fn.sum(msg='m', out='h'))
                    rst = rel_graph.dstdata['h']
                    degs = rel_graph.in_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                    h_t[srctype] = rst
                    h_l = self.mu_l(h_dict)[dsttype]
                    h_r = self.mu_r(h_t)[srctype]
                    edge_attention = F.elu(h_l + h_r)
                    rel_graph.ndata['m'] = {dsttype: edge_attention, srctype: torch.zeros((rel_graph.num_nodes(ntype=srctype),))}
                    reverse_graph = dgl.reverse(rel_graph)
                    reverse_graph.apply_edges(Fn.copy_src('m', 'alpha'))
                hg.edata['alpha'] = {(srctype, etype, dsttype): reverse_graph.edata['alpha']}
            attention = edge_softmax(hg, hg.edata['alpha'])
        return attention


class NodeAttention(nn.Module):
    """
    The node-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    out_dim: int
        the output dimension
    slope: float
        the negative slope used in the LeakyReLU
    """

    def __init__(self, in_dim, out_dim, slope):
        super(NodeAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Mu_l = nn.Linear(in_dim, in_dim)
        self.Mu_r = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, g, x, ntype, etype, presorted=False):
        """
        The forward part of the NodeAttention.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        with g.local_scope():
            src = g.edges()[0]
            dst = g.edges()[1]
            h_l = self.Mu_l(x)[src]
            h_r = self.Mu_r(x)[dst]
            edge_attention = self.leakyrelu((h_l + h_r) * g.edata['alpha'])
            edge_attention = edge_softmax(g, edge_attention)
            g.edata['alpha'] = edge_attention
            g.srcdata['x'] = x
            g.update_all(Fn.u_mul_e('x', 'alpha', 'm'), Fn.sum('m', 'x'))
            h = g.ndata['x']
        return h


class AttentionLayer(nn.Module):
    """
    This is the attention process used in HGNN\\_AC. For more details, you can check here_.
    
    Parameters
    -------------------
    in_dim: int
        nodes' topological embedding dimension
    hidden_dim: int
        hidden dimension
    dropout: float
        the drop rate used in the attention
    activation: callable activation function
        the activation function used in HGNN_AC.  default: ``F.elu``
    """

    def __init__(self, in_dim, hidden_dim, dropout, activation, cuda=False):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_cuda = cuda
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_dim, hidden_dim).type(torch.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(torch.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        """
        This is the forward part of the attention process.
        
        Parameters
        --------------
        bias: matrix
            the processed adjacency matrix related to the source nodes
        emb_dest: matrix
            the embeddings of the destination nodes
        emb_src: matrix
            the embeddings of the source nodes
        feature_src: matrix
            the features of the source nodes
        
        Returns
        ------------
        features: matrix
            the new features of the nodes
        """
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)
        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(bias > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, feature_src)
        return self.activation(h_prime)


class MetricCalcLayer(nn.Module):
    """
    Calculate metric in equation 3 of paper.

    Parameters
    ----------
    nhid : int
        The dimension of mapped features in the graph generating procedure.
    """

    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        """
        Parameters
        ----------
        h : tensor
            The result of the Hadamard product in equation 3 of paper.
        """
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate a graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """
        Parameters
        ----------
        left_h : tensor
            The first input embedding matrix.
        right_h : tensor
            The second input embedding matrix.
        """

        def cos_sim(a, b, eps=1e-08):
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
            return sim_mt
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0]))
        s = torch.zeros((left_h.shape[0], right_h.shape[0]))
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-08
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GraphChannelAttLayer(nn.Module):
    """
    The graph channel attention layer in equation 7, 9 and 10 of paper.
    """

    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, adj_list):
        """
        Parameters
        ----------
        adj_list : list
            The list of adjacent matrices.
        """
        adj_list = torch.stack(adj_list)
        adj_list = F.normalize(adj_list, dim=1, p=1)
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)


class GraphConvolution(nn.Module):
    """
    The downstream GCN layer.
    """

    def __init__(self, in_features, out_features, bias=True):

        def reset_parameters(self):
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        reset_parameters(self)

    def forward(self, inputs, adj):
        """
        Parameters
        ----------
        inputs : tensor
            The feature matrix.
        adj : tensor
            The adjacent matrix.
        """
        support = torch.mm(inputs, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """
    The downstream GCN model.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Parameters
        ----------
        x : tensor
            The feature matrix.
        adj : tensor
            The adjacent matrix.
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class HGTLayer(nn.Module):

    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        self.relation_pri = nn.Parameter(th.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(th.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)
                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]
                k = th.einsum('bij,ijk->bik', k, relation_att)
                v = th.einsum('bij,ijk->bik', v, relation_msg)
                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v
                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
                sub_graph.edata['t'] = attn_score.unsqueeze(-1)
            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) for etype, e_id in edge_dict.items()}, cross_reducer='mean')
            new_h = {}
            for ntype in G.ntypes:
                """
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = th.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HPNLayer(nn.Module):

    def __init__(self, meta_paths_dict, in_size, dropout, k_layer, alpha, edge_drop):
        super(HPNLayer, self).__init__()
        self.hidden = nn.Sequential(nn.Linear(in_features=in_size, out_features=in_size, bias=True), nn.ReLU())
        self.meta_paths_dict = meta_paths_dict
        semantic_attention = SemanticAttention(in_size=in_size)
        mods = nn.ModuleDict({mp: APPNPConv(k_layer, alpha, edge_drop) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h_dict):
        """
        Parameters
        -----------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        h_dict = {ntype: self.hidden(h_dict[ntype]) for ntype in h_dict}
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, mp_value)
        h_dict = self.model(self._cached_coalesced_graph, h_dict)
        return h_dict


def init_drop(dropout):
    if dropout > 0:
        return nn.Dropout(dropout)
    else:
        return lambda x: x


class SelfAttention(nn.Module):

    def __init__(self, hidden_dim, attn_drop, txt):
        """
        This part is used to calculate type-level attention and semantic-level attention, and utilize them to generate :math:`z^{sc}` and :math:`z^{mp}`.

        .. math::
           w_{n}&=\\frac{1}{|V|}\\sum\\limits_{i\\in V} \\textbf{a}^\\top \\cdot \\tanh\\left(\\textbf{W}h_i^{n}+\\textbf{b}\\right) \\\\
           \\beta_{n}&=\\frac{\\exp\\left(w_{n}\\right)}{\\sum_{i=1}^M\\exp\\left(w_{i}\\right)} \\\\
           z &= \\sum_{n=1}^M \\beta_{n}\\cdot h^{n}

        Parameters
        ----------
        txt : str
            A str to identify view, MP or SC

        Returns
        -------
        z : matrix
            The fused embedding matrix

        """
        super(SelfAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax(dim=0)
        self.attn_drop = init_drop(attn_drop)
        self.txt = txt

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        None
        z = 0
        for i in range(len(embeds)):
            z += embeds[i] * beta[i]
        return z


class GraphConv(nn.Module):

    def __init__(self, in_feats, out_feats, dropout, activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight1 = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, hg, feat, edge_weight=None):
        with hg.local_scope():
            outputs = {}
            norm = {}
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                hg.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            for e in hg.canonical_etypes:
                if e[0] == e[1]:
                    hg = dgl.remove_self_loop(hg, etype=e)
            feat_src, feat_dst = expand_as_pair(feat, hg)
            hg.srcdata['h'] = feat_src
            for e in hg.canonical_etypes:
                stype, etype, dtype = e
                sub_graph = hg[stype, etype, dtype]
                sub_graph.update_all(aggregate_fn, fn.sum(msg='m', out='out'))
                temp = hg.ndata['out'].pop(dtype)
                degs = sub_graph.in_degrees().float().clamp(min=1)
                if isinstance(temp, dict):
                    temp = temp[dtype]
                if outputs.get(dtype) is None:
                    outputs[dtype] = temp
                    norm[dtype] = degs
                else:
                    outputs[dtype].add_(temp)
                    norm[dtype].add_(degs)

            def _apply(ntype, h, norm):
                h = th.matmul(h + feat[ntype], self.weight1)
                if self.activation:
                    h = self.activation(h)
                return self.dropout(h)
            return {ntype: _apply(ntype, h, norm) for ntype, h in outputs.items()}


class Mp_encoder(nn.Module):

    def __init__(self, meta_paths_dict, hidden_size, attn_drop):
        """
        This part is to encode meta-path view.

        Returns
        -------
        z_mp : matrix
            The embedding matrix under meta-path view.

        """
        super(Mp_encoder, self).__init__()
        self.act = nn.PReLU()
        self.gcn_layers = nn.ModuleDict()
        for mp in meta_paths_dict:
            one_layer = GraphConv(hidden_size, hidden_size, activation=self.act, allow_zero_in_degree=True)
            one_layer.reset_parameters()
            self.gcn_layers[mp] = one_layer
        self.meta_paths_dict = meta_paths_dict
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.semantic_attention = SelfAttention(hidden_size, attn_drop, 'mp')

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, meta_path in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, meta_path)
        for mp, meta_path in self.meta_paths_dict.items():
            new_g = self._cached_coalesced_graph[mp]
            one = self.gcn_layers[mp](new_g, h)
            semantic_embeddings.append(one)
        z_mp = self.semantic_attention(semantic_embeddings)
        return z_mp


class Sc_encoder(nn.Module):

    def __init__(self, network_schema, hidden_size, attn_drop, sample_rate, category):
        """
        This part is to encode network schema view.

        Returns
        -------
        z_mp : matrix
            The embedding matrix under network schema view.

        Note
        -----------
        There is a different sampling strategy between original code and this code. In original code, the authors implement sampling without replacement if the number of neighbors exceeds a threshold,
        and with replacement if not. In this version, we simply use the API dgl.sampling.sample_neighbors to implement this operation, and set replacement as True.

        """
        super(Sc_encoder, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(network_schema)):
            one_layer = GATConv((hidden_size, hidden_size), hidden_size, num_heads=1, attn_drop=attn_drop, allow_zero_in_degree=True)
            one_layer.reset_parameters()
            self.gat_layers.append(one_layer)
        self.network_schema = list(tuple(ns) for ns in network_schema)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.inter = SelfAttention(hidden_size, attn_drop, 'sc')
        self.sample_rate = sample_rate
        self.category = category

    def forward(self, g, h):
        intra_embeddings = []
        for i, network_schema in enumerate(self.network_schema):
            src_type = network_schema[0]
            one_graph = g[network_schema]
            cate_num = torch.arange(0, g.num_nodes(self.category))
            sub_graph = sample_neighbors(one_graph, {self.category: cate_num}, {network_schema[1]: self.sample_rate[src_type]}, replace=True)
            one = self.gat_layers[i](sub_graph, (h[src_type], h[self.category]))
            one = one.squeeze(1)
            intra_embeddings.append(one)
        z_sc = self.inter(intra_embeddings)
        return z_sc


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lam):
        """
        This part is used to calculate the contrastive loss.

        Returns
        -------
        contra_loss : float
            The calculated loss

        """
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """
        This part is used to calculate the cosine similarity of each pair of nodes from different views.

        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        """
        This is the forward part of contrast part.

        We firstly project the embeddings under two views into the space where contrastive loss is calculated. Then, we calculate the contrastive loss with projected embeddings in a cross-view way.

        .. math::
           \\mathcal{L}_i^{sc}=-\\log\\frac{\\sum_{j\\in\\mathbb{P}_i}exp\\left(sim\\left(z_i^{sc}\\_proj,z_j^{mp}\\_proj\\right)/\\tau\\right)}{\\sum_{k\\in\\{\\mathbb{P}_i\\bigcup\\mathbb{N}_i\\}}exp\\left(sim\\left(z_i^{sc}\\_proj,z_k^{mp}\\_proj\\right)/\\tau\\right)}

        where we show the contrastive loss :math:`\\mathcal{L}_i^{sc}` under network schema view, and :math:`\\mathbb{P}_i` and :math:`\\mathbb{N}_i` are positives and negatives for node :math:`i`.

        In a similar way, we can get the contrastive loss :math:`\\mathcal{L}_i^{mp}` under meta-path view. Finally, we utilize combination parameter :math:`\\lambda` to add this two losses.

        Note
        -----------
        In implementation, each row of 'matrix_mp2sc' means the similarity with exponential between one node in meta-path view and all nodes in network schema view. Then, we conduct normalization for this row,
        and pick the results where the pair of nodes are positives. Finally, we sum these results for each row, and give a log to get the final loss.

        """
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-08)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()
        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-08)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        contra_loss = self.lam * lori_mp + (1 - self.lam) * lori_sc
        return contra_loss


class Generator(nn.Module):
    """
     A Discriminator :math:`D` eveluates the connectivity between the pair of nodes :math:`u` and :math:`v` w.r.t. a relation :math:`r`. It is formulated as follow:

    .. math::
        D(\\mathbf{e}_v|\\mathbf{u},\\mathbf{r};\\mathbf{\\theta}^D) = \\frac{1}{1+\\exp(-\\mathbf{e}_u^{D^T}) \\mathbf{M}_r^D \\mathbf{e}_v}

    where :math:`e_v \\in \\mathbb{R}^{d\\times 1}` is the input embeddings of the sample :math:`v`,
    :math:`e_u^D \\in \\mathbb{R}^{d \\times 1}` is the learnable embedding of node :math:`u`,
    :math:`M_r^D \\in \\mathbb{R}^{d \\times d}` is a learnable relation matrix for relation :math:`r`.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\\mathbf{u}, \\mathbf{r}; \\mathbf{\\theta}^G) = f(\\mathbf{W_2}f(\\mathbf{W}_1 \\mathbf{e} + \\mathbf{b}_1) + \\mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is :

    .. math::
        L_G = \\mathbb{E}_{\\langle u,v\\rangle \\sim P_G, e'_v \\sim G(u,r;\\theta^G)} = -\\log -D(e'_v|u,r)) +\\lambda^G || \\theta^G ||_2^2

    where :math:`\\theta^G` denote all the learnable parameters in Generator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """

    def __init__(self, emb_size, hg):
        super().__init__()
        self.n_relation = len(hg.etypes)
        self.node_emb_dim = emb_size
        self.nodes_embedding = nn.ParameterDict()
        for nodes_type, nodes_emb in hg.ndata['h'].items():
            self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)
        self.relation_matrix = nn.ParameterDict()
        for et in hg.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)
        self.fc = nn.Sequential(OrderedDict([('w_1', nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim, bias=True)), ('a_1', nn.LeakyReLU()), ('w_2', nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim)), ('a_2', nn.LeakyReLU())]))

    def forward(self, gen_hg, dis_node_emb, dis_relation_matrix, noise_emb):
        """
        Parameters
        -----------
        gen_hg: dgl.heterograph
            sampled graph for generator.
        dis_node_emb: dict[str: Tensor]
            discriminator node embedding.
        dis_relation_matrix: dict[str: Tensor]
            discriminator relation embedding.
        noise_emb: dict[str: Tensor]
            noise embedding.
        """
        score_list = []
        with gen_hg.local_scope():
            self.assign_node_data(gen_hg, dis_node_emb)
            self.assign_edge_data(gen_hg, dis_relation_matrix)
            self.generate_neighbor_emb(gen_hg, noise_emb)
            for et in gen_hg.canonical_etypes:
                gen_hg.apply_edges(lambda edges: {'s': edges.src['dh'].unsqueeze(1).matmul(edges.data['de']).squeeze()}, etype=et)
                gen_hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['g'])}, etype=et)
                score = torch.sum(gen_hg.edata['score'].pop(et), dim=1)
                score_list.append(score)
        return torch.cat(score_list)

    def get_parameters(self):
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}

    def generate_neighbor_emb(self, hg, noise_emb):
        for et in hg.canonical_etypes:
            hg.apply_edges(lambda edges: {'g': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).squeeze()}, etype=et)
            hg.apply_edges(lambda edges: {'g': edges.data['g'] + noise_emb[et]}, etype=et)
            hg.apply_edges(lambda edges: {'g': self.fc(edges.data['g'])}, etype=et)
        return {et: hg.edata['g'][et] for et in hg.canonical_etypes}

    def assign_edge_data(self, hg, dis_relation_matrix=None):
        for et in hg.canonical_etypes:
            n = hg.num_edges(et)
            e = self.relation_matrix[et[1]]
            hg.edata['e'] = {et: e.expand(n, -1, -1)}
            if dis_relation_matrix:
                de = dis_relation_matrix[et[1]]
                hg.edata['de'] = {et: de.expand(n, -1, -1)}

    def assign_node_data(self, hg, dis_node_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if dis_node_emb:
            hg.ndata['dh'] = dis_node_emb


class ScorePredictor(nn.Module):

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class lstm_aggr(nn.Module):
    """
    Aggregate the same neighbors with LSTM
    """

    def __init__(self, dim):
        super(lstm_aggr, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh


class aggregate_het_neigh(nn.Module):
    """
    It is a Aggregating Heterogeneous Neighbors(C3)
    Same Type Neighbors Aggregation

    """

    def __init__(self, ntypes, dim):
        super(aggregate_het_neigh, self).__init__()
        self.neigh_rnn = nn.ModuleDict({})
        self.ntypes = ntypes
        for n in ntypes:
            self.neigh_rnn[n] = lstm_aggr(dim)

    def forward(self, hg, inputs):
        with hg.local_scope():
            outputs = {}
            for i in self.ntypes:
                outputs[i] = []
            if isinstance(inputs, tuple) or hg.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in inputs.items()}
                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in src_inputs or dtype not in dst_inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](rel_graph, (src_inputs[stype], dst_inputs[dtype]))
                    outputs[dtype].append(dstdata)
            else:
                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](rel_graph, inputs[stype])
                    outputs[dtype].append(dstdata)
            return outputs


class het_content_encoder(nn.Module):
    """
    The Encoding Heterogeneous Contents(C2) in the paper
    For a specific node type, encoder different content features with a LSTM.

    In paper, it is (b) NN-1: node heterogeneous contents encoder in figure 2.

    Parameters
    ------------
    dim : int
        input dimension

    Attributes
    ------------
    content_rnn : nn.Module
        nn.LSTM encode different content feature
    """

    def __init__(self, dim):
        super(het_content_encoder, self).__init__()
        self.content_rnn = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
        self.content_rnn.flatten_parameters()
        self.dim = dim

    def forward(self, h_dict):
        """

        Parameters
        ----------
        h_dict: dict[str, th.Tensor]
            key means different content feature

        Returns
        -------
        content_h : th.tensor
        """
        concate_embed = []
        for _, h in h_dict.items():
            concate_embed.append(h)
        concate_embed = th.cat(concate_embed, 1)
        concate_embed = concate_embed.view(concate_embed.shape[0], -1, self.dim)
        all_state, last_state = self.content_rnn(concate_embed)
        out_h = th.mean(all_state, 1).squeeze()
        return out_h


class Het_Aggregate(nn.Module):
    """
    The whole model of HetGNN

    Attributes
    -----------
    content_rnn : nn.Module
        het_content_encoder
    neigh_rnn : nn.Module
        aggregate_het_neigh
    atten_w : nn.ModuleDict[str, nn.Module]


    """

    def __init__(self, ntypes, dim):
        super(Het_Aggregate, self).__init__()
        self.ntypes = ntypes
        self.dim = dim
        self.content_rnn = het_content_encoder(dim)
        self.neigh_rnn = aggregate_het_neigh(ntypes, dim)
        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(dim)
        self.embed_d = dim

    def forward(self, hg, h_dict):
        with hg.local_scope():
            content_h = {}
            for ntype, h in h_dict.items():
                content_h[ntype] = self.content_rnn(h)
            neigh_h = self.neigh_rnn(hg, content_h)
            dst_h = {k: v[:hg.number_of_dst_nodes(k)] for k, v in content_h.items()}
            out_h = {}
            for n in self.ntypes:
                d_h = dst_h[n]
                batch_size = d_h.shape[0]
                concat_h = []
                concat_emd = []
                for i in range(len(neigh_h[n])):
                    concat_h.append(th.cat((d_h, neigh_h[n][i]), 1))
                    concat_emd.append(neigh_h[n][i])
                concat_h.append(th.cat((d_h, d_h), 1))
                concat_emd.append(d_h)
                concat_h = th.hstack(concat_h).view(batch_size * (len(self.ntypes) + 1), self.dim * 2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, len(self.ntypes) + 1)
                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)
                concat_emd = th.hstack(concat_emd).view(batch_size, len(self.ntypes) + 1, self.dim)
                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                out_h[n] = weight_agg_batch
            return out_h


class HetSANNConv(nn.Module):
    """
    The HetSANN convolution layer.

    Parameters
    ----------
    num_heads: int
        the number of heads in the attention computing
    in_dim: int
        the input dimension of the features
    hidden_dim: int
        the hidden dimension of the features
    num_etypes: int
        the number of the edge types
    dropout: float
        the dropout rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    """

    def __init__(self, num_heads, in_dim, hidden_dim, num_etypes, dropout, negative_slope, residual, activation):
        super(HetSANNConv, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.W = TypedLinear(self.in_dim, self.hidden_dim * self.num_heads, num_etypes)
        self.a_l = TypedLinear(self.hidden_dim * self.num_heads, self.hidden_dim * self.num_heads, num_etypes)
        self.a_r = TypedLinear(self.hidden_dim * self.num_heads, self.hidden_dim * self.num_heads, num_etypes)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        if residual:
            self.residual = nn.Linear(in_dim, self.hidden_dim * num_heads)
        else:
            self.register_buffer('residual', None)
        self.activation = activation

    def forward(self, g, x, ntype, etype, presorted=False):
        """
        The forward part of the HetSANNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        feat = self.W(x, etype, presorted)
        h = self.dropout(feat)
        g.ndata['h'] = h
        g.apply_edges(Fn.copy_u('h', 'm'))
        h = g.edata['m']
        h = h.view(-1, self.num_heads, self.hidden_dim)
        h_l = self.a_l(h.view(-1, self.num_heads * self.hidden_dim), etype, presorted).view(-1, self.num_heads, self.hidden_dim).sum(dim=-1)
        h_r = self.a_r(h.view(-1, self.num_heads * self.hidden_dim), etype, presorted).view(-1, self.num_heads, self.hidden_dim).sum(dim=-1)
        attention = self.leakyrelu(h_l + h_r)
        attention = edge_softmax(g, attention)
        with g.local_scope():
            h = h.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = h @ attention.reshape(-1, self.num_heads, 1)
            g.update_all(Fn.copy_e('m', 'w'), Fn.sum('w', 'emb'))
            h_output = g.ndata['emb']
        if self.residual:
            res = self.residual(x)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)
        return h_output


class KGCN_Aggregate(nn.Module):

    def __init__(self, args):
        super(KGCN_Aggregate, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        if self.args.aggregate == 'CONCAT':
            self.agg = nn.Linear(self.in_dim * 2, self.out_dim)
        else:
            self.agg = nn.Linear(self.in_dim, self.out_dim)

    def aggregate(self):
        self.sub_g.update_all(fn.u_mul_e('embedding', 'weight', 'm'), fn.sum('m', 'ft'))
        self.userList = []
        self.labelList = []
        embeddingList = []
        for i in range(len(self.data)):
            weightIndex = np.where(self.itemlist == int(self.sub_g.dstdata['_ID'][i]))
            if self.args.aggregate == 'SUM':
                embeddingList.append(self.sub_g.dstdata['embedding'][i] + self.sub_g.dstdata['ft'][i][weightIndex])
            elif self.args.aggregate == 'CONCAT':
                embeddingList.append(th.cat([self.sub_g.dstdata['embedding'][i], self.sub_g.dstdata['ft'][i][weightIndex].squeeze(0)], dim=-1))
            elif self.args.aggregate == 'NEIGHBOR':
                embeddingList.append(self.sub_g.dstdata['embedding'][i])
            self.userList.append(int(self.user_indices[weightIndex]))
            self.labelList.append(int(self.labels[weightIndex]))
        self.sub_g.dstdata['embedding'] = th.stack(embeddingList).squeeze(1)
        output = F.dropout(self.sub_g.dstdata['embedding'], p=0)
        if self.layer + 1 == len(self.blocks):
            self.item_embeddings = th.tanh(self.agg(output))
        else:
            self.item_embeddings = th.relu(self.agg(output))

    def forward(self, blocks, inputdata):
        """
        Aggregate the entity representation and its neighborhood representation

        Parameters
        ----------
        blocks : list
            Blocks saves the information of neighbor nodes in each layer
        inputdata : numpy.ndarray
            Inputdata contains the relationship between the user and the entity

        Returns
        -------
        item_embeddings : torch.Tensor
            items' embeddings after aggregated
        userList : list
            Users corresponding to items
        labelList : list
            Labels corresponding to items
        """
        self.data = inputdata
        self.blocks = blocks
        self.user_indices = self.data[:, 0]
        self.itemlist = self.data[:, 1]
        self.labels = self.data[:, 2]
        for self.layer in range(len(blocks)):
            self.sub_g = blocks[self.layer]
            self.aggregate()
        return self.item_embeddings, self.userList, self.labelList


class MAGNN_attn_intra(nn.Module):

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.5, attn_drop=0.5, negative_slope=0.01, activation=F.elu):
        super(MAGNN_attn_intra, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, feat, metapath, metapath_idx):
        _metapath = metapath.split('-')
        device = feat[0].device
        h_meta = self.feat_drop(feat[0]).view(-1, self._num_heads, self._out_feats)
        er = (h_meta * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph_data = {('meta_inst', 'meta2{}'.format(_metapath[0]), _metapath[0]): (th.arange(0, metapath_idx.shape[0]), th.tensor(metapath_idx[:, 0]))}
        num_nodes_dict = {'meta_inst': metapath_idx.shape[0], _metapath[0]: feat[1].shape[0]}
        g_meta = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        g_meta.nodes['meta_inst'].data.update({'feat_src': h_meta, 'er': er})
        g_meta.apply_edges(func=fn.copy_u('er', 'e'), etype='meta2{}'.format(_metapath[0]))
        e = self.leaky_relu(g_meta.edata.pop('e'))
        g_meta.edata['a'] = self.attn_drop(edge_softmax(g_meta, e))
        g_meta.update_all(message_func=fn.u_mul_e('feat_src', 'a', 'm'), reduce_func=fn.sum('m', 'feat'))
        feat = self.activation(g_meta.dstdata['feat'])
        return feat.flatten(1)


class MAGNN_layer(nn.Module):

    def __init__(self, in_feats, inter_attn_feats, out_feats, num_heads, metapath_list, ntypes, edge_type_list, dst_ntypes, encoder_type='RotateE', last_layer=False):
        super(MAGNN_layer, self).__init__()
        self.in_feats = in_feats
        self.inter_attn_feats = inter_attn_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.metapath_list = metapath_list
        self.ntypes = ntypes
        self.edge_type_list = edge_type_list
        self.dst_ntypes = dst_ntypes
        self.encoder_type = encoder_type
        self.last_layer = last_layer
        in_feats_dst_meta = tuple((in_feats, in_feats))
        self.intra_attn_layers = nn.ModuleDict()
        for metapath in self.metapath_list:
            self.intra_attn_layers[metapath] = MAGNN_attn_intra(in_feats=in_feats_dst_meta, out_feats=in_feats, num_heads=num_heads)
        self.inter_linear = nn.ModuleDict()
        self.inter_attn_vec = nn.ModuleDict()
        for ntype in dst_ntypes:
            self.inter_linear[ntype] = nn.Linear(in_features=in_feats * num_heads, out_features=inter_attn_feats, bias=True)
            self.inter_attn_vec[ntype] = nn.Linear(in_features=inter_attn_feats, out_features=1, bias=False)
            nn.init.xavier_normal_(self.inter_linear[ntype].weight, gain=1.414)
            nn.init.xavier_normal_(self.inter_attn_vec[ntype].weight, gain=1.414)
        if encoder_type == 'RotateE':
            r_vec_ = nn.Parameter(th.empty(size=(len(edge_type_list) // 2, in_feats * num_heads // 2, 2)))
            nn.init.xavier_normal_(r_vec_.data, gain=1.414)
            self.r_vec = F.normalize(r_vec_, p=2, dim=2)
            self.r_vec = th.stack([self.r_vec, self.r_vec], dim=1)
            self.r_vec[:, 1, :, 1] = -self.r_vec[:, 1, :, 1]
            self.r_vec = self.r_vec.reshape(r_vec_.shape[0] * 2, r_vec_.shape[1], 2)
            self.r_vec_dict = nn.ParameterDict()
            for i, edge_type in zip(range(len(edge_type_list)), edge_type_list):
                self.r_vec_dict[edge_type] = nn.Parameter(self.r_vec[i])
        elif encoder_type == 'Linear':
            self.encoder_linear = nn.Linear(in_features=in_feats * num_heads, out_features=in_feats * num_heads)
        if last_layer:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=out_feats)
        else:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=num_heads * out_feats)
        nn.init.xavier_normal_(self._output_projection.weight, gain=1.414)

    def forward(self, feat_dict, metapath_idx_dict):
        feat_intra = {}
        for _metapath in self.metapath_list:
            feat_intra[_metapath] = self.intra_metapath_trans(feat_dict, metapath=_metapath, metapath_idx_dict=metapath_idx_dict)
        feat_inter = self.inter_metapath_trans(feat_dict=feat_dict, feat_intra=feat_intra, metapath_list=self.metapath_list)
        feat_final = self.output_projection(feat_inter=feat_inter)
        return feat_final, feat_inter

    def intra_metapath_trans(self, feat_dict, metapath, metapath_idx_dict):
        metapath_idx = metapath_idx_dict[metapath]
        intra_metapath_feat = self.encoder(feat_dict, metapath, metapath_idx)
        feat_intra = self.intra_attn_layers[metapath]([intra_metapath_feat, feat_dict[metapath.split('-')[0]]], metapath, metapath_idx)
        return feat_intra

    def inter_metapath_trans(self, feat_dict, feat_intra, metapath_list):
        meta_s = {}
        feat_inter = {}
        for metapath in metapath_list:
            _metapath = metapath.split('-')
            meta_feat = feat_intra[metapath]
            meta_feat = th.tanh(self.inter_linear[_metapath[0]](meta_feat)).mean(dim=0)
            meta_s[metapath] = self.inter_attn_vec[_metapath[0]](meta_feat)
        for ntype in self.ntypes:
            if ntype in self.dst_ntypes:
                metapaths = np.array(metapath_list)[[(meta.split('-')[0] == ntype) for meta in metapath_list]]
                meta_b = th.tensor(itemgetter(*metapaths)(meta_s))
                meta_b = F.softmax(meta_b, dim=0)
                meta_feat = itemgetter(*metapaths)(feat_intra)
                feat_inter[ntype] = th.stack([(meta_b[i] * meta_feat[i]) for i in range(len(meta_b))], dim=0).sum(dim=0)
            else:
                feat_inter[ntype] = feat_dict[ntype]
        return feat_inter

    def encoder(self, feat_dict, metapath, metapath_idx):
        _metapath = metapath.split('-')
        device = feat_dict[_metapath[0]].device
        feat = th.zeros((len(_metapath), metapath_idx.shape[0], feat_dict[_metapath[0]].shape[1]), device=device)
        for i, ntype in zip(range(len(_metapath)), _metapath):
            feat[i] = feat_dict[ntype][metapath_idx[:, i]]
        feat = feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2] // 2, 2)
        if self.encoder_type == 'RotateE':
            temp_r_vec = th.zeros((len(_metapath), feat.shape[-2], 2), device=device)
            temp_r_vec[0, :, 0] = 1
            for i in range(1, len(_metapath), 1):
                edge_type = '{}-{}'.format(_metapath[i - 1], _metapath[i])
                temp_r_vec[i] = self.complex_hada(temp_r_vec[i - 1], self.r_vec_dict[edge_type])
                feat[i] = self.complex_hada(feat[i], temp_r_vec[i], opt='feat')
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            return th.mean(feat, dim=0)
        elif self.encoder_type == 'Linear':
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            feat = self.encoder_linear(th.mean(feat, dim=0))
            return feat
        elif self.encoder_type == 'Average':
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            return th.mean(feat, dim=0)
        else:
            raise ValueError('The encoder type {} has not been implemented yet.'.format(self.encoder_type))

    @staticmethod
    def complex_hada(h, v, opt='r_vec'):
        if opt == 'r_vec':
            h_h, l_h = h[:, 0].clone(), h[:, 1].clone()
        else:
            h_h, l_h = h[:, :, 0].clone(), h[:, :, 1].clone()
        h_v, l_v = v[:, 0].clone(), v[:, 1].clone()
        res = th.zeros_like(h)
        if opt == 'r_vec':
            res[:, 0] = h_h * h_v - l_h * l_v
            res[:, 1] = h_h * l_v + l_h * h_v
        else:
            res[:, :, 0] = h_h * h_v - l_h * l_v
            res[:, :, 1] = h_h * l_v + l_h * h_v
        return res

    def output_projection(self, feat_inter):
        feat_final = {}
        for ntype in self.ntypes:
            feat_final[ntype] = self._output_projection(feat_inter[ntype])
        return feat_final


class HMAELayer(nn.Module):
    """
        HMAE: Hybrid Metapath Autonomous Extraction

        The method to generate l-hop hybrid adjacency matrix

        .. math::
            A_{(l)}=\\Pi_{i=1}^{l} A_{i}
    """

    def __init__(self, in_channels, out_channels, first=True):
        super(HMAELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.norm = EdgeWeightNorm(norm='right')
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, softmax_flag=False)
            self.conv2 = GTConv(in_channels, out_channels, softmax_flag=False)
        else:
            self.conv1 = GTConv(in_channels, out_channels, softmax_flag=False)

    def softmax_norm(self, H):
        norm_H = []
        for i in range(len(H)):
            g = H[i]
            g.edata['w_sum'] = self.norm(g, th.exp(g.edata['w_sum']))
            norm_H.append(g)
        return norm_H

    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.softmax_norm(self.conv1(A))
            result_B = self.softmax_norm(self.conv2(A))
            W = [self.conv1.weight.detach(), self.conv2.weight.detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [self.conv1.weight.detach().detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W, result_A


class HLHIA(nn.Module):
    """
        HLHIA: The Hop-Level Heterogeneous Information Aggregation

        The l-hop representation :math:`Z_{l}` is generated by the original node feature through a graph conv

        .. math::
           Z_{l}^{\\Phi_{p}} = \\sigma\\left[\\left(D_{(l)}^{\\Phi_{p}}\\right)^{-1} A_{(l)}^{\\Phi_{p}} h W^{\\Phi_{p}}\\right]

        where :math:`\\Phi_{p}` is the hybrid l-hop metapath and `\\mathcal{h}` is the original node feature.

    """

    def __init__(self, num_edge_type, num_channels, num_layers, in_dim, hidden_dim):
        super(HLHIA, self).__init__()
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HMAELayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(HMAELayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn_list = nn.ModuleList()
        for i in range(num_channels):
            self.gcn_list.append(GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu))
        self.norm = EdgeWeightNorm(norm='right')

    def forward(self, A, h):
        layer_list = []
        for i in range(len(self.layers)):
            if i == 0:
                H, W, first_adj = self.layers[i](A)
                layer_list.append(first_adj)
                layer_list.append(H)
            else:
                H, W, first_adj = self.layers[i](A, H)
                layer_list.append(H)
        channel_attention_list = []
        for i in range(self.num_channels):
            gcn = self.gcn_list[i]
            layer_attention_list = []
            for j in range(len(layer_list)):
                layer = layer_list[j][i]
                layer = dgl.remove_self_loop(layer)
                edge_weight = layer.edata['w_sum']
                layer = dgl.add_self_loop(layer)
                edge_weight = th.cat((edge_weight, th.full((layer.number_of_nodes(),), 1, device=layer.device)))
                edge_weight = self.norm(layer, edge_weight)
                layer_attention_list.append(gcn(layer, h, edge_weight=edge_weight))
            channel_attention_list.append(layer_attention_list)
        return channel_attention_list


class HSAF(nn.Module):
    """
        HSAF: Hierarchical Semantic Attention Fusion

        The HSAF model use two level attention mechanism to generate final representation

        * Hop-level attention

          .. math::
              \\alpha_{i, l}^{\\Phi_{p}}=\\sigma\\left[\\delta^{\\Phi_{p}} \\tanh \\left(W^{\\Phi_{p}} Z_{i, l}^{\\Phi_{p}}\\right)\\right]

          In which, :math:`\\alpha_{i, l}^{\\Phi_{p}}` is the importance of the information :math:`\\left(Z_{i, l}^{\\Phi_{p}}\\right)`
          of the l-th-hop neighbors of node i under the path :math:`\\Phi_{p}`, and :math:`\\delta^{\\Phi_{p}}` represents the learnable matrix.

          Then normalize :math:`\\alpha_{i, l}^{\\Phi_{p}}`

          .. math::
              \\beta_{i, l}^{\\Phi_{p}}=\\frac{\\exp \\left(\\alpha_{i, l}^{\\Phi_{p}}\\right)}{\\sum_{j=1}^{L} \\exp \\left(\\alpha_{i, j}^{\\Phi_{p}}\\right)}

          Finally, we get hop-level attention representation in one hybrid metapath.

          .. math::
              Z_{i}^{\\Phi_{p}}=\\sum_{l=1}^{L} \\beta_{l}^{\\Phi_{p}} Z_{l}^{\\Phi_{p}}

        * Channel-level attention

          It also can be seen as multi-head attention mechanism.

          .. math::
              \\alpha_{i, \\Phi_{p}}=\\sigma\\left[\\delta \\tanh \\left(W Z_{i}^{\\Phi_{p}}\\right)\\right.

          Then normalize :math:`\\alpha_{i, \\Phi_{p}}`

          .. math::
              \\beta_{i, \\Phi_{p}}=\\frac{\\exp \\left(\\alpha_{i, \\Phi_{p}}\\right)}{\\sum_{p^{\\prime} \\in P} \\exp \\left(\\alpha_{\\Phi_{p^{\\prime}}}\\right)}

          Finally, we get final representation of every nodes.

          .. math::
              Z_{i}=\\sum_{p \\in P} \\beta_{i, \\Phi_{p}} Z_{i, \\Phi_{p}}

    """

    def __init__(self, num_edge_type, num_channels, num_layers, in_dim, hidden_dim):
        super(HSAF, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.HLHIA_layer = HLHIA(num_edge_type, self.num_channels, self.num_layers, self.in_dim, self.hidden_dim)
        self.channel_attention = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Tanh(), nn.Linear(1, 1, bias=False), nn.ReLU())
        self.layers_attention = nn.ModuleList()
        for i in range(num_channels):
            self.layers_attention.append(nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Tanh(), nn.Linear(1, 1, bias=False), nn.ReLU()))

    def forward(self, A, h):
        attention_list = self.HLHIA_layer(A, h)
        channel_attention_list = []
        for i in range(self.num_channels):
            layer_level_feature_list = attention_list[i]
            layer_attention = self.layers_attention[i]
            for j in range(self.num_layers + 1):
                layer_level_feature = layer_level_feature_list[j]
                if j == 0:
                    layer_level_alpha = layer_attention(layer_level_feature)
                else:
                    layer_level_alpha = th.cat((layer_level_alpha, layer_attention(layer_level_feature)), dim=-1)
            layer_level_beta = th.softmax(layer_level_alpha, dim=-1)
            channel_attention_list.append(th.bmm(th.stack(layer_level_feature_list, dim=-1), layer_level_beta.unsqueeze(-1)).squeeze(-1))
        for i in range(self.num_channels):
            channel_level_feature = channel_attention_list[i]
            if i == 0:
                channel_level_alpha = self.channel_attention(channel_level_feature)
            else:
                channel_level_alpha = th.cat((channel_level_alpha, self.channel_attention(channel_level_feature)), dim=-1)
        channel_level_beta = th.softmax(channel_level_alpha, dim=-1)
        channel_attention = th.bmm(th.stack(channel_attention_list, dim=-1), channel_level_beta.unsqueeze(-1)).squeeze(-1)
        return channel_attention


class Base_model(nn.Module):

    def __init__(self):
        super(Base_model, self).__init__()

    def Micro_layer(self, h_dict):
        return h_dict

    def Macro_layer(self, h_dict):
        return h_dict

    def forward(self, h_dict):
        h_dict = self.Micro_layer(h_dict)
        h_dict = self.Macro_layer(h_dict)
        return h_dict


class Multi_level(nn.Module):

    def __init__(self):
        super(Multi_level, self).__init__()
        self.micro_layer = None
        self.macro_layer = None

    def forward(self):
        return


class FeedForwardNet(nn.Module):
    """
        A feedforward net.

        Input
        ------
        in_feats :
            input feature dimention
        hidden :
            hidden layer dimention
        out_feats :
            output feature dimention
        num_layers :
            number of layers
        dropout :
            dropout rate
    """

    def __init__(self, in_feats, hidden, out_feats, num_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.num_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.num_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    """
        The SIGN model.

        Parameters
        ------------
        in_feats :
            input feature dimention
        hidden :
            hidden layer dimention
        out_feats :
            output feature dimention
        num_hops :
            number of hops
        num_layers :
            number of layers
        dropout :
            dropout rate
        input_drop :
            whether or not to dropout when inputting features

    """

    def __init__(self, in_feats, hidden, out_feats, num_hops, num_layers, dropout, input_drop):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for i in range(num_hops):
            self.inception_ffs.append(FeedForwardNet(in_feats, hidden, hidden, num_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats, num_layers, dropout)

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(th.cat(hidden, dim=-1))))
        return out


class WeightedAggregator(nn.Module):
    """
        Get new features by multiplying the old features by the weight matrix.

        Parameters
        -------------
        num_feats :
            number of subsets
        in_feats :
            input feature dimention
        num_hops :
            number of hops


    """

    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(th.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats):
        new_feats = []
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats


class MLP_follow_model(nn.Module):

    def __init__(self, model, h_dim, out_dim):
        super(MLP_follow_model, self).__init__()
        self.gnn_model = model
        self.project = nn.Sequential(nn.Linear(h_dim, out_dim, bias=False))

    def forward(self, hg, h=None, category=None):
        if h is None:
            h = self.gnn_model(hg)
        else:
            h = self.gnn_model(hg, h)
        for i in h:
            h[i] = self.project(h[i])
        return h


class RGATLayer(nn.Module):

    def __init__(self, in_feat, out_feat, num_heads, rel_names, activation=None, dropout=0.0, bias=True):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.conv = dglnn.HeteroGraphConv({rel: dgl.nn.pytorch.GATConv(in_feat, out_feat, num_heads=num_heads, bias=bias, allow_zero_in_degree=True) for rel in rel_names})

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put


class RelGraphConvLayer(nn.Module):
    """Relational graph convolution layer.

    We use `HeteroGraphConv <https://docs.dgl.ai/api/python/nn.pytorch.html#heterographconv>`_ to implement the model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self, in_feat, out_feat, rel_names, num_bases, *, weight=True, bias=True, activation=None, self_loop=False, dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.batchnorm = False
        self.conv = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False) for rel in rel_names})
        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)} for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}
        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs
        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelationCrossing(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float=0.0, negative_slope: float=0.2):
        """
        Relation crossing layer

        Parameters
        ----------
        in_feats : pair of ints
            input feature size
        out_feats : int
            output feature size
        num_heads : int
            number of heads in Multi-Head Attention
        dropout : float
            optional, dropout rate, defaults: 0.0
        negative_slope : float
            optional, negative slope rate, defaults: 0.2
        """
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dsttype_node_features: torch.Tensor, relations_crossing_attention_weight: nn.Parameter):
        """
        Parameters
        ----------
        dsttype_node_features:
            a tensor of (dsttype_node_relations_num, num_dst_nodes, n_heads * hidden_dim)
        relations_crossing_attention_weight:
            Parameter the shape is (n_heads, hidden_dim)
        Returns:
        ----------
        output_features: Tensor

        """
        if len(dsttype_node_features) == 1:
            dsttype_node_features = dsttype_node_features.squeeze(dim=0)
        else:
            dsttype_node_features = dsttype_node_features.reshape(dsttype_node_features.shape[0], -1, self._num_heads, self._out_feats)
            dsttype_node_relation_attention = (dsttype_node_features * relations_crossing_attention_weight).sum(dim=-1, keepdim=True)
            dsttype_node_relation_attention = F.softmax(self.leaky_relu(dsttype_node_relation_attention), dim=0)
            dsttype_node_features = (dsttype_node_features * dsttype_node_relation_attention).sum(dim=0)
            dsttype_node_features = self.dropout(dsttype_node_features)
            dsttype_node_features = dsttype_node_features.reshape(-1, self._num_heads * self._out_feats)
        return dsttype_node_features


class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float=0.0, negative_slope: float=0.2):
        """

        Parameters
        ----------
        node_hidden_dim: int
            node hidden feature size
        relation_hidden_dim: int
            relation hidden feature size
        num_heads: int
            number of heads in Multi-Head Attention
        dropout: float
            dropout rate, defaults: 0.0
        negative_slope: float
            negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: list, dst_relation_embeddings: list, dst_node_feature_transformation_weight: list, dst_relation_embedding_transformation_weight: list):
        """
        Parameters
        ----------
        dst_node_features: list
            e.g [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        dst_relation_embeddings: list
            e.g [each shape is (n_heads * relation_hidden_dim)]
        dst_node_feature_transformation_weight: list
            e.g [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        dst_relation_embedding_transformation_weight:  list
            e.g [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]

        Returns
        ----------
        dst_node_relation_fusion_feature: Tensor
            the target node representation after relation-aware representations fusion
        """
        if len(dst_node_features) == 1:
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            dst_node_features = torch.stack(dst_node_features, dim=0).reshape(len(dst_node_features), -1, self.num_heads, self.node_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features), self.num_heads, self.relation_hidden_dim)
            dst_node_feature_transformation_weight = torch.stack(dst_node_feature_transformation_weight, dim=0).reshape(len(dst_node_features), self.num_heads, self.node_hidden_dim, self.node_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight, dim=0).reshape(len(dst_node_features), self.num_heads, self.relation_hidden_dim, self.node_hidden_dim)
            dst_node_features = torch.einsum('abcd,acde->abce', dst_node_features, dst_node_feature_transformation_weight)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings, dst_relation_embedding_transformation_weight)
            attention_scores = (dst_node_features * dst_relation_embeddings.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
            attention_scores = F.softmax(self.leaky_relu(attention_scores), dim=0)
            dst_node_relation_fusion_feature = (dst_node_features * attention_scores).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1, self.num_heads * self.node_hidden_dim)
        return dst_node_relation_fusion_feature


class AGNNConv(nn.Module):

    def __init__(self, eps=0.0, train_eps=False, learn_beta=True):
        super(AGNNConv, self).__init__()
        self.initial_eps = eps
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor(1))
        else:
            self.register_buffer('beta', th.Tensor(1))
        self.learn_beta = learn_beta
        if train_eps:
            self.eps = th.nn.Parameter(th.ones([eps]))
        else:
            self.register_buffer('eps', th.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.learn_beta:
            self.beta.data.fill_(1)

    def forward(self, graph, feat, edge_weight):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)
            e = self.beta * edge_weight
            graph.edata['p'] = edge_softmax(graph, e, norm_by='src')
            graph.update_all(fn.u_mul_e('norm_h', 'p', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata.pop('h')
            rst = (1 + self.eps) * feat + rst
            return rst


class NodeEncoder(torch.nn.Module):

    def __init__(self, base_embedding_dim, num_nodes, pretrained_node_embedding_tensor, is_pre_trained):
        super().__init__()
        self.pretrained_node_embedding_tensor = pretrained_node_embedding_tensor
        self.base_embedding_dim = base_embedding_dim
        if not is_pre_trained:
            self.base_embedding_layer = torch.nn.Embedding(num_nodes, base_embedding_dim)
            self.base_embedding_layer.weight.data.uniform_(-1, 1)
        else:
            self.base_embedding_layer = torch.nn.Embedding.from_pretrained(pretrained_node_embedding_tensor)

    def forward(self, node_id):
        node_id = torch.LongTensor([int(node_id)])
        x_base = self.base_embedding_layer(node_id)
        return x_base


class GCNGraphEncoder(torch.nn.Module):

    def __init__(self, G, pretrained_node_embedding_tensor, is_pre_trained, base_embedding_dim, max_length):
        super().__init__()
        self.g = G
        self.base_embedding_dim = base_embedding_dim
        self.max_length = max_length
        self.no_nodes = self.g.num_nodes()
        self.no_relations = self.g.num_edges()
        self.node_embedding = NodeEncoder(base_embedding_dim, self.no_nodes, pretrained_node_embedding_tensor, is_pre_trained)
        self.special_tokens = {'[PAD]': 0, '[MASK]': 1}
        self.special_embed = torch.nn.Embedding(len(self.special_tokens), base_embedding_dim)
        self.special_embed.weight.data.uniform_(-1, 1)

    def forward(self, subgraphs_list, masked_nodes):
        num_subgraphs = len(subgraphs_list)
        node_emb = torch.zeros(num_subgraphs, self.max_length + 1, self.base_embedding_dim)
        for ii, subgraph in enumerate(subgraphs_list):
            masked_set = masked_nodes[ii]
            for node in subgraph.nodes():
                node_id = subgraph.ndata[dgl.NID][int(node)]
                if node_id not in masked_set:
                    node_emb[ii][node] = self.node_embedding(int(node_id))
        special_tokens_embed = {}
        for token in self.special_tokens:
            node_id = Variable(torch.LongTensor([self.special_tokens[token]]))
            tmp_embed = self.special_embed(node_id)
            special_tokens_embed[self.special_tokens[token] + self.no_nodes] = {'token': token, 'embed': tmp_embed}
        return node_emb


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask == True, -1000000000.0)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_V = torch.nn.Linear(d_model, d_v * n_heads)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(d_k)
        self.wrap = torch.nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layerNorm = torch.nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_prod_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.wrap(context)
        return self.layerNorm(output + residual), attn


def gelu(x):
    """"Implementation of the gelu activation function by Hugging Face."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PoswiseFeedForwardNet(torch.nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(torch.nn.Module):

    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class SLiCEFinetuneLayer(torch.nn.Module):

    @classmethod
    def build_model_from_args(cls, args):
        return cls(d_model=args.d_model, ft_d_ff=args.ft_d_ff, ft_layer=args.ft_layer, ft_drop_rate=args.ft_drop_rate, ft_input_option=args.ft_input_option, n_layers=args.num_layers)

    def __init__(self, d_model, ft_d_ff, ft_layer, ft_drop_rate, ft_input_option, num_layers):
        super().__init__()
        self.d_model = d_model
        self.ft_layer = ft_layer
        self.ft_input_option = ft_input_option
        self.num_layers = num_layers
        if ft_input_option in ['last', 'last4_sum']:
            cnt_layers = 1
        elif ft_input_option in ['last4_cat']:
            cnt_layers = 4
        if self.num_layers == 0:
            cnt_layers = 1
        if self.ft_layer == 'linear':
            self.ft_decoder = torch.nn.Linear(d_model * cnt_layers, d_model)
        elif self.ft_layer == 'ffn':
            self.ffn1 = torch.nn.Linear(d_model * cnt_layers, ft_d_ff)
            None
            self.dropout = torch.nn.Dropout(ft_drop_rate)
            self.ffn2 = torch.nn.Linear(ft_d_ff, d_model)

    def forward(self, graphbert_layer_output):
        """
        graphbert_output = batch_sz * [CLS, source, target, relation, SEP] *
        [emb_size]
        """
        if self.ft_input_option == 'last':
            graphbert_output = graphbert_layer_output[:, -1, :, :].squeeze(1)
            source_embedding = graphbert_output[:, 0, :].unsqueeze(1)
            destination_embedding = graphbert_output[:, 1, :].unsqueeze(1)
        else:
            no_layers = graphbert_layer_output.size(1)
            if no_layers == 1:
                start_layer = 0
            else:
                start_layer = no_layers - 4
            for ii in range(start_layer, no_layers):
                source_embed = graphbert_layer_output[:, ii, 0, :].unsqueeze(1)
                destination_embed = graphbert_layer_output[:, ii, 1, :].unsqueeze(1)
                if self.ft_input_option == 'last4_cat':
                    try:
                        source_embedding = torch.cat((source_embedding, source_embed), 2)
                        destination_embedding = torch.cat((destination_embedding, destination_embed), 2)
                    except:
                        source_embedding = source_embed
                        destination_embedding = destination_embed
                elif self.ft_input_option == 'last4_sum':
                    try:
                        source_embedding = torch.add(source_embedding, 1, source_embed)
                        destination_embedding = torch.add(destination_embedding, 1, destination_embed)
                    except:
                        source_embedding = source_embed
                        destination_embedding = destination_embed
        if self.ft_layer == 'linear':
            src_embedding = self.ft_decoder(source_embedding)
            dst_embedding = self.ft_decoder(destination_embedding)
        elif self.ft_layer == 'ffn':
            src_embedding = torch.relu(self.dropout(self.ffn1(source_embedding)))
            src_embedding = self.ffn2(src_embedding)
            dst_embedding = torch.relu(self.dropout(self.ffn1(destination_embedding)))
            dst_embedding = self.ffn2(dst_embedding)
        dst_embedding = dst_embedding.transpose(1, 2)
        pred_score = torch.bmm(src_embedding, dst_embedding).squeeze(1)
        pred_score = torch.sigmoid(pred_score)
        return pred_score, src_embedding, dst_embedding.transpose(1, 2)


class BaseModel(nn.Module, metaclass=ABCMeta):

    @classmethod
    def build_model_from_args(cls, args, hg):
        """
        Build the model instance from args and hg.

        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError('Models must implement the build_model_from_args method')

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        """
        The model plays a role of encoder. So the forward will encoder original features into new features.

        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict[str, th.Tensor]
            the dict of heterogeneous feature

        Return
        -------
        out_dic : dict[str, th.Tensor]
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError

    def extra_loss(self):
        """
        Some model want to use L2Norm which is not applied all parameters.

        Returns
        -------
        th.Tensor
        """
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0] + pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        """
        Return the embedding of a model for further analysis.

        Returns
        -------
        numpy.array
        """
        raise NotImplementedError


def adam(grad, state_sum, nodes, lr, device, only_gpu):
    """ calculate gradients according to adam """
    grad_sum = (grad * grad).mean(1)
    if not only_gpu:
        grad_sum = grad_sum.cpu()
    state_sum.index_add_(0, nodes, grad_sum)
    std = state_sum[nodes]
    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
    grad = lr * grad / std_values
    return grad


def async_update(num_threads, model, queue):
    """ asynchronous embedding update """
    torch.set_num_threads(num_threads)
    while True:
        grad_u, grad_v, grad_v_neg, nodes, neg_nodes = queue.get()
        if grad_u is None:
            return
        with torch.no_grad():
            model.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
            model.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)
            if neg_nodes is not None:
                model.v_embeddings.weight.data.index_add_(0, neg_nodes.view(-1), grad_v_neg)


def init_emb2neg_index(walk_length, window_size, negative, batch_size):
    """select embedding of negative nodes from a batch of node embeddings
    for fast negative sampling

    Return
    ------
    index_emb_negu torch.LongTensor : the indices of u_embeddings
    index_emb_negv torch.LongTensor : the indices of v_embeddings
    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2negu = torch.index_select(emb_u, 0, index_emb_negu)
    """
    idx_list_u = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    idx_list_u += [i + b * walk_length] * negative
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    idx_list_u += [i + b * walk_length] * negative
    idx_list_v = list(range(batch_size * walk_length)) * negative * window_size * 2
    random.shuffle(idx_list_v)
    idx_list_v = idx_list_v[:len(idx_list_u)]
    index_emb_negu = torch.LongTensor(idx_list_u)
    index_emb_negv = torch.LongTensor(idx_list_v)
    return index_emb_negu, index_emb_negv


def init_emb2pos_index(walk_length, window_size, batch_size):
    """ select embedding of positive nodes from a batch of node embeddings

    Return
    ------
    index_emb_posu torch.LongTensor : the indices of u_embeddings
    index_emb_posv torch.LongTensor : the indices of v_embeddings
    Usage
    -----
    # emb_u.shape: [batch_size * walk_length, dim]
    batch_emb2posu = torch.index_select(emb_u, 0, index_emb_posu)
    """
    idx_list_u = []
    idx_list_v = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    idx_list_u.append(j + b * walk_length)
                    idx_list_v.append(i + b * walk_length)
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    idx_list_u.append(j + b * walk_length)
                    idx_list_v.append(i + b * walk_length)
    index_emb_posu = torch.LongTensor(idx_list_u)
    index_emb_posv = torch.LongTensor(idx_list_v)
    return index_emb_posu, index_emb_posv


def init_empty_grad(emb_dimension, walk_length, batch_size):
    """ initialize gradient matrix """
    grad_u = torch.zeros((batch_size * walk_length, emb_dimension))
    grad_v = torch.zeros((batch_size * walk_length, emb_dimension))
    return grad_u, grad_v


def init_weight(walk_length, window_size, batch_size):
    """ init context weight """
    weight = []
    for b in range(batch_size):
        for i in range(walk_length):
            for j in range(i - window_size, i):
                if j >= 0:
                    weight.append(1.0 - float(i - j - 1) / float(window_size))
            for j in range(i + 1, i + 1 + window_size):
                if j < walk_length:
                    weight.append(1.0 - float(j - i - 1) / float(window_size))
    return torch.Tensor(weight).unsqueeze(1)


class SkipGramModel(nn.Module):
    """ Negative sampling based skip-gram """

    def __init__(self, emb_size, emb_dimension, walk_length, window_size, batch_size, only_cpu, only_gpu, mix, neg_weight, negative, lr, lap_norm, fast_neg, record_loss, norm, use_context_weight, async_update, num_threads):
        """ initialize embedding on CPU
        Paremeters
        ----------
        emb_size int : number of nodes
        emb_dimension int : embedding dimension
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        batch_size int : number of node sequences in each batch
        only_cpu bool : training with CPU
        only_gpu bool : training with GPU
        mix bool : mixed training with CPU and GPU
        negative int : negative samples for each positve node pair
        neg_weight float : negative weight
        lr float : initial learning rate
        lap_norm float : weight of laplacian normalization
        fast_neg bool : do negative sampling inside a batch
        record_loss bool : print the loss during training
        norm bool : do normalizatin on the embedding after training
        use_context_weight : give different weights to the nodes in a context window
        async_update : asynchronous training
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.walk_length = walk_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.only_cpu = only_cpu
        self.only_gpu = only_gpu
        self.mixed_train = mix
        self.neg_weight = neg_weight
        self.negative = negative
        self.lr = lr
        self.lap_norm = lap_norm
        self.fast_neg = fast_neg
        self.record_loss = record_loss
        self.norm = norm
        self.use_context_weight = use_context_weight
        self.async_update = async_update
        self.num_threads = num_threads
        self.device = torch.device('cpu')
        self.u_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(self.emb_size, self.emb_dimension, sparse=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        self.lookup_table = torch.sigmoid(torch.arange(-6.01, 6.01, 0.01))
        self.lookup_table[0] = 0.0
        self.lookup_table[-1] = 1.0
        if self.record_loss:
            self.logsigmoid_table = torch.log(torch.sigmoid(torch.arange(-6.01, 6.01, 0.01)))
            self.loss = []
        self.index_emb_posu, self.index_emb_posv = init_emb2pos_index(self.walk_length, self.window_size, self.batch_size)
        self.index_emb_negu, self.index_emb_negv = init_emb2neg_index(self.walk_length, self.window_size, self.negative, self.batch_size)
        if self.use_context_weight:
            self.context_weight = init_weight(self.walk_length, self.window_size, self.batch_size)
        self.state_sum_u = torch.zeros(self.emb_size)
        self.state_sum_v = torch.zeros(self.emb_size)
        self.grad_u, self.grad_v = init_empty_grad(self.emb_dimension, self.walk_length, self.batch_size)

    def create_async_update(self):
        """ Set up the async update subprocess.
        """
        self.async_q = Queue(1)
        self.async_p = mp.Process(target=async_update, args=(self.num_threads, self, self.async_q))
        self.async_p.start()

    def finish_async_update(self):
        """ Notify the async update subprocess to quit.
        """
        self.async_q.put((None, None, None, None, None))
        self.async_p.join()

    def share_memory(self):
        """ share the parameters across subprocesses """
        self.u_embeddings.weight.share_memory_()
        self.v_embeddings.weight.share_memory_()
        self.state_sum_u.share_memory_()
        self.state_sum_v.share_memory_()

    def set_device(self, gpu_id):
        """ set gpu device """
        self.device = torch.device('cuda:%d' % gpu_id)
        None
        self.lookup_table = self.lookup_table
        if self.record_loss:
            self.logsigmoid_table = self.logsigmoid_table
        self.index_emb_posu = self.index_emb_posu
        self.index_emb_posv = self.index_emb_posv
        self.index_emb_negu = self.index_emb_negu
        self.index_emb_negv = self.index_emb_negv
        self.grad_u = self.grad_u
        self.grad_v = self.grad_v
        if self.use_context_weight:
            self.context_weight = self.context_weight

    def all_to_device(self, gpu_id):
        """ move all of the parameters to a single GPU """
        self.device = torch.device('cuda:%d' % gpu_id)
        self.set_device(gpu_id)
        self.u_embeddings = self.u_embeddings
        self.v_embeddings = self.v_embeddings
        self.state_sum_u = self.state_sum_u
        self.state_sum_v = self.state_sum_v

    def fast_sigmoid(self, score):
        """ do fast sigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.lookup_table[idx]

    def fast_logsigmoid(self, score):
        """ do fast logsigmoid by looking up in a pre-defined table """
        idx = torch.floor((score + 6.01) / 0.01).long()
        return self.logsigmoid_table[idx]

    def fast_learn(self, batch_walks, neg_nodes=None):
        """ Learn a batch of random walks in a fast way. It has the following features:
            1. It calculating the gradients directly without the forward operation.
            2. It does sigmoid by a looking up table.
        Specifically, for each positive/negative node pair (i,j), the updating procedure is as following:
            score = self.fast_sigmoid(u_embedding[i].dot(v_embedding[j]))
            # label = 1 for positive samples; label = 0 for negative samples.
            u_embedding[i] += (label - score) * v_embedding[j]
            v_embedding[i] += (label - score) * u_embedding[j]
        Parameters
        ----------
        batch_walks list : a list of node sequnces
        lr float : current learning rate
        neg_nodes torch.LongTensor : a long tensor of sampled true negative nodes. If neg_nodes is None,
            then do negative sampling randomly from the nodes in batch_walks as an alternative.
        Usage example
        -------------
        batch_walks = [torch.LongTensor([1,2,3,4]),
                       torch.LongTensor([2,3,4,2])])
        lr = 0.01
        neg_nodes = None
        """
        lr = self.lr
        if isinstance(batch_walks, list):
            nodes = torch.stack(batch_walks)
        elif isinstance(batch_walks, torch.LongTensor):
            nodes = batch_walks
        if self.only_gpu:
            nodes = nodes
            if neg_nodes is not None:
                neg_nodes = neg_nodes
        emb_u = self.u_embeddings(nodes).view(-1, self.emb_dimension)
        emb_v = self.v_embeddings(nodes).view(-1, self.emb_dimension)
        bs = len(batch_walks)
        if bs < self.batch_size:
            index_emb_posu, index_emb_posv = init_emb2pos_index(self.walk_length, self.window_size, bs)
            index_emb_posu = index_emb_posu
            index_emb_posv = index_emb_posv
        else:
            index_emb_posu = self.index_emb_posu
            index_emb_posv = self.index_emb_posv
        emb_pos_u = torch.index_select(emb_u, 0, index_emb_posu)
        emb_pos_v = torch.index_select(emb_v, 0, index_emb_posv)
        pos_score = torch.sum(torch.mul(emb_pos_u, emb_pos_v), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        score = (1 - self.fast_sigmoid(pos_score)).unsqueeze(1)
        if self.record_loss:
            self.loss.append(torch.mean(self.fast_logsigmoid(pos_score)).item())
        if self.lap_norm > 0:
            grad_u_pos = score * emb_pos_v + self.lap_norm * (emb_pos_v - emb_pos_u)
            grad_v_pos = score * emb_pos_u + self.lap_norm * (emb_pos_u - emb_pos_v)
        else:
            grad_u_pos = score * emb_pos_v
            grad_v_pos = score * emb_pos_u
        if self.use_context_weight:
            if bs < self.batch_size:
                context_weight = init_weight(self.walk_length, self.window_size, bs)
            else:
                context_weight = self.context_weight
            grad_u_pos *= context_weight
            grad_v_pos *= context_weight
        if bs < self.batch_size:
            grad_u, grad_v = init_empty_grad(self.emb_dimension, self.walk_length, bs)
            grad_u = grad_u
            grad_v = grad_v
        else:
            self.grad_u = self.grad_u
            self.grad_u.zero_()
            self.grad_v = self.grad_v
            self.grad_v.zero_()
            grad_u = self.grad_u
            grad_v = self.grad_v
        grad_u.index_add_(0, index_emb_posu, grad_u_pos)
        grad_v.index_add_(0, index_emb_posv, grad_v_pos)
        if bs < self.batch_size:
            index_emb_negu, index_emb_negv = init_emb2neg_index(self.walk_length, self.window_size, self.negative, bs)
            index_emb_negu = index_emb_negu
            index_emb_negv = index_emb_negv
        else:
            index_emb_negu = self.index_emb_negu
            index_emb_negv = self.index_emb_negv
        emb_neg_u = torch.index_select(emb_u, 0, index_emb_negu)
        if neg_nodes is None:
            emb_neg_v = torch.index_select(emb_v, 0, index_emb_negv)
        else:
            emb_neg_v = self.v_embeddings.weight[neg_nodes]
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        score = -self.fast_sigmoid(neg_score).unsqueeze(1)
        if self.record_loss:
            self.loss.append(self.negative * self.neg_weight * torch.mean(self.fast_logsigmoid(-neg_score)).item())
        grad_u_neg = self.neg_weight * score * emb_neg_v
        grad_v_neg = self.neg_weight * score * emb_neg_u
        grad_u.index_add_(0, index_emb_negu, grad_u_neg)
        if neg_nodes is None:
            grad_v.index_add_(0, index_emb_negv, grad_v_neg)
        nodes = nodes.view(-1)
        grad_u = adam(grad_u, self.state_sum_u, nodes, lr, self.device, self.only_gpu)
        grad_v = adam(grad_v, self.state_sum_v, nodes, lr, self.device, self.only_gpu)
        if neg_nodes is not None:
            grad_v_neg = adam(grad_v_neg, self.state_sum_v, neg_nodes, lr, self.device, self.only_gpu)
        if self.mixed_train:
            grad_u = grad_u.cpu()
            grad_v = grad_v.cpu()
            if neg_nodes is not None:
                grad_v_neg = grad_v_neg.cpu()
            else:
                grad_v_neg = None
            if self.async_update:
                grad_u.share_memory_()
                grad_v.share_memory_()
                nodes.share_memory_()
                if neg_nodes is not None:
                    neg_nodes.share_memory_()
                    grad_v_neg.share_memory_()
                self.async_q.put((grad_u, grad_v, grad_v_neg, nodes, neg_nodes))
        if not self.async_update:
            self.u_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_u)
            self.v_embeddings.weight.data.index_add_(0, nodes.view(-1), grad_v)
            if neg_nodes is not None:
                self.v_embeddings.weight.data.index_add_(0, neg_nodes.view(-1), grad_v_neg)
        return

    def forward(self, pos_u, pos_v, neg_v):
        """ Do forward and backward. It is designed for future use. """
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=6, min=-6)
        score = -F.logsigmoid(score)
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        return torch.sum(score), torch.sum(neg_score)

    def save_embedding(self, dataset, file_name):
        """ Write embedding to local file. Only used when node ids are numbers.
        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        if self.norm:
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(-1, 1)
        np.save(file_name, embedding)

    def save_embedding_pt(self, dataset, file_name):
        """ For ogb leaderboard.
        """
        try:
            max_node_id = max(dataset.node2id.keys())
            if max_node_id + 1 != self.emb_size:
                None
            embedding = torch.zeros(max_node_id + 1, self.emb_dimension)
            index = torch.LongTensor(list(map(lambda id: dataset.id2node[id], list(range(self.emb_size)))))
            embedding.index_add_(0, index, self.u_embeddings.weight.cpu().data)
            if self.norm:
                embedding /= torch.sqrt(torch.sum(embedding.mul(embedding), 1) + 1e-06).unsqueeze(1)
            torch.save(embedding, file_name)
        except:
            self.save_embedding_pt_dgl_graph(dataset, file_name)

    def save_embedding_pt_dgl_graph(self, dataset, file_name):
        """ For ogb leaderboard """
        embedding = torch.zeros_like(self.u_embeddings.weight.cpu().data)
        valid_seeds = torch.LongTensor(dataset.valid_seeds)
        valid_embedding = self.u_embeddings.weight.cpu().data.index_select(0, valid_seeds)
        embedding.index_add_(0, valid_seeds, valid_embedding)
        if self.norm:
            embedding /= torch.sqrt(torch.sum(embedding.mul(embedding), 1) + 1e-06).unsqueeze(1)
        torch.save(embedding, file_name)

    def save_embedding_txt(self, dataset, file_name):
        """ Write embedding to local file. For future use.
        Parameter
        ---------
        dataset DeepwalkDataset : the dataset
        file_name str : the file name
        """
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        if self.norm:
            embedding /= np.sqrt(np.sum(embedding * embedding, 1)).reshape(-1, 1)
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (self.emb_size, self.emb_dimension))
            for wid in range(self.emb_size):
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (str(dataset.id2node[wid]), e))


MODEL_REGISTRY = {}


def register_model(name):
    """
    New models types can be added to cogdl with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseModel):
            (...)

    Args:
        name (str): the name of the models
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate models ({})'.format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError('Model ({}: {}) must extend BaseModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls
    return register_model_cls


def transform_relation_graph_list(hg, category, identity=True):
    """
        extract subgraph :math:`G_i` from :math:`G` in which
        only edges whose type :math:`R_i` belongs to :math:`\\mathcal{R}`

        Parameters
        ----------
            hg : dgl.heterograph
                Input heterogeneous graph
            category : string
                Type of predicted nodes.
            identity : bool
                If True, the identity matrix will be added to relation matrix set.
    """
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata='h')
    loc = g.ndata[dgl.NTYPE] == category_id
    category_idx = th.arange(g.num_nodes())[loc]
    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    num_edge_type = th.max(etype).item()
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = th.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        sg.edata['w'] = th.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = th.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        sg.edata['w'] = th.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
    return graph_list, g.ndata['h'], category_idx


class fastGTN(BaseModel):
    """
        fastGTN from paper `Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs
        <https://arxiv.org/abs/2106.06218>`__.
        It is the extension paper  of GTN.
        `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\\mathcal{R}`.Then we extract
        the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
        the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
        by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels, in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim, num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm, identity):
        super(fastGTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity
        layers = []
        for i in range(num_layers):
            layers.append(GTConv(num_edge_type, num_channels))
        self.params = nn.ParameterList()
        for i in range(num_channels):
            self.params.append(nn.Parameter(th.Tensor(in_dim, hidden_dim)))
        self.layers = nn.ModuleList(layers)
        self.gcn = GCNConv()
        self.norm = EdgeWeightNorm(norm='right')
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.params is not None:
            for para in self.params:
                nn.init.xavier_uniform_(para)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['h'] = h
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category, identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            A = self.A
            H = []
            for n_c in range(self.num_channels):
                H.append(th.matmul(h, self.params[n_c]))
            for i in range(self.num_layers):
                hat_A = self.layers[i](A)
                for n_c in range(self.num_channels):
                    edge_weight = self.norm(hat_A[n_c], hat_A[n_c].edata['w_sum'])
                    H[n_c] = self.gcn(hat_A[n_c], H[n_c], edge_weight=edge_weight)
            X_ = self.linear1(th.cat(H, dim=1))
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}


class ieHGCNConv(nn.Module):
    """
    The ieHGCN convolution layer.

    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the edge type list of a heterogeneous graph
    activation: str
        the activation function
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """

    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation=F.elu, bias=False, batchnorm=False, dropout=0.0):
        super(ieHGCNConv, self).__init__()
        self.bias = bias
        self.batchnorm = batchnorm
        self.dropout = dropout
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = attn_size
        self.W_self = dglnn.HeteroLinear(node_size, out_size)
        self.W_al = dglnn.HeteroLinear(attn_vector, 1)
        self.W_ar = dglnn.HeteroLinear(attn_vector, 1)
        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        mods = {etype: dglnn.GraphConv(in_size, out_size, norm='right', weight=True, bias=True, allow_zero_in_degree=True) for etype in etypes}
        self.mods = nn.ModuleDict(mods)
        self.linear_q = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.activation = activation
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_size)
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.
        
        Parameters
        ----------
        hg : object or list[block]
            the dgl heterogeneous graph or the list of blocks
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        if hg.is_block:
            src_inputs = h_dict
            dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_dict.items()}
        else:
            src_inputs = h_dict
            dst_inputs = h_dict
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            dst_inputs = self.W_self(dst_inputs)
            query = {}
            key = {}
            attn = {}
            attention = {}
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](dst_inputs[ntype])
                key[ntype] = self.linear_k[ntype](dst_inputs[ntype])
            h_l = self.W_al(key)
            h_r = self.W_ar(query)
            for ntype in hg.dsttypes:
                attention[ntype] = F.elu(h_l[ntype] + h_r[ntype])
                attention[ntype] = attention[ntype].unsqueeze(0)
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                dstdata = self.mods[etype](rel_graph, (src_inputs[srctype], dst_inputs[dsttype]))
                outputs[dsttype].append(dstdata)
                attn[dsttype] = self.linear_k[dsttype](dstdata)
                h_attn = self.W_al(attn)
                attn.clear()
                edge_attention = F.elu(h_attn[dsttype] + h_r[dsttype])
                attention[dsttype] = torch.cat((attention[dsttype], edge_attention.unsqueeze(0)))
            for ntype in hg.dsttypes:
                attention[ntype] = F.softmax(attention[ntype], dim=0)
            rst = {ntype: (0) for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                data = [dst_inputs[ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]

        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)
        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}


class HeteroDotProductPredictor(th.nn.Module):
    """
    References: `documentation of dgl <https://docs.dgl.ai/guide/training-link.html#heterogeneous-graphs>_`
    
    """

    def forward(self, edge_subgraph, x, *args, **kwargs):
        """
        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link
        x: dict[str: th.Tensor]
            the embedding dict. The key only contains the nodes involving with the target link.
    
        Returns
        -------
        score: th.Tensor
            the prediction of the edges in edge_subgraph
        """
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(value)
                score = th.cat(result)
            return score.squeeze()


class HeteroDistMultPredictor(th.nn.Module):

    def forward(self, edge_subgraph, x, r_embedding, *args, **kwargs):
        """
        DistMult factorization (Yang et al. 2014) as the scoring function,
        which is known to perform well on standard link prediction benchmarks when used on its own.
    
        In DistMult, every relation r is associated with a diagonal matrix :math:`R_{r} \\in \\mathbb{R}^{d 	imes d}`
        and a triple (s, r, o) is scored as
    
        .. math::
            f(s, r, o)=e_{s}^{T} R_{r} e_{o}
    
        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link
        x: dict[str: th.Tensor]
            the node embedding dict. The key only contains the nodes involving with the target link.
        r_embedding: th.Tensor
            the all relation types embedding
    
        Returns
        -------
        score: th.Tensor
            the prediction of the edges in edge_subgraph
        """
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            for etype in edge_subgraph.canonical_etypes:
                e = r_embedding[etype[1]]
                n = edge_subgraph.num_edges(etype)
                if 1 == len(edge_subgraph.canonical_etypes):
                    edge_subgraph.edata['e'] = e.expand(n, -1)
                else:
                    edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                edge_subgraph.apply_edges(dgl.function.u_mul_e('x', 'e', 's'), etype=etype)
                edge_subgraph.apply_edges(dgl.function.e_mul_v('s', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(th.sum(value, dim=1))
                score = th.cat(result)
            else:
                score = th.sum(score, dim=1)
            return score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionLayer,
     lambda: ([], {'in_dim': 4, 'hidden_dim': 4, 'dropout': 0.5, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (AvgReadout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Base_model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm1dNode,
     lambda: ([], {'dim_in': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Contrast,
     lambda: ([], {'hidden_dim': 4, 'tau': 4, 'lam': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderLayer,
     lambda: ([], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'd_ff': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (FeedForwardNet,
     lambda: ([], {'in_feats': 4, 'hidden': 4, 'out_feats': 4, 'num_layers': 1, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GCN,
     lambda: ([], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (GeneralLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GraphConvolution,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (GraphGenerator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LogReg,
     lambda: ([], {'ft_in': 4, 'nb_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MetricCalcLayer,
     lambda: ([], {'nhid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiLinearLayer,
     lambda: ([], {'linear_list': [4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Multi_level,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (PoswiseFeedForwardNet,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RelationCrossing,
     lambda: ([], {'in_feats': 4, 'out_feats': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SIGN,
     lambda: ([], {'in_feats': 4, 'hidden': 4, 'out_feats': 4, 'num_hops': 4, 'num_layers': 1, 'dropout': 0.5, 'input_drop': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'hidden_dim': 4, 'attn_drop': 0.5, 'txt': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedAggregator,
     lambda: ([], {'num_feats': 4, 'in_feats': 4, 'num_hops': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_BUPT_GAMMA_OpenHGNN(_paritybench_base):
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

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

