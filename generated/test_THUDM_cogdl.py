import sys
_module = sys.modules[__name__]
del sys
data = _module
batch = _module
dataloader = _module
dataset = _module
download = _module
extract = _module
in_memory_dataset = _module
makedirs = _module
datasets = _module
edgelist_label = _module
gatne = _module
matlab_matrix = _module
pyg = _module
layers = _module
maggregator = _module
se_layer = _module
models = _module
base_model = _module
emb = _module
deepwalk = _module
dngr = _module
gatne = _module
graph2vec = _module
grarep = _module
hope = _module
line = _module
netmf = _module
netsmf = _module
node2vec = _module
prone = _module
sdne = _module
spectral = _module
nn = _module
gat = _module
gcn = _module
gin = _module
graphsage = _module
infograph = _module
mlp = _module
pyg_cheb = _module
pyg_drgat = _module
pyg_drgcn = _module
pyg_gat = _module
pyg_gcn = _module
pyg_infomax = _module
pyg_unet = _module
options = _module
tasks = _module
base_task = _module
community_detection = _module
graph_classification = _module
influence_maximization = _module
link_prediction = _module
multiplex_link_prediction = _module
node_classification = _module
unsupervised_graph_classification = _module
unsupervised_node_classification = _module
conf = _module
display_data = _module
parallel_train = _module
train = _module
setup = _module
test_pyg = _module

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


import torch


import torch.nn as nn


import numpy as np


import scipy.sparse as sp


import torch.nn.functional as F


from collections import defaultdict


import random


import math


from torch.nn.parameter import Parameter


import copy


import warnings


from scipy import sparse as sp


import itertools


from collections import namedtuple


import torch.multiprocessing as mp


class MeanAggregator(torch.nn.Module):

    def __init__(self, in_channels, out_channels, improved=False, cached=
        False, bias=True):
        super(MeanAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.linear = nn.Linear(in_channels, out_channels, bias)

    @staticmethod
    def norm(x, edge_index):
        deg = torch.sparse.sum(edge_index, 1)
        deg_inv = deg.pow(-1).to_dense()
        x = torch.matmul(edge_index, x)
        x = x.t() * deg_inv
        return x.t()

    def forward(self, x, edge_index, edge_weight=None, bias=True):
        """"""
        x = self.linear(x)
        x = self.norm(x, edge_index)
        return x

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.
            in_channels, self.out_channels)


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.se_channels = se_channels
        self.encoder_decoder = nn.Sequential(nn.Linear(in_channels,
            se_channels), nn.ELU(), nn.Linear(se_channels, in_channels), nn
            .Sigmoid())
        self.reset_parameters()

    def forward(self, x):
        """"""
        x_global = torch.mean(x, dim=0)
        s = self.encoder_decoder(x_global)
        return x * s


class BaseModel(nn.Module):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError(
            'Models must implement the build_model_from_args method')


class GATNEModel(nn.Module):

    def __init__(self, num_nodes, embedding_size, embedding_u_size,
        edge_type_count, dim_a):
        super(GATNEModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a
        self.node_embeddings = Parameter(torch.FloatTensor(num_nodes,
            embedding_size))
        self.node_type_embeddings = Parameter(torch.FloatTensor(num_nodes,
            edge_type_count, embedding_u_size))
        self.trans_weights = Parameter(torch.FloatTensor(edge_type_count,
            embedding_u_size, embedding_size))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(edge_type_count,
            embedding_u_size, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count,
            dim_a, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.node_embeddings.data.uniform_(-1.0, 1.0)
        self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.
            embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.
            embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.
            embedding_size))

    def forward(self, train_inputs, train_types, node_neigh):
        node_embed = self.node_embeddings[train_inputs]
        node_embed_neighbors = self.node_type_embeddings[node_neigh]
        node_embed_tmp = torch.cat([node_embed_neighbors[:, (i), :, (i), :]
            .unsqueeze(1) for i in range(self.edge_type_count)], dim=1)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)
        trans_w = self.trans_weights[train_types]
        trans_w_s1 = self.trans_weights_s1[train_types]
        trans_w_s2 = self.trans_weights_s2[train_types]
        attention = F.softmax(torch.matmul(F.tanh(torch.matmul(
            node_type_embed, trans_w_s1)), trans_w_s2).squeeze()).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w
            ).squeeze()
        last_node_embed = F.normalize(node_embed, dim=1)
        return last_node_embed


class NSLoss(nn.Module):

    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(torch.Tensor([((math.log(k + 2) -
            math.log(k + 1)) / math.log(num_nodes + 1)) for k in range(
            num_nodes)]), dim=0)
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(torch.sigmoid(torch.sum(torch.mul(embs, self
            .weights[label]), 1)))
        negs = torch.multinomial(self.sample_weights, self.num_sampled * n,
            replacement=True).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(torch.log(torch.sigmoid(torch.bmm(noise,
            embs.unsqueeze(2)))), 1).squeeze()
        loss = log_target + sum_log_sampled
        return -loss.sum() / n


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)
            ], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[(0), :] * ctx.N + a._indices()[(1), :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):

    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, edge):
        N = input.size()[0]
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[(edge[(0), :]), :], h[(edge[(1), :]), :]), dim=1
            ).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
            torch.ones(size=(N, 1)))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, edge_index):
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.
            shape[1]).float(), (input.shape[0], input.shape[0]))
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GINLayer(nn.Module):

    def __init__(self, apply_func=None, eps=0, train_eps=True):
        super(GINLayer, self).__init__()
        if train_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([eps]))
        self.apply_func = apply_func

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = remove_self_loops(edge_index)
        edge_weight = torch.ones(edge_index.shape[1]
            ) if edge_weight is None else edge_weight
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (x.shape[0],
            x.shape[0]))
        adj = adj
        out = (1 + self.eps) * x + torch.spmm(adj, x)
        if self.apply_func is not None:
            out = self.apply_func(out)
        return out


class GINMLP(nn.Module):

    def __init__(self, in_feats, out_feats, hidden_dim, num_layers, use_bn=
        True, activation=None):
        super(GINMLP, self).__init__()
        self.use_bn = use_bn
        self.nn = nn.ModuleList()
        if use_bn:
            self.bn = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError('number of MLP layers should be positive')
        elif num_layers == 1:
            self.nn.append(nn.Linear(in_feats, out_feats))
        else:
            for i in range(num_layers - 1):
                if i == 0:
                    self.nn.append(nn.Linear(in_feats, hidden_dim))
                else:
                    self.nn.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    self.bn.append(nn.BatchNorm1d(hidden_dim))
            self.nn.append(nn.Linear(hidden_dim, out_feats))

    def forward(self, x):
        h = x
        for i in range(self.num_layers - 1):
            h = self.nn[i](h)
            if self.use_bn:
                h = self.bn[i](h)
            h = F.relu(h)
        return self.nn[self.num_layers - 1](h)


class SUPEncoder(torch.nn.Module):

    def __init__(self, num_features, dim, num_layers=1):
        super(SUPEncoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)
        nnu = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 
            dim * dim))
        self.conv = NNConv(dim, dim, nnu, aggr='mean', root_weight=False)
        self.gru = nn.GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data, **kwargs):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map.append(out)
        out = self.set2set(out, data.batch)
        return out, feat_map[-1]


class Encoder(nn.Module):

    def __init__(self, in_feats, hidden_dim, num_layers=3, num_mlp_layers=2,
        pooling='sum'):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = GINMLP(in_feats, hidden_dim, hidden_dim,
                    num_mlp_layers, use_bn=True)
            else:
                mlp = GINMLP(hidden_dim, hidden_dim, hidden_dim,
                    num_mlp_layers, use_bn=True)
            self.gnn_layers.append(GINLayer(mlp, eps=0, train_eps=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        if pooling == 'sum':
            self.pooling = scatter_add
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1))
        layer_rep = []
        for i in range(self.num_layers):
            x = F.relu(self.bn_layers[i](self.gnn_layers[i](x, edge_index)))
            layer_rep.append(x)
        pooled_rep = [self.pooling(h, batch, 0) for h in layer_rep]
        node_rep = torch.cat(layer_rep, dim=1)
        graph_rep = torch.cat(pooled_rep, dim=1)
        return graph_rep, node_rep


class FF(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(FF, self).__init__()
        self.block = GINMLP(in_feats, out_feats, out_feats, num_layers=3,
            use_bn=False)
        self.shortcut = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        return F.relu(self.block(x)) + self.shortcut(x)


class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_THUDM_cogdl(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(GraphAttentionLayer(*[], **{'in_features': 4, 'out_features': 4, 'dropout': 0.5, 'alpha': 4}), [torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(GINMLP(*[], **{'in_feats': 4, 'out_feats': 4, 'hidden_dim': 4, 'num_layers': 1}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(FF(*[], **{'in_feats': 4, 'out_feats': 4}), [torch.rand([4, 4, 4, 4])], {})
