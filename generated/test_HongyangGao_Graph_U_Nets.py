import sys
_module = sys.modules[__name__]
del sys
transform = _module
main = _module
mlp_dropout = _module
network = _module
ops = _module
main = _module
util = _module
main = _module
mol_lib = _module
embedding = _module
mlp = _module
pytorch_util = _module
s2v_lib = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import random


import numpy as np


import torch.nn as nn


import torch.optim as optim


import math


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import torch.nn.functional as F


import scipy.sparse as sp


cmd_opt = argparse.ArgumentParser(description='Argparser for harvard cep')


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        model = GUNet
        self.s2v = model(latent_dim=cmd_args.latent_dim, output_dim=
            cmd_args.out_dim, num_node_feats=cmd_args.feat_dim + cmd_args.
            attr_dim, num_edge_feats=0, k=cmd_args.sortpooling_k)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = self.s2v.dense_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.
            hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout
            )

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False
        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag:
                tmp = torch.from_numpy(batch_graph[i].node_features).type(
                    'torch.FloatTensor')
                concat_feat.append(tmp)
        if node_tag_flag:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)
        if node_feat_flag:
            node_feat = torch.cat(concat_feat, 0)
        if node_feat_flag and node_tag_flag:
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag is False and node_tag_flag:
            node_feat = node_tag
        elif node_feat_flag and node_tag_flag is False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)
        if cmd_args.mode == 'gpu':
            node_feat = node_feat
            labels = labels
        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return embed, labels


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)


def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)
    for name, p in m.named_parameters():
        if not '.' in name:
            _param_init(p)


class MLPRegression(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        pred = self.h2_weights(h1)
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)
        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y
                .size()[0])
            return logits, loss, acc
        else:
            return logits


class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)
        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None
        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data)
                )
        return grad_matrix1, grad_matrix2


def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)


class GUNet(nn.Module):

    def __init__(self, output_dim, num_node_feats, num_edge_feats,
        latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32],
        conv1d_kws=[0, 5]):
        None
        super(GUNet, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i - 1], latent_dim[i])
                )
        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0
            ], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels
            [1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        ks = [0.9, 0.7, 0.6, 0.5]
        self.gUnet = ops.GraphUnet(ks, num_node_feats, 97)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [(torch.Tensor(graph_list[i].degs) + 1) for i in range(
            len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        if isinstance(node_feat, torch.FloatTensor):
            n2n_sp = n2n_sp
            e2n_sp = e2n_sp
            subg_sp = subg_sp
            node_degs = node_degs
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)
        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp,
            subg_sp, graph_sizes, node_degs)
        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp,
        subg_sp, graph_sizes, node_degs):
        """ if exists edge feature, concatenate to node feature vector """
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)
        """ graph convolution layers """
        A = ops.normalize_adj(n2n_sp)
        ver = 2
        if ver == 2:
            cur_message_layer = self.gUnet(A, node_feat)
        else:
            lv = 0
            cur_message_layer = node_feat
            cat_message_layers = []
            while lv < len(self.latent_dim):
                n2npool = gnn_spmm(n2n_sp, cur_message_layer
                    ) + cur_message_layer
                node_linear = self.conv_params[lv](n2npool)
                normalized_linear = node_linear.div(node_degs)
                cur_message_layer = F.tanh(normalized_linear)
                cat_message_layers.append(cur_message_layer)
                lv += 1
            cur_message_layer = torch.cat(cat_message_layers, 1)
        """ sortpooling layer """
        sort_channel = cur_message_layer[:, (-1)]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k,
            self.total_latent_dim)
        if isinstance(node_feat.data, torch.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(subg_sp.size()[0]):
            to_sort = sort_channel[accum_count:accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k - k, self.total_latent_dim)
                if isinstance(node_feat.data, torch.FloatTensor):
                    to_pad = to_pad
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]
        """ traditional 1d convlution and dense layers """
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.
            total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)
        to_dense = conv1d_res.view(len(graph_sizes), -1)
        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense
        return F.relu(reluact_fp)


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=48):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2 * dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim))
            self.up_gcns.append(GCN(dim, dim))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        org_X = X
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        return X


class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k * num_nodes))
        new_X = X[(idx), :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[(idx), :]
        A = A[:, (idx)]
        return A, new_X, idx


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, A, X):
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            None
            sys.exit()
        self.s2v = model(latent_dim=cmd_args.latent_dim, output_dim=
            cmd_args.out_dim, num_node_feats=cmd_args.feat_dim,
            num_edge_feats=0, max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.
            hidden, num_class=cmd_args.num_class)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        concat_feat = []
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            concat_feat += batch_graph[i].node_tags
        concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
        node_feat = torch.zeros(n_nodes, cmd_args.feat_dim)
        node_feat.scatter_(1, concat_feat, 1)
        if cmd_args.mode == 'gpu':
            node_feat = node_feat
            labels = labels
        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return self.mlp(embed, labels)


class Regressor(nn.Module):

    def __init__(self):
        super(Regressor, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            None
            sys.exit()
        self.s2v = model(latent_dim=cmd_args.latent_dim, output_dim=
            cmd_args.out_dim, num_node_feats=MOLLIB.num_node_feats,
            num_edge_feats=MOLLIB.num_edge_feats, max_lv=cmd_args.max_lv)
        self.mlp = MLPRegression(input_size=cmd_args.out_dim, hidden_size=
            cmd_args.hidden)

    def forward(self, batch_graph):
        node_feat, edge_feat, labels = MOLLIB.PrepareFeatureLabel(batch_graph)
        if cmd_args.mode == 'gpu':
            node_feat = node_feat
            edge_feat = edge_feat
            labels = labels
        embed = self.s2v(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)


class EmbedMeanField(nn.Module):

    def __init__(self, latent_dim, output_dim, num_node_feats,
        num_edge_feats, max_lv=3):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.max_lv = max_lv
        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)
        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        if type(node_feat) is torch.FloatTensor:
            n2n_sp = n2n_sp
            e2n_sp = e2n_sp
            subg_sp = subg_sp
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        h = self.mean_field(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp)
        return h

    def mean_field(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp):
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        input_potential = F.relu(input_message)
        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            n2npool = gnn_spmm(n2n_sp, cur_message_layer)
            node_linear = self.conv_params(n2npool)
            merged_linear = node_linear + input_message
            cur_message_layer = F.relu(merged_linear)
            lv += 1
        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = cur_message_layer
        y_potential = gnn_spmm(subg_sp, reluact_fp)
        return F.relu(y_potential)


class EmbedLoopyBP(nn.Module):

    def __init__(self, latent_dim, output_dim, num_node_feats,
        num_edge_feats, max_lv=3):
        super(EmbedLoopyBP, self).__init__()
        self.latent_dim = latent_dim
        self.max_lv = max_lv
        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        self.out_params = nn.Linear(latent_dim, output_dim)
        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        n2e_sp, e2e_sp, e2n_sp, subg_sp = S2VLIB.PrepareLoopyBP(graph_list)
        if type(node_feat) is torch.FloatTensor:
            n2e_sp = n2e_sp
            e2e_sp = e2e_sp
            e2n_sp = e2n_sp
            subg_sp = subg_sp
        node_feat = Variable(node_feat)
        edge_feat = Variable(edge_feat)
        n2e_sp = Variable(n2e_sp)
        e2e_sp = Variable(e2e_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        h = self.loopy_bp(node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp
            )
        return h

    def loopy_bp(self, node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp):
        input_node_linear = self.w_n2l(node_feat)
        input_edge_linear = self.w_e2l(edge_feat)
        n2epool_input = gnn_spmm(n2e_sp, input_node_linear)
        input_message = input_edge_linear + n2epool_input
        input_potential = F.relu(input_message)
        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            e2epool = gnn_spmm(e2e_sp, cur_message_layer)
            edge_linear = self.conv_params(e2epool)
            merged_linear = edge_linear + input_message
            cur_message_layer = F.relu(merged_linear)
            lv += 1
        e2npool = gnn_spmm(e2n_sp, cur_message_layer)
        hidden_msg = F.relu(e2npool)
        out_linear = self.out_params(hidden_msg)
        reluact_fp = F.relu(out_linear)
        y_potential = gnn_spmm(subg_sp, reluact_fp)
        return F.relu(y_potential)


class MLPRegression(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        pred = self.h2_weights(h1)
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)
        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum() / float(y.size()[0]
                )
            return logits, loss, acc
        else:
            return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_HongyangGao_Graph_U_Nets(_paritybench_base):
    pass
    def test_000(self):
        self._check(GCN(*[], **{'in_dim': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(MLPClassifier(*[], **{'input_size': 4, 'hidden_size': 4, 'num_class': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MLPRegression(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

