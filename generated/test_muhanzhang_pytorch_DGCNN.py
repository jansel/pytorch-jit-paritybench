import sys
_module = sys.modules[__name__]
del sys
DGCNN_embedding = _module
gnn_lib = _module
pytorch_util = _module
main = _module
mlp_dropout = _module
util = _module

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


import numpy as np


import torch


import random


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import math


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


class DGCNN(nn.Module):

    def __init__(self, output_dim, num_node_feats, num_edge_feats,
        latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32],
        conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        None
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats,
            latent_dim[0]))
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
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [(torch.Tensor(graph_list[i].degs) + 1) for i in range(
            len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)
        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)
        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.
            FloatTensor):
            n2n_sp = n2n_sp
            e2n_sp = e2n_sp
            subg_sp = subg_sp
            node_degs = node_degs
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.
                cuda.FloatTensor):
                edge_feat = edge_feat
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
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)
        """ graph convolution layers """
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer
            node_linear = self.conv_params[lv](n2npool)
            normalized_linear = node_linear.div(node_degs)
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1
        cur_message_layer = torch.cat(cat_message_layers, 1)
        """ sortpooling layer """
        sort_channel = cur_message_layer[:, (-1)]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k,
            self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.
            cuda.FloatTensor):
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
                if torch.cuda.is_available() and isinstance(node_feat.data,
                    torch.cuda.FloatTensor):
                    to_pad = to_pad
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]
        """ traditional 1d convlution and dense layers """
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.
            total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)
        to_dense = conv1d_res.view(len(graph_sizes), -1)
        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense
        return self.conv1d_activation(reluact_fp)


cmd_opt = argparse.ArgumentParser(description=
    'Argparser for graph_classification')


class MLPRegression(nn.Module):

    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout
        weights_init(self)

    def forward(self, x, y=None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, (0)]
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_muhanzhang_pytorch_DGCNN(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(MLPRegression(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(MLPClassifier(*[], **{'input_size': 4, 'hidden_size': 4, 'num_class': 4}), [torch.rand([4, 4, 4, 4])], {})

