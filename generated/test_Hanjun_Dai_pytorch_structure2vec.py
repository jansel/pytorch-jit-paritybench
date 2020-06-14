import sys
_module = sys.modules[__name__]
del sys
main = _module
util = _module
main = _module
mol_lib = _module
embedding = _module
mlp = _module
pytorch_util = _module
s2v_lib = _module

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


import random


import numpy as np


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


cmd_opt = argparse.ArgumentParser(description='Argparser for harvard cep')


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


def get_torch_version():
    return float('.'.join(torch.__version__.split('.')[0:2]))


def is_cuda_float(mat):
    version = get_torch_version()
    if version >= 0.4:
        return mat.is_cuda
    return type(mat) is torch.cuda.FloatTensor


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
        if is_cuda_float(node_feat):
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
        n2e_sp, e2e_sp, e2n_sp, subg_sp = S2VLIB.PrepareLoopyBP(graph_list)
        if is_cuda_float(node_feat):
            n2e_sp = n2e_sp
            e2e_sp = e2e_sp
            e2n_sp = e2n_sp
            subg_sp = subg_sp
        node_feat = Variable(node_feat)
        if edge_feat is not None:
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
        n2epool_input = gnn_spmm(n2e_sp, input_node_linear)
        input_message = n2epool_input
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            input_message += input_edge_linear
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
        if self.output_dim > 0:
            out_linear = self.out_params(hidden_msg)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = hidden_msg
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


def to_scalar(mat):
    version = get_torch_version()
    if version >= 0.4:
        return mat.item()
    return mat.data.cpu().numpy()[0]


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
            acc = to_scalar(pred.eq(y.data.view_as(pred)).sum())
            acc = float(acc) / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Hanjun_Dai_pytorch_structure2vec(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(MLPClassifier(*[], **{'input_size': 4, 'hidden_size': 4, 'num_class': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(MLPRegression(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

