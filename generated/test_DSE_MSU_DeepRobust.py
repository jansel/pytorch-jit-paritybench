import sys
_module = sys.modules[__name__]
del sys
deeprobust = _module
graph = _module
black_box = _module
data = _module
attacked_data = _module
dataset = _module
defense = _module
adv_training = _module
gcn = _module
gcn_preprocess = _module
pgd = _module
prognn = _module
r_gcn = _module
global_attack = _module
base_attack = _module
dice = _module
ig_attack = _module
mettack = _module
nipa = _module
random = _module
topology_attack = _module
rl = _module
env = _module
nipa = _module
nipa_config = _module
nipa_env = _module
nipa_nstep_replay_mem = _module
nipa_q_net_node = _module
nstep_replay_mem = _module
q_net_node = _module
rl_s2v = _module
rl_s2v_config = _module
rl_s2v_env = _module
targeted_attack = _module
base_attack = _module
evaluation = _module
fga = _module
ig_attack = _module
nettack = _module
rl_s2v = _module
rnd = _module
utils = _module
image = _module
BPDA = _module
Nattack = _module
Universal = _module
YOPOpgd = _module
attack = _module
cw = _module
deepfool = _module
fgsm = _module
l2_attack = _module
lbfgs = _module
onepixel = _module
pgd = _module
config = _module
LIDclassifier = _module
TherEncoding = _module
YOPO = _module
base_defense = _module
fast = _module
fgsmtraining = _module
pgdtraining = _module
test_PGD_defense = _module
trade = _module
trades = _module
evaluation_attack = _module
CNN = _module
CNN_multilayer = _module
YOPOCNN = _module
netmodels = _module
densenet = _module
preact_resnet = _module
resnet = _module
train_model = _module
train_resnet = _module
vgg = _module
optimizer = _module
test_adv_train_evasion = _module
test_adv_train_poisoning = _module
test_all = _module
test_dice = _module
test_fga = _module
test_gcn = _module
test_gcn_jaccard = _module
test_gcn_svd = _module
test_ig = _module
test_mettack = _module
test_min_max = _module
test_nettack = _module
test_nipa = _module
test_pgd = _module
test_prognn = _module
test_random = _module
test_rgcn = _module
test_rl_s2v = _module
test_rnd = _module
test1 = _module
test_PGD = _module
test_cw = _module
test_deepfool = _module
test_fgsm = _module
test_lbfgs = _module
test_nattack = _module
test_onepixel = _module
test_pgdtraining = _module
test_trade = _module
test_train = _module
testprint_mnist = _module
setup = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


from torch.nn.modules.module import Module


import scipy.sparse as sp


import numpy as np


import math


import torch.optim as optim


from torch.nn.parameter import Parameter


from copy import deepcopy


import time


from torch.distributions.multivariate_normal import MultivariateNormal


import random


import torch.multiprocessing as mp


from torch import optim


from torch.nn import functional as F


from itertools import count


from scipy.sparse.linalg.eigen.arpack import eigsh


from torch import spmm


import torch.sparse as ts


import torch.utils.data as data_utils


from torch.autograd.gradcheck import zero_gradients


from torch.autograd import Variable


import logging


from numpy import linalg as LA


import scipy.optimize as so


import torch.backends.cudnn as cudnn


from torch.nn.modules.loss import _Loss


from collections import OrderedDict


from typing import Tuple


from typing import List


from typing import Dict


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01,
        weight_decay=0.0005, with_relu=True, with_bias=True, device=None):
        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        """
            adj: normalized adjacency matrix
        """
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None,
        train_iters=200, initialize=True, verbose=False, normalize=True,
        patience=500):
        """
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        """
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels,
                device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)
        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        elif patience < train_iters:
            self._train_with_early_stopping(labels, idx_train, idx_val,
                train_iters, patience, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters,
                verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=
            self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose
        ):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=
            self.weight_decay)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        if verbose:
            None
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val,
        train_iters, patience, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=
            self.weight_decay)
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            None
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        None
        return acc_test

    def _set_parameters():
        pass

    def predict(self, features=None, adj=None):
        """By default, inputs are unnormalized data"""
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.
                    device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            adj = self.estimated_adj + self.estimated_adj.t()
        else:
            adj = self.estimated_adj
        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


class GGCL_F(Module):
    """GGCL: the input is feature"""

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GGCL_F, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features,
            out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features,
            out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, features, adj_norm1, adj_norm2, gamma=1):
        features = F.dropout(features, self.dropout, training=self.training)
        self.miu = F.elu(torch.mm(features, self.weight_miu))
        self.sigma = F.relu(torch.mm(features, self.weight_sigma))
        Att = torch.exp(-gamma * self.sigma)
        miu_out = adj_norm1 @ (self.miu * Att)
        sigma_out = adj_norm2 @ (self.sigma * Att * Att)
        return miu_out, sigma_out


class GGCL_D(Module):
    """GGCL_D: the input is distribution"""

    def __init__(self, in_features, out_features, dropout):
        super(GGCL_D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features,
            out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features,
            out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, miu, sigma, adj_norm1, adj_norm2, gamma=1):
        miu = F.dropout(miu, self.dropout, training=self.training)
        sigma = F.dropout(sigma, self.dropout, training=self.training)
        miu = F.elu(miu @ self.weight_miu)
        sigma = F.relu(sigma @ self.weight_sigma)
        Att = torch.exp(-gamma * sigma)
        mean_out = adj_norm1 @ (miu * Att)
        sigma_out = adj_norm2 @ (sigma * Att * Att)
        return mean_out, sigma


class GaussianConvolution(Module):

    def __init__(self, in_features, out_features):
        super(GaussianConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_miu = Parameter(torch.FloatTensor(in_features,
            out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features,
            out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, previous_miu, previous_sigma, adj_norm1=None,
        adj_norm2=None, gamma=1):
        if adj_norm1 is None and adj_norm2 is None:
            return torch.mm(previous_miu, self.weight_miu), torch.mm(
                previous_miu, self.weight_miu)
        Att = torch.exp(-gamma * previous_sigma)
        M = adj_norm1 @ (previous_miu * Att) @ self.weight_miu
        Sigma = adj_norm2 @ (previous_sigma * Att * Att) @ self.weight_sigma
        return M, Sigma

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class RGCN(Module):

    def __init__(self, nnodes, nfeat, nhid, nclass, gamma=1.0, beta1=0.0005,
        beta2=0.0005, lr=0.01, dropout=0.6, device='cpu'):
        super(RGCN, self).__init__()
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclass = nclass
        self.nhid = nhid // 2
        self.gc1 = GGCL_F(nfeat, nhid, dropout=dropout)
        self.gc2 = GGCL_D(nhid, nclass, dropout=dropout)
        self.dropout = dropout
        self.gaussian = MultivariateNormal(torch.zeros(nnodes, self.nclass),
            torch.diag_embed(torch.ones(nnodes, self.nclass)))
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None

    def forward(self):
        features = self.features
        miu, sigma = self.gc1(features, self.adj_norm1, self.adj_norm2,
            self.gamma)
        miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2,
            self.gamma)
        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(
            sigma + 1e-08)
        return F.log_softmax(output, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None,
        train_iters=200, verbose=True):
        adj, features, labels = utils.to_tensor(adj.todense(), features.
            todense(), labels, device=self.device)
        self.features, self.labels = features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1 / 2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)
        None
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters,
                verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
        self.eval()
        output = self.forward()
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose
        ):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                None
            self.eval()
            output = self.forward()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
        None

    def test(self, idx_test):
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        None

    def _loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        miu1 = self.gc1.miu
        sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-08 + sigma1)
            ).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + torch.norm(self
            .gc1.weight_sigma, 2).pow(2)
        return loss + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def _normalize_adj(self, adj, power=-1 / 2):
        """Row-normalize sparse matrix"""
        A = adj + torch.eye(len(adj)).to(self.device)
        D_power = A.sum(1).pow(power)
        D_power[torch.isinf(D_power)] = 0.0
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power


class BaseAttack(Module):

    def __init__(self, model, nnodes, attack_structure=True,
        attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

    def attack(self):
        pass

    def check_adj(self, adj):
        """
            check if the modified adjacency is symmetric and unweighted
        """
        assert np.abs(adj - adj.T).sum() == 0, 'Input graph is not symmetric'
        assert adj.tocsr().max() == 1, 'Max value should be 1!'
        assert adj.tocsr().min() == 0, 'Min value should be 0!'

    def save_adj(self, root='/tmp/', name='mod_adj'):
        assert self.modified_adj is not None, 'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj
        if type(modified_adj) is torch.Tensor:
            sparse_adj = utils.to_scipy(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)
        else:
            sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root='/tmp/', name='mod_features'):
        assert self.modified_features is not None, 'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features
        if type(modified_features) is torch.Tensor:
            sparse_features = utils.to_scipy(modified_features)
            sp.save_npz(osp.join(root, name), sparse_features)
        else:
            sp.save_npz(osp.join(root, name), modified_features)


class StaticGraph(object):
    graph = None

    @staticmethod
    def get_gsize():
        return torch.Size((len(StaticGraph.graph), len(StaticGraph.graph)))


class GraphNormTool(object):

    def __init__(self, normalize, gm, device):
        self.adj_norm = normalize
        self.gm = gm
        g = StaticGraph.graph
        edges = np.array(g.edges(), dtype=np.int64)
        rev_edges = np.array([edges[:, (1)], edges[:, (0)]], dtype=np.int64)
        edges = np.hstack((edges.T, rev_edges))
        idxes = torch.LongTensor(edges)
        values = torch.ones(idxes.size()[1])
        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.
            get_gsize())
        self.raw_adj = self.raw_adj.to(device)
        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                self.normed_adj = utils.normalize_adj_tensor(self.
                    normed_adj, sparse=True)
            else:
                self.normed_adj = utils.degree_normalize_adj_tensor(self.
                    normed_adj, sparse=True)

    def norm_extra(self, added_adj=None):
        if added_adj is None:
            return self.normed_adj
        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                new_adj = utils.normalize_adj_tensor(new_adj, sparse=True)
            else:
                new_adj = utils.degree_normalize_adj_tensor(new_adj, sparse
                    =True)
        return new_adj


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


class QNetNode(nn.Module):

    def __init__(self, node_features, node_labels, list_action_space,
        n_injected, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm=
        'mean_field', device='cpu'):
        """
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        """
        super(QNetNode, self).__init__()
        self.node_features = node_features
        self.identity = torch.eye(node_labels.max() + 1).to(node_labels.device)
        self.n_injected = n_injected
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm
        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 3, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, 1)
        else:
            self.linear_out = nn.Linear(embed_dim * 3, 1)
        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim)
            )
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=
            device)
        weights_init(self)
        input_dim = (node_labels.max() + 1) * self.n_injected
        self.label_encoder_1 = nn.Linear(input_dim, mlp_hidden)
        self.label_encoder_2 = nn.Linear(mlp_hidden, embed_dim)
        self.device = self.node_features.device

    def to_onehot(self, labels):
        return self.identity[labels].view(-1, self.identity.shape[1])

    def get_label_embedding(self, labels):
        onehot = self.to_onehot(labels).view(1, -1)
        x = F.relu(self.label_encoder_1(onehot))
        x = F.relu(self.label_encoder_2(x))
        return x

    def get_action_label_encoding(self, label):
        onehot = self.to_onehot(label)
        zeros = torch.zeros((onehot.shape[0], self.embed_dim - onehot.shape[1])
            ).to(onehot.device)
        return torch.cat((onehot, zeros), dim=1)

    def get_graph_embedding(self, adj):
        if self.node_features.data.is_sparse:
            node_embed = torch.spmm(self.node_features, self.w_n2l)
        else:
            node_embed = torch.mm(self.node_features, self.w_n2l)
        node_embed += self.bias_n2l
        input_message = node_embed
        node_embed = F.relu(input_message)
        for i in range(self.max_lv):
            n2npool = torch.spmm(adj, node_embed)
            node_linear = self.conv_params(n2npool)
            merged_linear = node_linear + input_message
            node_embed = F.relu(merged_linear)
        graph_embed = torch.mean(node_embed, dim=0, keepdim=True)
        return graph_embed, node_embed

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)
        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows,
            n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp
        return sp

    def forward(self, time_t, states, actions, greedy_acts=False,
        is_inference=False):
        preds = torch.zeros(len(states)).to(self.device)
        batch_graph, modified_labels = zip(*states)
        greedy_actions = []
        with torch.set_grad_enabled(mode=not is_inference):
            for i in range(len(batch_graph)):
                if batch_graph[i] is None:
                    continue
                adj = self.norm_tool.norm_extra(batch_graph[i].
                    get_extra_adj(self.device))
                graph_embed, node_embed = self.get_graph_embedding(adj)
                label_embed = self.get_label_embedding(modified_labels[i])
                if time_t != 2:
                    action_embed = node_embed[actions[i]].view(-1, self.
                        embed_dim)
                else:
                    action_embed = self.get_action_label_encoding(actions[i])
                embed_s = torch.cat((graph_embed, label_embed), dim=1)
                embed_s = embed_s.repeat(len(action_embed), 1)
                embed_s_a = torch.cat((embed_s, action_embed), dim=1)
                if self.mlp_hidden:
                    embed_s_a = F.relu(self.linear_1(embed_s_a))
                raw_pred = self.linear_out(embed_s_a)
                if greedy_acts:
                    action_id = raw_pred.argmax(0)
                    raw_pred = raw_pred.max()
                    greedy_actions.append(actions[i][action_id])
                else:
                    raw_pred = raw_pred.max()
                preds[i] += raw_pred
        return greedy_actions, preds


class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, node_labels,
        list_action_space, n_injected, bilin_q=1, embed_dim=64, mlp_hidden=
        64, max_lv=1, gm='mean_field', device='cpu'):
        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        list_mod = []
        for i in range(0, num_steps):
            list_mod.append(QNetNode(node_features, node_labels,
                list_action_space, n_injected, bilin_q, embed_dim,
                mlp_hidden, max_lv, gm=gm, device=device))
        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False,
        is_inference=False):
        time_t = time_t % 3
        return self.list_mod[time_t](time_t, states, actions, greedy_acts,
            is_inference)


def node_greedy_actions(target_nodes, picked_nodes, list_q, net):
    assert len(target_nodes) == len(list_q)
    actions = []
    values = []
    for i in range(len(target_nodes)):
        region = net.list_action_space[target_nodes[i]]
        if picked_nodes is not None and picked_nodes[i] is not None:
            region = net.list_action_space[picked_nodes[i]]
        if region is None:
            assert list_q[i].size()[0] == net.total_nodes
        else:
            assert len(region) == list_q[i].size()[0]
        val, act = torch.max(list_q[i], dim=0)
        values.append(val)
        if region is not None:
            act = region[act.data.cpu().numpy()[0]]
            act = torch.LongTensor([act])
            actions.append(act)
        else:
            actions.append(act)
    return torch.cat(actions, dim=0).data, torch.cat(values, dim=0).data


class QNetNode(nn.Module):

    def __init__(self, node_features, node_labels, list_action_space,
        bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field',
        device='cpu'):
        """
        bilin_q: bilinear q or not
        mlp_hidden: mlp hidden layer size
        mav_lv: max rounds of message passing
        """
        super(QNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        self.bilin_q = bilin_q
        self.embed_dim = embed_dim
        self.mlp_hidden = mlp_hidden
        self.max_lv = max_lv
        self.gm = gm
        if bilin_q:
            last_wout = embed_dim
        else:
            last_wout = 1
            self.bias_target = Parameter(torch.Tensor(1, embed_dim))
        if mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, mlp_hidden)
            self.linear_out = nn.Linear(mlp_hidden, last_wout)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, last_wout)
        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim)
            )
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=
            device)
        weights_init(self)

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)
        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows,
            n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp
        return sp

    def forward(self, time_t, states, actions, greedy_acts=False,
        is_inference=False):
        if self.node_features.data.is_sparse:
            input_node_linear = torch.spmm(self.node_features, self.w_n2l)
        else:
            input_node_linear = torch.mm(self.node_features, self.w_n2l)
        input_node_linear += self.bias_n2l
        target_nodes, batch_graph, picked_nodes = zip(*states)
        list_pred = []
        prefix_sum = []
        for i in range(len(batch_graph)):
            region = self.list_action_space[target_nodes[i]]
            node_embed = input_node_linear.clone()
            if picked_nodes is not None and picked_nodes[i] is not None:
                with torch.set_grad_enabled(mode=not is_inference):
                    picked_sp = self.make_spmat(self.total_nodes, 1,
                        picked_nodes[i], 0)
                    node_embed += torch.spmm(picked_sp, self.bias_picked)
                    region = self.list_action_space[picked_nodes[i]]
            if not self.bilin_q:
                with torch.set_grad_enabled(mode=not is_inference):
                    target_sp = self.make_spmat(self.total_nodes, 1,
                        target_nodes[i], 0)
                    node_embed += torch.spmm(target_sp, self.bias_target)
            with torch.set_grad_enabled(mode=not is_inference):
                device = self.node_features.device
                adj = self.norm_tool.norm_extra(batch_graph[i].
                    get_extra_adj(device))
                lv = 0
                input_message = node_embed
                node_embed = F.relu(input_message)
                while lv < self.max_lv:
                    n2npool = torch.spmm(adj, node_embed)
                    node_linear = self.conv_params(n2npool)
                    merged_linear = node_linear + input_message
                    node_embed = F.relu(merged_linear)
                    lv += 1
                target_embed = node_embed[(target_nodes[i]), :].view(-1, 1)
                if region is not None:
                    node_embed = node_embed[region]
                graph_embed = torch.mean(node_embed, dim=0, keepdim=True)
                if actions is None:
                    graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
                else:
                    if region is not None:
                        act_idx = region.index(actions[i])
                    else:
                        act_idx = actions[i]
                    node_embed = node_embed[(act_idx), :].view(1, -1)
                embed_s_a = torch.cat((node_embed, graph_embed), dim=1)
                if self.mlp_hidden:
                    embed_s_a = F.relu(self.linear_1(embed_s_a))
                raw_pred = self.linear_out(embed_s_a)
                if self.bilin_q:
                    raw_pred = torch.mm(raw_pred, target_embed)
                list_pred.append(raw_pred)
        if greedy_acts:
            actions, _ = node_greedy_actions(target_nodes, picked_nodes,
                list_pred, self)
        return actions, list_pred


class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, node_labels,
        list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1,
        gm='mean_field', device='cpu'):
        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        list_mod = []
        for i in range(0, num_steps):
            list_mod.append(QNetNode(node_features, node_labels,
                list_action_space, bilin_q, embed_dim, mlp_hidden, max_lv,
                gm=gm, device=device))
        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False,
        is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps
        return self.list_mod[time_t](time_t, states, actions, greedy_acts,
            is_inference)


class BaseAttack(Module):

    def __init__(self, model, nnodes, attack_structure=True,
        attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device
        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes
        self.modified_adj = None
        self.modified_features = None

    def attack(self):
        pass

    def check_adj(self, adj):
        """
            check if the modified adjacency is symmetric and unweighted
        """
        if type(adj) is torch.Tensor:
            adj = adj.cpu().numpy()
        assert np.abs(adj - adj.T).sum() == 0, 'Input graph is not symmetric'
        if sp.issparse(adj):
            assert adj.tocsr().max() == 1, 'Max value should be 1!'
            assert adj.tocsr().min() == 0, 'Min value should be 0!'
        else:
            assert adj.max() == 1, 'Max value should be 1!'
            assert adj.min() == 0, 'Min value should be 0!'


class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof=0.0001):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        y = self.layer(x)
        H = torch.sum(y * p)
        return H


def cal_l2_norm(layer: torch.nn.Module):
    loss = 0.0
    for name, param in layer.named_parameters():
        if name == 'weight':
            loss = loss + 0.5 * torch.norm(param) ** 2
    return loss


class CrossEntropyWithWeightPenlty(_Loss):

    def __init__(self, module, DEVICE, reg_cof=0.0001):
        super(CrossEntropyWithWeightPenlty, self).__init__()
        self.reg_cof = reg_cof
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.module = module

    def __call__(self, pred, label):
        cross_loss = self.criterion(pred, label)
        weight_loss = cal_l2_norm(self.module)
        loss = cross_loss + self.reg_cof * weight_loss
        return loss


class Net(nn.Module):

    def __init__(self, in_channel1=1, out_channel1=32, out_channel2=64, H=
        28, W=28):
        super(Net, self).__init__()
        self.H = H
        self.W = W
        self.out_channel2 = out_channel2
        self.conv1 = nn.Conv2d(in_channels=in_channel1, out_channels=
            out_channel1, kernel_size=5, stride=1, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=out_channel1, out_channels=
            out_channel2, kernel_size=5, stride=1, padding=(2, 2))
        self.fc1 = nn.Linear(int(H / 4) * int(W / 4) * out_channel2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, int(self.H / 4) * int(self.W / 4) * self.out_channel2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, int(self.H / 4) * int(self.W / 4) * self.out_channel2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net(nn.Module):

    def __init__(self, in_channel1=1, out_channel1=32, out_channel2=64, H=
        28, W=28):
        super(Net, self).__init__()
        self.H = H
        self.W = W
        self.out_channel2 = out_channel2
        self.conv1 = nn.Conv2d(in_channels=in_channel1, out_channels=
            out_channel1, kernel_size=5, stride=1, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=out_channel1, out_channels=
            out_channel2, kernel_size=5, stride=1, padding=(2, 2))
        self.fc1 = nn.Linear(int(H / 4) * int(W / 4) * out_channel2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        self.layers[0] = F.relu(self.conv1(x))
        self.layers[1] = F.max_pool2d(x, 2, 2)
        self.layers[2] = F.relu(self.conv2(x))
        self.layers[3] = F.max_pool2d(x, 2, 2)
        self.layers[4] = x.view(-1, int(self.H / 4) * int(self.W / 4) *
            self.out_channel2)
        self.layers[5] = F.relu(self.fc1(x))
        self.layers[6] = self.fc2(x)
        return F.log_softmax(layers[6], dim=1)


class Net(nn.Module):

    def __init__(self, drop=0.5):
        super(Net, self).__init__()
        self.num_channels = 1
        self.num_labels = 10
        activ = nn.ReLU(True)
        self.conv1 = nn.Conv2d(self.num_channels, 32, 3)
        self.layer_one = nn.Sequential(OrderedDict([('conv1', self.conv1),
            ('relu1', activ)]))
        self.feature_extractor = nn.Sequential(OrderedDict([('conv2', nn.
            Conv2d(32, 32, 3)), ('relu2', activ), ('maxpool1', nn.MaxPool2d
            (2, 2)), ('conv3', nn.Conv2d(32, 64, 3)), ('relu3', activ), (
            'conv4', nn.Conv2d(64, 64, 3)), ('relu4', activ), ('maxpool2',
            nn.MaxPool2d(2, 2))]))
        self.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(64 *
            4 * 4, 200)), ('relu1', activ), ('drop', nn.Dropout(drop)), (
            'fc2', nn.Linear(200, 200)), ('relu2', activ), ('fc3', nn.
            Linear(200, self.num_labels))]))
        self.other_layers = nn.ModuleList()
        self.other_layers.append(self.feature_extractor)
        self.other_layers.append(self.classifier)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        y = self.layer_one(input)
        self.layer_one_out = y
        self.layer_one_out.requires_grad_()
        self.layer_one_out.retain_grad()
        features = self.feature_extractor(y)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


class Bottleneck(nn.Module):

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5,
        num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1,
            bias=False)
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(Net, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding
                    =1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_DSE_MSU_DeepRobust(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GGCL_D(*[], **{'in_features': 4, 'out_features': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(GGCL_F(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(GaussianConvolution(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(GraphConvolution(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(PreActBlock(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(PreActBottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(Transition(*[], **{'in_planes': 4, 'out_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

