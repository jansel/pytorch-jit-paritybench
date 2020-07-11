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
base_attack = _module
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
utils = _module
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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.module import Module


import scipy.sparse as sp


import numpy as np


import math


import torch.optim as optim


from torch.nn.parameter import Parameter


from copy import deepcopy


from sklearn.metrics import f1_score


from torch.optim.sgd import SGD


from torch.optim.optimizer import required


from torch.optim import Optimizer


import sklearn


import time


from torch.distributions.multivariate_normal import MultivariateNormal


import random


import torch.multiprocessing as mp


from torch import optim


from torch.nn import functional as F


from itertools import count


from scipy.sparse.linalg.eigen.arpack import eigsh


from torch import spmm


from sklearn.model_selection import train_test_split


import torch.sparse as ts


import torchvision.models as models


import logging


import torchvision


import torchvision.transforms as transforms


import torch.utils.data as data_utils


from torch.autograd.gradcheck import zero_gradients


from torch.autograd import Variable


from abc import ABCMeta


import torch as torch


import copy


from numpy import linalg as LA


import scipy.optimize as so


import torch.backends.cudnn as cudnn


from torchvision import datasets


from torchvision import transforms


from torch.nn.modules.loss import _Loss


from collections import OrderedDict


from typing import Tuple


from typing import List


from typing import Dict


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision import models


import matplotlib.pyplot as plt


import numpy as py


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
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, with_relu=True, with_bias=True, device=None):
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

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500):
        """
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        """
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features
            adj = adj
            labels = labels
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
            self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            None
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class GCNSVD(GCN):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=0.0005, with_relu=True, with_bias=True, device='cpu'):
        super(GCNSVD, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device

    def fit(self, features, adj, labels, idx_train, idx_val=None, k=50, train_iters=200, initialize=True, verbose=True):
        modified_adj = self.truncatedSVD(adj, k=k)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def truncatedSVD(self, data, k=50):
        None
        if sp.issparse(data):
            data = data.asfptype()
            U, S, V = sp.linalg.svds(data, k=k)
            None
            diag_S = np.diag(S)
        else:
            U, S, V = np.linalg.svd(data)
            U = U[:, :k]
            S = S[:k]
            V = V[:k, :]
            None
            diag_S = np.diag(S)
            None
        return U @ diag_S @ V


class GCNJaccard(GCN):

    def __init__(self, nfeat, nhid, nclass, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=0.0005, with_relu=True, with_bias=True, device='cpu'):
        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.binary_feature = binary_feature

    def fit(self, features, adj, labels, idx_train, idx_val=None, threshold=0.01, train_iters=200, initialize=True, verbose=True):
        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def drop_dissimilar_edges(self, features, adj):
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        modified_adj = adj.copy().tolil()
        None
        edges = np.array(modified_adj.nonzero()).T
        removed_cnt = 0
        for edge in tqdm(edges):
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue
            if self.binary_feature:
                J = self._jaccard_similarity(features[n1], features[n2])
                if J < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
        None
        return modified_adj

    def _jaccard_similarity(self, a, b):
        intersection = a.multiply(b).count_nonzero()
        J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        return J

    def _cosine_similarity(self, a, b):
        inner_product = (features[n1] * features[n2]).sum()
        C = inner_product / np.sqrt(np.square(a).sum() + np.square(b).sum())
        return C


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
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
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
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
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
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, previous_miu, previous_sigma, adj_norm1=None, adj_norm2=None, gamma=1):
        if adj_norm1 is None and adj_norm2 is None:
            return torch.mm(previous_miu, self.weight_miu), torch.mm(previous_miu, self.weight_miu)
        Att = torch.exp(-gamma * previous_sigma)
        M = adj_norm1 @ (previous_miu * Att) @ self.weight_miu
        Sigma = adj_norm2 @ (previous_sigma * Att * Att) @ self.weight_sigma
        return M, Sigma

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGCN(Module):

    def __init__(self, nnodes, nfeat, nhid, nclass, gamma=1.0, beta1=0.0005, beta2=0.0005, lr=0.01, dropout=0.6, device='cpu'):
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
        self.gaussian = MultivariateNormal(torch.zeros(nnodes, self.nclass), torch.diag_embed(torch.ones(nnodes, self.nclass)))
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None

    def forward(self):
        features = self.features
        miu, sigma = self.gc1(features, self.adj_norm1, self.adj_norm2, self.gamma)
        miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)
        output = miu + self.gaussian.sample() * torch.sqrt(sigma + 1e-08)
        return F.log_softmax(output, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=True):
        adj, features, labels = utils.to_tensor(adj.todense(), features.todense(), labels, device=self.device)
        self.features, self.labels = features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1 / 2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)
        None
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

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

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
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
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-08 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + torch.norm(self.gc1.weight_sigma, 2).pow(2)
        return loss + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def _normalize_adj(self, adj, power=-1 / 2):
        """Row-normalize sparse matrix"""
        A = adj + torch.eye(len(adj))
        D_power = A.sum(1).pow(power)
        D_power[torch.isinf(D_power)] = 0.0
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power


class BaseAttack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def generate(self, image, label, **kwargs):
        """
        :param x: input takes form (N, C, H, W)
        """
        return input

    def parse_params(self, **kwargs):
        return True

    def check_type_device(self, image, label):
        if self.device == 'cuda':
            image = image
            label = label
            self.model = self.model
        elif self.device == 'cpu':
            image = image.cpu()
            label = label.cpu()
            self.model = self.model.cpu()
        else:
            raise ValueError('Please input cpu or cuda')
        if type(image).__name__ == 'Tensor':
            image = image.float()
            image = image.float().clone().detach().requires_grad_(True)
        elif type(x).__name__ == 'ndarray':
            image = image.astype('float')
            image = torch.tensor(image, requires_grad=True)
        else:
            raise ValueError('Input values only take numpy arrays or torch tensors')
        if type(label).__name__ == 'Tensor':
            label = label.long()
        elif type(label).__name__ == 'ndarray':
            label = label.astype('long')
            label = torch.tensor(y)
        else:
            raise ValueError('Input labels only take numpy arrays or torch tensors')
        self.image = image
        self.label = label
        return True

    def get_or_predict_lable(self, image):
        output = self.model(image)
        pred = output.argmax(dim=1, keepdim=True)
        return pred


class DICE(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        """
        As is described in ADVERSARIAL ATTACKS ON GRAPH NEURAL NETWORKS VIA META LEARNING (ICLR'19),
        'DICE (delete internally, connect externally) is a baseline where, for each perturbation,
        we randomly choose whether to insert or remove an edge. Edges are only removed between
        nodes from the same classes, and only inserted between nodes from different classes.
        """
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, adj, labels, n_perturbations):
        """
        Delete internally, connect externally. This baseline has all true class labels
        (train and test) available.
        """
        None
        modified_adj = adj.tolil()
        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)
        nonzero = set(zip(*adj.nonzero()))
        indices = sp.triu(modified_adj).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1]) if labels[x[0]] == labels[x[1]]]
        remove_indices = np.random.permutation(possible_indices)[:n_remove]
        modified_adj[remove_indices[:, (0)], remove_indices[:, (1)]] = 0
        modified_adj[remove_indices[:, (1)], remove_indices[:, (0)]] = 0
        n_insert = n_perturbations - n_remove
        for i in range(n_insert):
            node1 = np.random.randint(adj.shape[0])
            possible_nodes = [x for x in range(adj.shape[0]) if labels[x] != labels[node1] and modified_adj[x, node1] == 0]
            node2 = possible_nodes[np.random.randint(len(possible_nodes))]
            modified_adj[node1, node2] = 1
            modified_adj[node2, node1] = 1
        self.check_adj(modified_adj)
        return modified_adj

    def sample_forever(self, adj, exclude):
        """
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        """
        while True:
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

    def random_sample_edges(self, adj, n, exclude):
        """
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        """
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]


class IGAttack(BaseAttack):
    """IGAttack: IG-FGSM"""

    def __init__(self, model=None, nnodes=None, feature_shape=None, attack_structure=True, attack_features=True, device='cpu'):
        super(IGAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.modified_adj = None
        self.modified_features = None
        self.target_node = None

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, steps=10):
        self.surrogate.eval()
        self.target_node = target_node
        modified_adj = ori_adj.todense()
        modified_features = ori_features.todense()
        adj, features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)
        adj_norm = utils.normalize_adj_tensor(adj)
        s_e = np.zeros(adj.shape[1])
        s_f = np.zeros(features.shape[1])
        if self.attack_structure:
            s_e = self.calc_importance_edge(features, adj_norm, labels, idx_train, steps)
        if self.attack_features:
            s_f = self.calc_importance_feature(features, adj_norm, labels, idx_train, steps)
        for t in range(n_perturbations):
            s_e_max = np.argmax(s_e)
            s_f_max = np.argmax(s_f)
            if s_e[s_e_max] >= s_f[s_f_max]:
                value = np.abs(1 - modified_adj[target_node, s_e_max])
                modified_adj[target_node, s_e_max] = value
                modified_adj[s_e_max, target_node] = value
                s_e[s_e_max] = 0
            else:
                modified_features[target_node, s_f_max] = np.abs(1 - modified_features[target_node, s_f_max])
                s_f[s_f_max] = 0
        self.modified_adj = sp.csr_matrix(modified_adj)
        self.modified_features = sp.csr_matrix(modified_features)
        self.check_adj(modified_adj)

    def calc_importance_edge(self, features, adj_norm, labels, idx_train, steps):
        baseline_add = adj_norm.clone()
        baseline_remove = adj_norm.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0
        adj_norm.requires_grad = True
        integrated_grad_list = []
        i = self.target_node
        for j in tqdm(range(adj_norm.shape[1])):
            if adj_norm[i][j]:
                scaled_inputs = [(baseline_remove + float(k) / steps * (adj_norm - baseline_remove)) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [(baseline_add - float(k) / steps * (baseline_add - adj_norm)) for k in range(0, steps + 1)]
            _sum = 0
            for new_adj in scaled_inputs:
                output = self.surrogate(features, new_adj)
                loss = F.nll_loss(output[idx_train], labels[idx_train])
                adj_grad = torch.autograd.grad(loss, adj_norm)[0]
                adj_grad = adj_grad[i][j]
                _sum += adj_grad
            if adj_norm[i][j]:
                avg_grad = (adj_norm[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - adj_norm[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        integrated_grad_list[i] = 0
        integrated_grad_list = np.array(integrated_grad_list)
        adj = (adj_norm > 0).cpu().numpy()
        integrated_grad_list = (-2 * adj[self.target_node] + 1) * integrated_grad_list
        return integrated_grad_list

    def calc_importance_feature(self, features, adj_norm, labels, idx_train, steps):
        baseline_add = features.clone()
        baseline_remove = features.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0
        features.requires_grad = True
        integrated_grad_list = []
        i = self.target_node
        for j in tqdm(range(features.shape[1])):
            if features[i][j]:
                scaled_inputs = [(baseline_add + float(k) / steps * (features - baseline_add)) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [(baseline_remove - float(k) / steps * (baseline_remove - features)) for k in range(0, steps + 1)]
            _sum = 0
            for new_features in scaled_inputs:
                output = self.surrogate(new_features, adj_norm)
                loss = F.nll_loss(output[idx_train], labels[idx_train])
                feature_grad = torch.autograd.grad(loss, features)[0]
                feature_grad = feature_grad[i][j]
                _sum += feature_grad
            if features[i][j]:
                avg_grad = (features[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - features[i][j]) * _sum.mean()
            integrated_grad_list.append(avg_grad.detach().item())
        features = (features > 0).cpu().numpy()
        integrated_grad_list = np.array(integrated_grad_list)
        integrated_grad_list = (-2 * features[self.target_node] + 1) * integrated_grad_list
        return integrated_grad_list


class BaseMeta(BaseAttack):

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.modified_adj = None
        self.modified_features = None
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)
        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)
        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        ind = np.diag_indices(self.adj_changes.shape[0])
        adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
        modified_adj = adj_changes_symm + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """
        degrees = modified_adj.sum(0)
        degree_one = degrees == 1
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def self_training_label(self, labels, idx_train):
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges, modified_adj, ori_adj, t_d_min, ll_cutoff)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        adj_meta_grad -= adj_meta_grad.min()
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad * singleton_mask
        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


class Metattack(BaseMeta):

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):
        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias
        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass
        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid))
            w_velocity = torch.zeros(weight.shape)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)
            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid))
                b_velocity = torch.zeros(bias.shape)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)
            previous_size = nhid
        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass))
        output_w_velocity = torch.zeros(output_weight.shape)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)
        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass))
            output_b_velocity = torch.zeros(output_bias.shape)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)
        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)
        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1.0 / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()
        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True
            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True
        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [(self.momentum * v + g) for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [(self.momentum * v + g) for v, g in zip(self.b_velocities, bias_grads)]
            self.weights = [(w - self.lr * v) for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [(b - self.lr * v) for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)
        output = F.log_softmax(hidden, dim=1)
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled
        None
        None
        None
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features
        for i in tqdm(range(perturbations), desc='Perturbing graph'):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)
            adj_meta_score = torch.tensor(0.0)
            feature_meta_score = torch.tensor(0.0)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)
            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += -2 * modified_adj[row_idx][col_idx] + 1
                self.adj_changes.data[col_idx][row_idx] += -2 * modified_adj[row_idx][col_idx] + 1
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.features_changes.data[row_idx][col_idx] += -2 * modified_features[row_idx][col_idx] + 1
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()


class MetaApprox(BaseMeta):

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.01):
        super(MetaApprox, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, device)
        self.lr = lr
        self.train_iters = train_iters
        self.adj_meta_grad = None
        self.features_meta_grad = None
        if self.attack_structure:
            self.adj_grad_sum = torch.zeros(nnodes, nnodes)
        if self.attack_features:
            self.feature_grad_sum = torch.zeros(feature_shape)
        self.with_bias = with_bias
        self.weights = []
        self.biases = []
        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid))
            bias = Parameter(torch.FloatTensor(nhid))
            previous_size = nhid
            self.weights.append(weight)
            self.biases.append(bias)
        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass))
        output_bias = Parameter(torch.FloatTensor(self.nclass))
        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.optimizer = optim.Adam(self.weights + self.biases, lr=lr)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)
        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            hidden = features
            for w, b in zip(self.weights, self.biases):
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled
            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)
            self.optimizer.step()
            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        None
        None

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        self.sparse_features = sp.issparse(ori_features)
        modified_adj = ori_adj
        modified_features = ori_features
        for i in tqdm(range(perturbations), desc='Perturbing graph'):
            self._initialize()
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)
            self.inner_train(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)
            adj_meta_score = torch.tensor(0.0)
            feature_meta_score = torch.tensor(0.0)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)
            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += -2 * modified_adj[row_idx][col_idx] + 1
                self.adj_changes.data[col_idx][row_idx] += -2 * modified_adj[row_idx][col_idx] + 1
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.features_changes.data[row_idx][col_idx] += -2 * modified_features[row_idx][col_idx] + 1
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()


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


model = Net()


class Random(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, add_nodes=False, device='cpu'):
        """
        """
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.add_nodes = add_nodes
        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, adj, n_perturbations, type='add'):
        """
        type: 'add', 'remove', 'flip'
        """
        if self.attack_structure:
            modified_adj = self.perturb_adj(adj, n_perturbations, type)
            return modified_adj

    def perturb_adj(self, adj, n_perturbations, type='add'):
        """
        Randomly add or flip edges.
        """
        modified_adj = adj.tolil()
        type = type.lower()
        assert type in ['add', 'remove', 'flip']
        if type == 'flip':
            edges = self.random_sample_edges(adj, n_perturbations, exclude=set())
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
                modified_adj[n2, n1] = 1 - modified_adj[n2, n1]
        if type == 'add':
            nonzero = set(zip(*adj.nonzero()))
            edges = self.random_sample_edges(adj, n_perturbations, exclude=nonzero)
            for n1, n2 in edges:
                modified_adj[n1, n2] = 1
                modified_adj[n2, n1] = 1
        if type == 'remove':
            nonzero = np.array(adj.nonzero()).T
            indices = np.random.permutation(nonzero)[:n_perturbations].T
            modified_adj[indices[0], indices[1]] = 0
            modified_adj[indices[1], indices[0]] = 0
        self.check_adj(modified_adj)
        return modified_adj

    def perturb_features(self, features, n_perturbations):
        """
        Randomly perturb features.
        """
        None
        return modified_features

    def inject_nodes(self, adj, n_add, n_perturbations):
        """
        For each added node, randomly connect with other nodes.
        """
        None
        raise NotImplementedError
        modified_adj = adj.tolil()
        return modified_adj

    def random_sample_edges(self, adj, n, exclude):
        """
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        """
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        """
        while True:
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))


class PGDAttack(BaseAttack):

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'
        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)))
            self.adj_changes.data.fill_(0)
        if attack_features:
            assert True, 'Topology Attack does not support attack feature'
        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, perturbations):
        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        victim_model.eval()
        epochs = 200
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            self.projection(perturbations)
        self.random_sample(ori_adj, ori_features, labels, idx_train, perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()

    def random_sample(self, ori_adj, ori_features, labels, idx_train, perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)
                None
                if sampled.sum() > perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                None
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == 'CE':
            loss = F.nll_loss(output, labels)
        if self.loss_type == 'CW':
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, perturbations, epsilon=1e-05)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj):
        if self.complementary is None:
            self.complementary = torch.ones_like(ori_adj) - torch.eye(self.nnodes) - ori_adj - ori_adj
        m = torch.zeros((self.nnodes, self.nnodes))
        tril_indices = torch.tril_indices(row=self.nnodes - 1, col=self.nnodes - 1, offset=0)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj
        return modified_adj

    def bisection(self, a, b, perturbations, epsilon):

        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - perturbations
        miu = a
        while b - a >= epsilon:
            miu = (a + b) / 2
            if func(miu) == 0.0:
                break
            if func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
        return miu


class MinMax(PGDAttack):

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        super(MinMax, self).__init__(model, nnodes, loss_type, feature_shape, attack_structure, attack_features, device=device)

    def attack(self, ori_features, ori_adj, labels, idx_train, perturbations):
        victim_model = self.surrogate
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)
        epochs = 200
        victim_model.eval()
        for t in tqdm(range(epochs)):
            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]
            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            self.projection(perturbations)
        self.random_sample(ori_adj, ori_features, labels, idx_train, perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()


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
        self.raw_adj = torch.sparse.FloatTensor(idxes, values, StaticGraph.get_gsize())
        self.raw_adj = self.raw_adj
        self.normed_adj = self.raw_adj.clone()
        if self.adj_norm:
            if self.gm == 'gcn':
                self.normed_adj = utils.normalize_adj_tensor(self.normed_adj, sparse=True)
            else:
                self.normed_adj = utils.degree_normalize_adj_tensor(self.normed_adj, sparse=True)

    def norm_extra(self, added_adj=None):
        if added_adj is None:
            return self.normed_adj
        new_adj = self.raw_adj + added_adj
        if self.adj_norm:
            if self.gm == 'gcn':
                new_adj = utils.normalize_adj_tensor(new_adj, sparse=True)
            else:
                new_adj = utils.degree_normalize_adj_tensor(new_adj, sparse=True)
        return new_adj


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

    def __init__(self, node_features, node_labels, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
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
        self.w_n2l = Parameter(torch.Tensor(node_features.size()[1], embed_dim))
        self.bias_n2l = Parameter(torch.Tensor(embed_dim))
        self.bias_picked = Parameter(torch.Tensor(1, embed_dim))
        self.conv_params = nn.Linear(embed_dim, embed_dim)
        self.norm_tool = GraphNormTool(normalize=True, gm=self.gm, device=device)
        weights_init(self)

    def make_spmat(self, n_rows, n_cols, row_idx, col_idx):
        idxes = torch.LongTensor([[row_idx], [col_idx]])
        values = torch.ones(1)
        sp = torch.sparse.FloatTensor(idxes, values, torch.Size([n_rows, n_cols]))
        if next(self.parameters()).is_cuda:
            sp = sp
        return sp

    def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):
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
                    picked_sp = self.make_spmat(self.total_nodes, 1, picked_nodes[i], 0)
                    node_embed += torch.spmm(picked_sp, self.bias_picked)
                    region = self.list_action_space[picked_nodes[i]]
            if not self.bilin_q:
                with torch.set_grad_enabled(mode=not is_inference):
                    target_sp = self.make_spmat(self.total_nodes, 1, target_nodes[i], 0)
                    node_embed += torch.spmm(target_sp, self.bias_target)
            with torch.set_grad_enabled(mode=not is_inference):
                device = self.node_features.device
                adj = self.norm_tool.norm_extra(batch_graph[i].get_extra_adj(device))
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
            actions, _ = node_greedy_actions(target_nodes, picked_nodes, list_pred, self)
        return actions, list_pred


class NStepQNetNode(nn.Module):

    def __init__(self, num_steps, node_features, node_labels, list_action_space, bilin_q=1, embed_dim=64, mlp_hidden=64, max_lv=1, gm='mean_field', device='cpu'):
        super(NStepQNetNode, self).__init__()
        self.node_features = node_features
        self.node_labels = node_labels
        self.list_action_space = list_action_space
        self.total_nodes = len(list_action_space)
        list_mod = []
        for i in range(0, num_steps):
            list_mod.append(QNetNode(node_features, node_labels, list_action_space, bilin_q, embed_dim, mlp_hidden, max_lv, gm=gm, device=device))
        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):
        assert time_t >= 0 and time_t < self.num_steps
        return self.list_mod[time_t](time_t, states, actions, greedy_acts, is_inference)


class FGA(BaseAttack):

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        super(FGA, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        if self.attack_structure:
            self.adj_changes = Parameter(torch.FloatTensor(nnodes))
            self.adj_changes.data.fill_(0)
        assert not self.attack_features, 'not support attacking features'
        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, features, adj, labels, idx_train, target_node, n_perturbations):
        modified_adj = adj.todense()
        features = features.todense()
        modified_adj, features, labels = utils.to_tensor(modified_adj, features, labels, device=self.device)
        self.surrogate.eval()
        None
        for i in range(n_perturbations):
            modified_row = modified_adj[target_node] + self.adj_changes
            modified_adj[target_node] = modified_row
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            if self.attack_structure:
                output = self.surrogate(features, adj_norm)
                loss = F.nll_loss(output[idx_train], labels[idx_train])
                grad = torch.autograd.grad(loss, self.adj_changes, retain_graph=True)[0]
                grad = grad * (-2 * modified_row + 1)
                grad[target_node] = 0
                grad_argmax = torch.argmax(grad)
            value = -2 * modified_row[grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value
            if self.attack_features:
                pass
        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees
    return ll


def filter_chisquare(ll_ratios, cutoff):
    return ll_ratios < cutoff


def filter_singletons(edges, adj):
    """
    Filter edges that, if removed, would turn one or more nodes into singleton nodes.
    """
    degs = np.squeeze(np.array(np.sum(adj, 0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degs[np.array(edges)] + 2 * (1 - existing_edges[:, (None)]) - 1
    else:
        edge_degrees = degs[np.array(edges)] + 1
    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0


def update_Sx(S_old, n_old, d_old, d_new, d_min):
    """
    Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting
    a single edge.
    """
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)
    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)
    return new_S_d, new_n


class Nettack(BaseAttack):

    def __init__(self, model, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Nettack, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """
        degrees = modified_adj.sum(0)
        degree_one = degrees == 1
        resh = degree_one.repeat(self.nnodes, 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def get_linearized_weight(self):
        surrogate = self.surrogate
        W = surrogate.gc1.weight @ surrogate.gc2.weight
        return W.detach().cpu().numpy()

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers=0, ll_cutoff=0.004, verbose=True):
        """
        Perform an attack on the surrogate model.
        """
        if self.nnodes is None:
            self.nnodes = adj.shape[0]
        self.target_node = target_node
        if type(adj) is torch.Tensor:
            self.ori_adj = utils.to_scipy(adj).tolil()
            self.modified_adj = utils.to_scipy(adj).tolil()
            self.ori_features = utils.to_scipy(features).tolil()
            self.modified_features = utils.to_scipy(features).tolil()
        else:
            self.ori_adj = adj.tolil()
            self.modified_adj = adj.tolil()
            self.ori_features = features.tolil()
            self.modified_features = features.tolil()
        self.cooc_matrix = self.modified_features.T.dot(self.modified_features).tolil()
        attack_features = self.attack_features
        attack_structure = self.attack_structure
        assert not (direct == False and n_influencers == 0), 'indirect mode requires at least one influencer node'
        assert n_perturbations > 0, 'need at least one perturbation'
        assert attack_features or attack_structure, 'either attack_features or attack_structure must be true'
        self.adj_norm = utils.normalize_adj(self.modified_adj)
        self.W = self.get_linearized_weight()
        logits = (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[target_node]
        self.label_u = labels[target_node]
        label_target_onehot = np.eye(int(self.nclass))[labels[target_node]]
        best_wrong_class = (logits - 1000 * label_target_onehot).argmax()
        surrogate_losses = [logits[labels[target_node]] - logits[best_wrong_class]]
        if verbose:
            None
            if attack_structure and attack_features:
                None
            elif attack_features:
                None
            elif attack_structure:
                None
            if direct:
                None
            else:
                None
            None
        if attack_structure:
            degree_sequence_start = self.ori_adj.sum(0).A1
            current_degree_sequence = self.modified_adj.sum(0).A1
            d_min = 2
            S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)
            log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)
        if len(self.influencer_nodes) == 0:
            if not direct:
                infls, add_infls = self.get_attacker_nodes(n_influencers, add_additional_nodes=True)
                self.influencer_nodes = np.concatenate((infls, add_infls)).astype('int')
                self.potential_edges = np.row_stack([np.column_stack((np.tile(infl, self.nnodes - 2), np.setdiff1d(np.arange(self.nnodes), np.array([target_node, infl])))) for infl in self.influencer_nodes])
                if verbose:
                    None
            else:
                influencers = [target_node]
                self.potential_edges = np.column_stack((np.tile(target_node, self.nnodes - 1), np.setdiff1d(np.arange(self.nnodes), target_node)))
                self.influencer_nodes = np.array(influencers)
        self.potential_edges = self.potential_edges.astype('int32')
        for _ in range(n_perturbations):
            if verbose:
                None
            if attack_structure:
                singleton_filter = filter_singletons(self.potential_edges, self.modified_adj)
                filtered_edges = self.potential_edges[singleton_filter]
                deltas = 2 * (1 - self.modified_adj[tuple(filtered_edges.T)].toarray()[0]) - 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, (None)]
                new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
                new_alphas = compute_alpha(new_n, new_S_d, d_min)
                new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
                alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
                new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
                new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)
                powerlaw_filter = filter_chisquare(new_ratios, ll_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]
                a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final, target_node)
                struct_scores = self.struct_score(a_hat_uv_new, self.modified_features @ self.W)
                best_edge_ix = struct_scores.argmin()
                best_edge_score = struct_scores.min()
                best_edge = filtered_edges_final[best_edge_ix]
            if attack_features:
                feature_ixs, feature_scores = self.feature_scores()
                best_feature_ix = feature_ixs[0]
                best_feature_score = feature_scores[0]
            if attack_structure and attack_features:
                if best_edge_score < best_feature_score:
                    if verbose:
                        None
                    change_structure = True
                else:
                    if verbose:
                        None
                    change_structure = False
            elif attack_structure:
                change_structure = True
            elif attack_features:
                change_structure = False
            if change_structure:
                self.modified_adj[tuple(best_edge)] = self.modified_adj[tuple(best_edge[::-1])] = 1 - self.modified_adj[tuple(best_edge)]
                self.adj_norm = utils.normalize_adj(self.modified_adj)
                self.structure_perturbations.append(tuple(best_edge))
                self.feature_perturbations.append(())
                surrogate_losses.append(best_edge_score)
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]
            else:
                self.modified_features[tuple(best_feature_ix)] = 1 - self.modified_features[tuple(best_feature_ix)]
                self.feature_perturbations.append(tuple(best_feature_ix))
                self.structure_perturbations.append(())
                surrogate_losses.append(best_feature_score)

    def get_attacker_nodes(self, n=5, add_additional_nodes=False):
        assert n < self.nnodes - 1, 'number of influencers cannot be >= number of nodes in the graph!'
        neighbors = self.ori_adj[self.target_node].nonzero()[1]
        assert self.target_node not in neighbors
        potential_edges = np.column_stack((np.tile(self.target_node, len(neighbors)), neighbors)).astype('int32')
        a_hat_uv = self.compute_new_a_hat_uv(potential_edges, self.target_node)
        XW = self.modified_features @ self.W
        struct_scores = self.struct_score(a_hat_uv, XW)
        if len(neighbors) >= n:
            influencer_nodes = neighbors[np.argsort(struct_scores)[:n]]
            if add_additional_nodes:
                return influencer_nodes, np.array([])
            return influencer_nodes
        else:
            influencer_nodes = neighbors
            if add_additional_nodes:
                poss_add_infl = np.setdiff1d(np.setdiff1d(np.arange(self.nnodes), neighbors), self.target_node)
                n_possible_additional = len(poss_add_infl)
                n_additional_attackers = n - len(neighbors)
                possible_edges = np.column_stack((np.tile(self.target_node, n_possible_additional), poss_add_infl))
                a_hat_uv_additional = self.compute_new_a_hat_uv(possible_edges, self.target_node)
                additional_struct_scores = self.struct_score(a_hat_uv_additional, XW)
                additional_influencers = poss_add_infl[np.argsort(additional_struct_scores)[-n_additional_attackers:]]
                return influencer_nodes, additional_influencers
            else:
                return influencer_nodes

    def compute_logits(self):
        return (self.adj_norm @ self.adj_norm @ self.modified_features @ self.W)[self.target_node]

    def strongest_wrong_class(self, logits):
        label_u_onehot = np.eye(self.nclass)[self.label_u]
        return (logits - 1000 * label_u_onehot).argmax()

    def feature_scores(self):
        """
        Compute feature scores for all possible feature changes.
        """
        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        surrogate_loss = logits[self.label_u] - logits[best_wrong_class]
        gradient = self.gradient_wrt_x(self.label_u) - self.gradient_wrt_x(best_wrong_class)
        gradients_flipped = sp.lil_matrix(gradient * -1)
        gradients_flipped[self.modified_features.nonzero()] *= -1
        X_influencers = sp.lil_matrix(self.modified_features.shape)
        X_influencers[self.influencer_nodes] = self.modified_features[self.influencer_nodes]
        gradients_flipped = gradients_flipped.multiply(self.cooc_constraint + X_influencers > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T
        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]
        scores = surrogate_loss - grads
        return sorted_ixs[::-1], scores.A1[::-1]

    def compute_cooccurrence_constraint(self, nodes):
        """
        Co-occurrence constraint as described in the paper.

        Parameters
        ----------
        nodes: np.array
            Nodes whose features are considered for change

        Returns
        -------
        np.array [len(nodes), D], dtype bool
            Binary matrix of dimension len(nodes) x D. A 1 in entry n,d indicates that
            we are allowed to add feature d to the features of node n.

        """
        words_graph = self.cooc_matrix.copy()
        D = self.modified_features.shape[1]
        words_graph.setdiag(0)
        words_graph = words_graph > 0
        word_degrees = np.sum(words_graph, axis=0).A1
        inv_word_degrees = np.reciprocal(word_degrees.astype(float) + 1e-08)
        sd = np.zeros([self.nnodes])
        for n in range(self.nnodes):
            n_idx = self.modified_features[(n), :].nonzero()[1]
            sd[n] = np.sum(inv_word_degrees[n_idx.tolist()])
        scores_matrix = sp.lil_matrix((self.nnodes, D))
        for n in nodes:
            common_words = words_graph.multiply(self.modified_features[n])
            idegs = inv_word_degrees[common_words.nonzero()[1]]
            nnz = common_words.nonzero()[0]
            scores = np.array([idegs[nnz == ix].sum() for ix in range(D)])
            scores_matrix[n] = scores
        self.cooc_constraint = sp.csr_matrix(scores_matrix - 0.5 * sd[:, (None)] > 0)

    def gradient_wrt_x(self, label):
        return self.adj_norm.dot(self.adj_norm)[self.target_node].T.dot(self.W[:, (label)].reshape(1, -1))

    def reset(self):
        """
        Reset Nettack
        """
        self.modified_adj = self.ori_adj.copy()
        self.modified_features = self.ori_features.copy()
        self.structure_perturbations = []
        self.feature_perturbations = []
        self.influencer_nodes = []
        self.potential_edges = []
        self.cooc_constraint = None

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.

        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """
        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, (self.label_u)]
        struct_scores = logits_for_correct_class - best_wrong_class_logits
        return struct_scores

    def compute_new_a_hat_uv(self, potential_edges, target_node):
        """
        Compute the updated A_hat_square_uv entries that would result from inserting/deleting the input edges,
        for every edge.

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int
            The edges to check.

        Returns
        -------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix, where P is len(possible_edges).
        """
        edges = np.array(self.modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = self.adj_norm @ self.adj_norm
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, (0)], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = self.modified_adj.sum(0).A1 + 1
        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, edges_set, twohop_ixs, values_before, degrees, potential_edges.astype(np.int32), target_node)
        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, (0)], ixs_arr[:, (1)])), shape=[len(potential_edges), self.nnodes])
        return a_hat_uv


class RND(BaseAttack):

    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        """
        As is described in Adversarial Attacks on Neural Networks for Graph Data (KDD'19),
        'Rnd is an attack in which we modify the structure of the graph. Given our target node v,
        in each step we randomly sample nodes u whose lable is different from v and
        add the edge u,v to the graph structure

        """
        super(RND, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'RND does NOT support attacking features except adding nodes'

    def attack(self, adj, labels, idx_train, target_node, n_perturbations):
        """
        Randomly sample nodes u whose lable is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        """
        None
        modified_adj = adj.tolil()
        row = adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)
        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[:n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[:n_perturbations - len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        self.modified_features = modified_features

    def add_nodes(self, features, adj, labels, idx_train, target_node, n_added=1, n_perturbations=10):
        """
        For each added node, first connect the target node with added fake nodes.
        Then randomly connect the fake nodes with other nodes whose label is
        different from target node. As for the node feature, simply copy arbitary node
        """
        None
        N = adj.shape[0]
        D = features.shape[1]
        modified_adj = self.reshape_mx(adj, shape=(N + n_added, N + n_added))
        modified_features = self.reshape_mx(features, shape=(N + n_added, D))
        diff_labels = [l for l in range(labels.max() + 1) if l != labels[target_node]]
        diff_labels = np.random.permutation(diff_labels)
        possible_nodes = [x for x in idx_train if labels[x] == diff_labels[0]]
        for fake_node in range(N, N + n_added):
            sampled_nodes = np.random.permutation(possible_nodes)[:n_perturbations]
            modified_adj[fake_node, target_node] = 1
            modified_adj[target_node, fake_node] = 1
            for node in sampled_nodes:
                modified_adj[fake_node, node] = 1
                modified_adj[node, fake_node] = 1
            modified_features[fake_node] = features[node]
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        self.modified_features = modified_features

    def reshape_mx(self, mx, shape):
        indices = mx.nonzero()
        return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape).tolil()


def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)
    count = 0
    for x in xs:
        pixels = np.split(x, len(x) / 5)
        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, x_pos, y_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, x_pos, y_pos] = (b / 255.0 - 0.4465) / 0.201
        count += 1
    return imgs


def attack_success(x, img, target_calss, net, targeted_attack=False, print_log=False, device='cuda'):
    attack_image = perturb_image(x, img.clone())
    confidence = F.softmax(net(attack_image)).data.cpu().numpy()[0]
    pred = np.argmax(confidence)
    if print_log:
        None
    if targeted_attack and pred == target_calss or not targeted_attack and pred != target_calss:
        return True


_MACHEPS = np.finfo(np.float64).eps


class DifferentialEvolutionSolver(object):
    """This class implements the differential evolution solver
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population could be used, for
        example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    """
    _binomial = {'best1bin': '_best1', 'randtobest1bin': '_randtobest1', 'currenttobest1bin': '_currenttobest1', 'best2bin': '_best2', 'rand2bin': '_rand2', 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1', 'rand1exp': '_rand1', 'randtobest1exp': '_randtobest1', 'currenttobest1exp': '_currenttobest1', 'best2exp': '_best2', 'rand2exp': '_rand2'}
    __init_error_msg = "The population initialization method must be one of 'latinhypercube' or 'random', or an array of shape (M, N) where N is the number of parameters and M>5"

    def __init__(self, func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, maxfun=np.inf, callback=None, disp=False, polish=True, init='latinhypercube', atol=0):
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError('Please select a valid mutation strategy')
        self.strategy = strategy
        self.callback = callback
        self.polish = polish
        self.tol, self.atol = tol, atol
        self.scale = mutation
        if not np.all(np.isfinite(mutation)) or np.any(np.array(mutation) >= 2) or np.any(np.array(mutation) < 0):
            raise ValueError('The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).')
        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()
        self.cross_over_probability = recombination
        self.func = func
        self.args = args
        self.limits = np.array(bounds, dtype='float').T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError('bounds should be a sequence containing real valued (min, max) pairs for each value in x')
        if maxiter is None:
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:
            maxfun = np.inf
        self.maxfun = maxfun
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])
        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)
        self.num_population_members = max(5, popsize * self.parameter_count)
        self.population_shape = self.num_population_members, self.parameter_count
        self._nfev = 0
        if isinstance(init, string_types):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)
        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator
        segsize = 1.0 / self.num_population_members
        samples = segsize * rng.random_sample(self.population_shape) + np.linspace(0.0, 1.0, self.num_population_members, endpoint=False)[:, (np.newaxis)]
        self.population = np.zeros_like(samples)
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, (j)] = samples[order, j]
        self.population_energies = np.ones(self.num_population_members) * np.inf
        self._nfev = 0

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)
        self.population_energies = np.ones(self.num_population_members) * np.inf
        self._nfev = 0

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The population is clipped to the lower and upper `bounds`.
        """
        popn = np.asfarray(init)
        if np.size(popn, 0) < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2:
            raise ValueError('The population supplied needs to have shape (M, len(x)), where M > 4.')
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)
        self.num_population_members = np.size(self.population, 0)
        self.population_shape = self.num_population_members, self.parameter_count
        self.population_energies = np.ones(self.num_population_members) * np.inf
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return np.std(self.population_energies) / np.abs(np.mean(self.population_energies) + _MACHEPS)

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        for nit in xrange(1, self.maxiter + 1):
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break
            if self.disp:
                None
            convergence = self.convergence
            if self.callback and self.callback(self._scale_parameters(self.population[0]), convergence=self.tol / convergence) is True:
                warning_flag = True
                status_message = 'callback function requested stop early by returning True'
                break
            intol = np.std(self.population_energies) <= self.atol + self.tol * np.abs(np.mean(self.population_energies))
            if warning_flag or intol:
                break
        else:
            status_message = _status_message['maxiter']
            warning_flag = True
        DE_result = OptimizeResult(x=self.x, fun=self.population_energies[0], nfev=self._nfev, nit=nit, message=status_message, success=warning_flag is not True)
        if self.polish:
            result = minimize(self.func, np.copy(DE_result.x), method='L-BFGS-B', bounds=self.limits.T, args=self.args)
            self._nfev += result.nfev
            DE_result.nfev = self._nfev
            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)
        return DE_result

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates])
        energies = self.func(parameters, *self.args)
        self.population_energies = energies
        self._nfev += itersize
        minval = np.argmin(self.population_energies)
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy
        self.population[([0, minval]), :] = self.population[([minval, 0]), :]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()
        if self.dither is not None:
            self.scale = self.random_number_generator.rand() * (self.dither[1] - self.dither[0]) + self.dither[0]
        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))
        trials = np.array([self._mutate(c) for c in range(itersize)])
        for trial in trials:
            self._ensure_constraint(trial)
        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize
        for candidate, (energy, trial) in enumerate(zip(energies, trials)):
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial
        return self.x, self.population_energies[0]

    def next(self):
        """
        Evolve the population by a single generation
        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index in np.where((trial < 0) | (trial > 1))[0]:
            trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])
        rng = self.random_number_generator
        fill_point = rng.randint(0, self.parameter_count)
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))
        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial
        elif self.strategy in self._exponential:
            i = 0
            while i < self.parameter_count and rng.rand() < self.cross_over_probability:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1
            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return self.population[0] + self.scale * (self.population[r0] - self.population[r1])

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return self.population[r0] + self.scale * (self.population[r1] - self.population[r2])

    def _randtobest1(self, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        """
        currenttobest1bin, currenttobest1exp
        """
        r0, r1 = samples[:2]
        bprime = self.population[candidate] + self.scale * (self.population[0] - self.population[candidate] + self.population[r0] - self.population[r1])
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = self.population[0] + self.scale * (self.population[r0] + self.population[r1] - self.population[r2] - self.population[r3])
        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = self.population[r0] + self.scale * (self.population[r1] + self.population[r2] - self.population[r3] - self.population[r4])
        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs


def differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0):
    """Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient
    methods) to find the minimium, and can search large areas of candidate
    space, but often requires larger numbers of function evaluations than
    conventional gradient based techniques.
    The algorithm is due to Storn and Price [1]_.
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'currenttobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'currenttobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals (unless the initial population is
        supplied via the `init` keyword).
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    init : str or array-like, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
            - array specifying the initial population. The array should have
              shape ``(M, len(x))``, where len(x) is the number of parameters.
              `init` is clipped to `bounds` before use.
        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random'
        initializes the population randomly - this has the drawback that
        clustering can occur, preventing the whole of parameter space being
        covered. Use of an array to specify a population subset could be used,
        for example, to create a tight bunch of initial guesses in an location
        where the solution is known to exist, thereby reducing time for
        convergence.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.  If `polish`
        was employed, and a lower minimum was obtained by the polishing, then
        OptimizeResult also contains the ``jac`` attribute.
    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the population
    the algorithm mutates each candidate solution by mixing with other candidate
    solutions to create a trial candidate. There are several strategies [2]_ for
    creating trial candidates, which suit some problems more than others. The
    'best1bin' strategy is a good starting point for many systems. In this
    strategy two members of the population are randomly chosen. Their difference
    is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,
    so far:
    .. math::
        b' = b_0 + mutation * (population[rand0] - population[rand1])
    A trial vector is then constructed. Starting with a randomly chosen 'i'th
    parameter the trial is sequentially filled (in modulo) with parameters from
    `b'` or the original candidate. The choice of whether to use `b'` or the
    original candidate is made with a binomial distribution (the 'bin' in
    'best1bin') - a random number in [0, 1) is generated.  If this number is
    less than the `recombination` constant then the parameter is loaded from
    `b'`, otherwise it is loaded from the original candidate.  The final
    parameter is always loaded from `b'`.  Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.
    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.
    .. versionadded:: 0.15.0
    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.
    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)
    Next find the minimum of the Ackley function
    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).
    >>> from scipy.optimize import differential_evolution
    >>> import numpy as np
    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds)
    >>> result.x, result.fun
    (array([ 0.,  0.]), 4.4408920985006262e-16)
    References
    ----------
    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
    .. [3] http://en.wikipedia.org/wiki/Differential_evolution
    """
    solver = DifferentialEvolutionSolver(func, bounds, args=args, strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation, recombination=recombination, seed=seed, polish=polish, callback=callback, disp=disp, init=init, atol=atol)
    return solver.solve()


def predict_classes(xs, img, target_calss, net, minimize=True, device='cuda'):
    imgs_perturbed = perturb_image(xs, img.clone())
    predictions = F.softmax(net(imgs_perturbed)).data.cpu().numpy()[:, (target_calss)]
    return predictions if minimize else 1 - predictions


class Onepixel(BaseAttack):

    def __init__(self, model, device='cuda'):
        super(Onepixel, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        label = label.type(torch.FloatTensor)
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        return self.one_pixel(self.image, self.label, self.targeted_attack, self.pixels, self.maxiter, self.popsize, self.print_log)

    def get_pred():
        return self.adv_pred

    def parse_params(self, pixels=1, maxiter=100, popsize=400, samples=100, targeted_attack=False, print_log=True, target=0):
        self.pixels = pixels
        self.maxiter = maxiter
        self.popsize = popsize
        self.samples = samples
        self.targeted_attack = targeted_attack
        self.print_log = print_log
        self.target = target
        return True

    def one_pixel(self, img, label, targeted_attack=False, target=0, pixels=1, maxiter=75, popsize=400, print_log=False):
        target_calss = target if targeted_attack else label
        bounds = [(0, 32), (0, 32), (0, 255), (0, 255), (0, 255)] * pixels
        popmul = max(1, popsize / len(bounds))
        predict_fn = lambda xs: predict_classes(xs, img, target_calss, self.model, targeted_attack, self.device)
        callback_fn = lambda x, convergence: attack_success(x, img, target_calss, self.model, targeted_attack, print_log, self.device)
        inits = np.zeros([popmul * len(bounds), len(bounds)])
        for init in inits:
            for i in range(pixels):
                init[i * 5 + 0] = np.random.random() * 32
                init[i * 5 + 1] = np.random.random() * 32
                init[i * 5 + 2] = np.random.normal(128, 127)
                init[i * 5 + 3] = np.random.normal(128, 127)
                init[i * 5 + 4] = np.random.normal(128, 127)
        attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul, recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
        attack_image = perturb_image(attack_result.x, img)
        attack_var = Variable(attack_image, volatile=True)
        predicted_probs = F.softmax(self.model(attack_var)).data.cpu().numpy()[0]
        predicted_class = np.argmax(predicted_probs)
        if not targeted_attack and predicted_class != label or targeted_attack and predicted_class == target_calss:
            self.adv_pred = predict_class
            return attack_image
        return [None]


attack = Onepixel(model, 'cuda')


class NATTACK(BaseAttack):

    def __init__(self, model, device='cuda'):
        super(NATTACK, self).__init__(model, device)
        self.model = model
        self.device = device

    def generate(self, **kwargs):
        assert self.parse_params(**kwargs)
        return attack(self.model, self.dataloader, self.classnum, self.clip_max, self.clip_min, self.epsilon, self.population, self.max_iterations, self.learning_rate, self.sigma, self.target_or_not)
        assert self.check_type_device(self.dataloader)

    def parse_params(self, dataloader, classnum, target_or_not=False, clip_max=1, clip_min=0, epsilon=0.2, population=300, max_iterations=400, learning_rate=2, sigma=0.1):
        self.dataloader = dataloader
        self.classnum = classnum
        self.target_or_not = target_or_not
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.epsilon = epsilon
        self.population = population
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma
        return True


class FASTPGD(BaseAttack):

    def __init__(self, eps=6 / 255.0, sigma=3 / 255.0, nb_iter=20, norm=np.inf, DEVICE=torch.device('cpu'), mean=torch.tensor(np.array([0]).astype(np.float32)[(np.newaxis), :, (np.newaxis), (np.newaxis)]), std=torch.tensor(np.array([1.0]).astype(np.float32)[(np.newaxis), :, (np.newaxis), (np.newaxis)]), random_start=True):
        """
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        """
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DEVICE = DEVICE
        self._mean = mean
        self._std = std
        self.random_start = random_start

    def single_attack(self, net, inp, label, eta, target=None):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        """
        adv_inp = inp + eta
        pred = net(adv_inp)
        if target is not None:
            targets = torch.sum(pred[:, (target)])
            grad_sign = torch.autograd.grad(targets, adv_in, only_inputs=True, retain_graph=False)[0].sign()
        else:
            loss = self.criterion(pred, label)
            grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs=True, retain_graph=False)[0].sign()
        adv_inp = adv_inp + grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std + self._mean
        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        tmp_eta = tmp_adv_inp - tmp_inp
        if self.norm == np.inf:
            tmp_eta = torch.clamp(tmp_eta, -self.eps, self.eps)
        eta = tmp_eta / self._std
        return eta

    def attack(self, net, inp, label, target=None):
        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta
        eta = (eta - self._mean) / self._std
        net.eval()
        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label, eta, target)
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std
        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean
        self._std = self._std
        self.criterion = self.criterion


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable. 
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2
        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t
        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2
        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def onehot_like(a, index, value=1):
    """Creates an array like a, with all values
    set to 0 except one.
    Parameters
    ----------
    a : array_like
        The returned one-hot array will have the same shape
        and dtype as this array
    index : int
        The index that should be set to `value`
    value : single value compatible with a.dtype
        The value to set at the given index
    Returns
    -------
    `numpy.ndarray`
        One-hot array with the given value at the given
        location and zeros everywhere else.
    """
    x = np.zeros_like(a)
    x[index] = value
    return x


class CarliniWagner(BaseAttack):

    def __init__(self, model, device='cuda'):
        super(CarliniWagner, self).__init__(model, device)
        self.model = model
        self.device = device

    def generate(self, image, label, target_label, **kwargs):
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        self.target = target_label
        return self.cw(self.model, self.image, self.label, self.target, self.confidence, self.clip_min, self.clip_min, self.max_iterations, self.initial_const, self.binary_search_steps, self.learning_rate)

    def parse_params(self, classnum=10, confidence=0.0001, clip_max=1, clip_min=0, max_iterations=1000, initial_const=0.01, binary_search_steps=5, learning_rate=1e-05, abort_early=True):
        self.classnum = classnum
        self.confidence = confidence
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.binary_search_steps = binary_search_steps
        self.learning_rate = learning_rate
        self.abort_early = abort_early
        return True

    def cw(self, model, image, label, target, confidence, clip_max, clip_min, max_iterations, initial_const, binary_search_steps, learning_rate):
        """
        parameters:
        :param model: the target model to attack
        :param image: original image to perturb
        :param label: true label of original image
        :param target: target class
        :param confidence: 
        :param clip_max, clip_min:
        :param max_iterations: the maximum number of iteration in cw attack procedure
        :param initial_const:
        :param binary_search_steps:
        :param learning_rate:
        """
        img_tanh = self.to_attack_space(image.cpu())
        img_ori, _ = self.to_model_space(img_tanh)
        img_ori = img_ori
        c = initial_const
        c_low = 0
        c_high = np.inf
        found_adv = False
        last_loss = np.inf
        for step in range(binary_search_steps):
            w = torch.from_numpy(img_tanh.numpy())
            optimizer = AdamOptimizer(img_tanh.shape)
            is_adversarial = False
            for iteration in range(max_iterations):
                img_adv, adv_grid = self.to_model_space(w)
                img_adv = img_adv
                img_adv.requires_grad = True
                output = model.get_logits(img_adv)
                is_adversarial = self.pending_f(img_adv)
                loss, loss_grad = self.loss_function(img_adv, c, self.target, img_ori, self.confidence, self.clip_min, self.clip_max)
                gradient = adv_grid * loss_grad
                w = w + torch.from_numpy(optimizer(gradient.cpu().detach().numpy(), learning_rate)).float()
                if is_adversarial:
                    found_adv = True
            if found_adv:
                c_high = c
            else:
                c_low = c
            if c_high == np.inf:
                c *= 10
            else:
                c = (c_high + c_low) / 2
            if step % 10 == 0:
                None
            if self.abort_early == True and step % 10 == 0 and step > 100:
                None
                if not loss <= 0.9999 * last_loss:
                    break
                last_loss = loss
        return img_adv.detach()

    def loss_function(self, x_p, const, target, reconstructed_original, confidence, min_, max_):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""
        x_p.requires_grad = True
        logits = self.model.get_logits(x_p)
        targetlabel_mask = torch.from_numpy(onehot_like(np.zeros(self.classnum), target)).double()
        secondlargest_mask = torch.from_numpy(np.ones(self.classnum)) - targetlabel_mask
        secondlargest = np.argmax((logits.double() * secondlargest_mask).cpu().detach().numpy())
        is_adv_loss = logits[0][secondlargest] - logits[0][target]
        is_adv_loss += confidence
        if is_adv_loss == 0:
            is_adv_loss_grad = 0
        else:
            is_adv_loss.backward()
            is_adv_loss_grad = x_p.grad
        is_adv_loss = max(0, is_adv_loss)
        s = max_ - min_
        squared_l2_distance = np.sum(((x_p - reconstructed_original) ** 2).cpu().detach().numpy()) / s ** 2
        total_loss = squared_l2_distance + const * is_adv_loss
        squared_l2_distance_grad = 2 / s ** 2 * (x_p - reconstructed_original)
        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad

    def pending_f(self, x_p):
        """Pending is the loss function is less than 0
        """
        targetlabel_mask = torch.from_numpy(onehot_like(np.zeros(self.classnum), self.target))
        secondlargest_mask = torch.from_numpy(np.ones(self.classnum)) - targetlabel_mask
        targetlabel_mask = targetlabel_mask
        secondlargest_mask = secondlargest_mask
        Zx_i = np.max((self.model.get_logits(x_p).double() * secondlargest_mask).cpu().detach().numpy())
        Zx_t = np.max((self.model.get_logits(x_p).double() * targetlabel_mask).cpu().detach().numpy())
        if Zx_i - Zx_t < -self.confidence:
            return True
        else:
            return False

    def to_attack_space(self, x):
        x = x.detach()
        a = (self.clip_min + self.clip_max) / 2
        b = (self.clip_max - self.clip_min) / 2
        x = (x - a) / b
        x = x * 0.999999
        return np.arctanh(x)

    def to_model_space(self, x):
        """Transforms an input from the attack space
        to the model space. This transformation and
        the returned gradient are elementwise."""
        x = np.tanh(x)
        grad = 1 - np.square(x)
        a = (self.clip_min + self.clip_max) / 2
        b = (self.clip_max - self.clip_min) / 2
        x = x * b + a
        grad = grad * b
        return x, grad


def deepfool(model, image, num_classes, overshoot, max_iter, device):
    """
       :param image: 1*H*W*3
            -a batch of Image
       :param model:
            -network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: int
            -num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: float
            -used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: int
            -maximum number of iterations for deepfool (default = 50)
       :return: tensor
            -minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    f_image = model.forward(image).data.cpu().numpy().flatten()
    output = np.array(f_image).flatten().argsort()[::-1]
    output = output[0:num_classes]
    label = output[0]
    input_shape = image.cpu().numpy().shape
    x = copy.deepcopy(image).requires_grad_(True)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    fs = model.forward(x)
    fs_list = [fs[0, output[k]] for k in range(num_classes)]
    current_pred_label = label
    for i in range(max_iter):
        pert = np.inf
        fs[0, output[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        for k in range(1, num_classes):
            zero_gradients(x)
            fs[0, output[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            w_k = cur_grad - grad_orig
            f_k = (fs[0, output[k]] - fs[0, output[0]]).data.cpu().numpy()
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
            if pert_k < pert:
                pert = pert_k
                w = w_k
        r_i = (pert + 0.0001) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
        x = pert_image.detach().requires_grad_(True)
        fs = model.forward(x)
        if not np.argmax(fs.data.cpu().numpy().flatten()) == label:
            break
    r_tot = (1 + overshoot) * r_tot
    return pert_image, r_tot, i


class DeepFool(BaseAttack):

    def __init__(self, model, device='cuda'):
        super(DeepFool, self).__init__(model, device)
        self.model = model
        self.device = device

    def generate(self, image, label, **kwargs):
        assert self.check_type_device(image, label)
        is_cuda = torch.cuda.is_available()
        if is_cuda and self.device == 'cuda':
            self.image = image
            self.model = self.model
        else:
            self.image = image
        assert self.parse_params(**kwargs)
        adv_img, self.r, self.ite = deepfool(self.model, self.image, self.num_classes, self.overshoot, self.max_iteration, self.device)
        return adv_img

    def getpert(self):
        return self.r, self.ite

    def parse_params(self, num_classes=10, overshoot=0.02, max_iteration=50):
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iteration = max_iteration
        return True


def fgm(model, image, label, epsilon, order, clip_min, clip_max, device):
    imageArray = image.cpu().detach().numpy()
    X_fgsm = torch.tensor(imageArray)
    X_fgsm.requires_grad = True
    opt = optim.SGD([X_fgsm], lr=0.001)
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(X_fgsm), label)
    loss.backward()
    if order == np.inf:
        d = epsilon * X_fgsm.grad.data.sign()
    elif order == 2:
        gradient = X_fgsm.grad
        d = torch.zeros(gradient.shape, device=device)
        for i in range(gradient.shape[0]):
            norm_grad = gradient[i].data / LA.norm(gradient[i].data.cpu().numpy())
            d[i] = norm_grad * epsilon
    else:
        raise ValueError('Other p norms may need other algorithms')
    x_adv = X_fgsm + d
    if clip_max == None and clip_min == None:
        clip_max = np.inf
        clip_min = -np.inf
    x_adv = torch.clamp(x_adv, clip_min, clip_max)
    return x_adv


class FGSM(BaseAttack):

    def __init__(self, model, device='cuda'):
        super(FGSM, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        label = label.type(torch.FloatTensor)
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        return fgm(self.model, self.image, self.label, self.epsilon, self.order, self.clip_min, self.clip_max, self.device)

    def parse_params(self, epsilon=0.2, order=np.inf, clip_max=None, clip_min=None):
        self.epsilon = epsilon
        self.order = order
        self.clip_max = clip_max
        self.clip_min = clip_min
        return True


def optimize(model, image, label, target_label, bounds, epsilon, maxiter, class_num, device):
    x_t = image
    x0 = image[0].detach().numpy()
    min_, max_ = bounds
    target_dist = torch.tensor(target_label)
    target_dist = target_dist.unsqueeze_(0).long()
    shape = x0.shape
    dtype = x0.dtype
    x0 = x0.flatten().astype(np.float64)
    n = len(x0)
    bounds = [(min_, max_)] * n

    def distance(x, y):
        x = torch.from_numpy(x).double()
        y = torch.from_numpy(y).double()
        dist_squ = torch.norm(x - y)
        return dist_squ ** 2

    def loss(x, c):
        v1 = distance(x0, x)
        x = torch.tensor(x.astype(dtype).reshape(shape))
        x = x.unsqueeze_(0).float()
        predict = model(x)
        v2 = F.nll_loss(predict, target_dist)
        v = c * v1 + v2
        return np.float64(v)

    def pending_attack(target_model, adv_exp, target_label):
        adv_exp = adv_exp.reshape(shape).astype(dtype)
        adv_exp = torch.from_numpy(adv_exp)
        adv_exp = adv_exp.unsqueeze_(0).float()
        predict1 = target_model(adv_exp)
        label = predict1.argmax(dim=1, keepdim=True)
        if label == target_label:
            return True
        else:
            return False

    def lbfgs_b(c):
        approx_grad_eps = (max_ - min_) / 100
        None
        optimize_output, f, d = so.fmin_l_bfgs_b(loss, x0, args=(c,), approx_grad=True, bounds=bounds, m=15, maxiter=maxiter, factr=10000000000.0, maxls=5, epsilon=approx_grad_eps)
        None
        if np.amax(optimize_output) > max_ or np.amin(optimize_output) < min_:
            logging.info('Input out of bounds (min, max = {}, {}). Performing manual clip.'.format(np.amin(optimize_output), np.amax(optimize_output)))
            optimize_output = np.clip(optimize_output, min_, max_)
        is_adversarial = pending_attack(target_model=model, adv_exp=optimize_output, target_label=target_label)
        return optimize_output, is_adversarial
    c = epsilon
    None
    for i in range(30):
        c = 2 * c
        x_new, is_adversarial = lbfgs_b(c)
        if is_adversarial == False:
            break
    None
    if is_adversarial == True:
        None
        return
    None
    c_low = 0
    c_high = c
    while c_high - c_low >= epsilon:
        None
        c_half = (c_low + c_high) / 2
        x_new, is_adversarial = lbfgs_b(c_half)
        if is_adversarial:
            c_low = c_half
        else:
            c_high = c_half
    x_new, is_adversarial = lbfgs_b(c_low)
    dis = distance(x_new, x0)
    mintargetfunc = loss(x_new, c_low)
    x_new = x_new.astype(dtype)
    x_new = x_new.reshape(shape)
    x_new = torch.from_numpy(x_new).unsqueeze_(0).float()
    return x_new, dis, mintargetfunc


class LBFGS(BaseAttack):

    def __init__(self, model, label, device='cuda'):
        super(LBFGS, self).__init__(model, device)

    def generate(self, image, label, target_label, **kwargs):
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        self.target_label = target_label
        adv_img, self.dist, self.loss = optimize(self.model, self.image, self.label, self.target_label, self.bounds, self.epsilon, self.maxiter, self.class_num, self.device)
        return adv_img

    def distance(self):
        return self.dist

    def loss(self):
        return self.loss

    def parse_params(self, clip_max=1, clip_min=0, class_num=10, epsilon=1e-05, maxiter=20):
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.class_num = class_num
        self.bounds = clip_min, clip_max
        return True


def pgd_attack(model, X, y, epsilon, clip_max, clip_min, num_steps, step_size, print_process):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)
    X_pgd = torch.tensor(imageArray).float()
    X_pgd.requires_grad = True
    for i in range(num_steps):
        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        if print_process:
            None
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()
    return X_pgd


class PGD(BaseAttack):

    def __init__(self, model, device='cuda'):
        super(PGD, self).__init__(model, device)

    def generate(self, image, label, **kwargs):
        label = label.type(torch.FloatTensor)
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)
        return pgd_attack(self.model, self.image, self.label, self.epsilon, self.clip_max, self.clip_min, self.num_steps, self.step_size, self.print_process)

    def parse_params(self, epsilon=0.03, num_steps=40, step_size=0.01, clip_max=1.0, clip_min=0.0, print_process=False):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.print_process = print_process
        return True


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
        self.criterion = nn.CrossEntropyLoss()
        self.module = module

    def __call__(self, pred, label):
        cross_loss = self.criterion(pred, label)
        weight_loss = cal_l2_norm(self.module)
        loss = cross_loss + self.reg_cof * weight_loss
        return loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
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

    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
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
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCNJaccard,
     lambda: ([], {'nfeat': 4, 'nhid': 4, 'nclass': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GCNSVD,
     lambda: ([], {'nfeat': 4, 'nhid': 4, 'nclass': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GGCL_D,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GGCL_F,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GaussianConvolution,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (GraphConvolution,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (Hamiltonian,
     lambda: ([], {'layer': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreActBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreActBottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DSE_MSU_DeepRobust(_paritybench_base):
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

