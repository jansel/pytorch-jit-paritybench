import sys
_module = sys.modules[__name__]
del sys
load_data = _module
models = _module
run = _module
semi_utils = _module
utils = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.nn import Parameter


import time


import random


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


import logging


import scipy


import scipy.spatial


from sklearn import preprocessing


from sklearn.metrics.pairwise import euclidean_distances


from scipy.spatial.distance import cdist


class Encoder(torch.nn.Module):

    def __init__(self, name, hiddens, heads, activation, feat_drop, attn_drop, negative_slope, bias):
        super(Encoder, self).__init__()
        self.name = name
        self.hiddens = hiddens
        self.heads = heads
        self.num_layers = len(hiddens) - 1
        self.gnn_layers = nn.ModuleList()
        self.activation = activation
        self.feat_drop = feat_drop
        for l in range(0, self.num_layers):
            if self.name == 'gcn-align':
                self.gnn_layers.append(GCNAlign_GCNConv(in_channels=self.hiddens[l], out_channels=self.hiddens[l + 1], improved=False, cached=True, bias=bias))
            elif self.name == 'naea':
                self.gnn_layers.append(NAEA_GATConv(in_channels=self.hiddens[l] * self.heads[l - 1], out_channels=self.hiddens[l + 1], heads=self.heads[l], concat=True, negative_slope=negative_slope, dropout=attn_drop, bias=bias))
            elif self.name == 'kecg':
                self.gnn_layers.append(KECG_GATConv(in_channels=self.hiddens[l] * self.heads[l - 1], out_channels=self.hiddens[l + 1], heads=self.heads[l], concat=False, negative_slope=negative_slope, dropout=attn_drop, bias=bias))
            else:
                raise NotImplementedError('bad encoder name: ' + self.name)
        if self.name == 'naea':
            self.weight = Parameter(torch.Tensor(self.hiddens[0], self.hiddens[-1]))
            nn.init.xavier_normal_(self.weight)

    def forward(self, edges, x, r=None):
        edges = edges.t()
        if self.name == 'alinet':
            stack = [F.normalize(x, p=2, dim=1)]
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges)
                stack.append(F.normalize(x_, p=2, dim=1))
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            return torch.cat(stack, dim=1)
        elif self.name == 'naea':
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges, r)
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            x = torch.sigmoid(x)
            return x, self.weight
        else:
            for l in range(self.num_layers):
                x = F.dropout(x, p=self.feat_drop, training=self.training)
                x_ = self.gnn_layers[l](x, edges)
                x = x_
                if l != self.num_layers - 1:
                    x = self.activation(x)
            return x

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.name, '\n'.join([layer.__repr__() for layer in self.gnn_layers]))


class Align(torch.nn.Module):

    def __init__(self, p):
        super(Align, self).__init__()
        self.p = p

    def forward(self, e1, e2):
        pred = -torch.norm(e1 - e2, p=self.p, dim=1)
        return pred

    def only_pos_loss(self, e1, r, e2):
        return -F.logsigmoid(-torch.sum(torch.pow(e1 + r - e2, 2), 1)).sum()


class AlignEA(torch.nn.Module):

    def __init__(self, p, feat_drop, params):
        super(AlignEA, self).__init__()
        self.params = params

    def forward(self, e1, r, e2):
        return torch.sum(torch.pow(e1 + r - e2, 2), 1)

    def only_pos_loss(self, e1, r, e2):
        return -F.logsigmoid(-torch.sum(torch.pow(e1 + r - e2, 2), 1)).sum()

    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score - self.params[0]).sum() + self.params[1] * F.relu(self.params[2] - neg_score).sum()


class ComplEx(torch.nn.Module):

    def __init__(self, feat_drop):
        super(ComplEx, self).__init__()
        self.feat_drop = feat_drop

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        e1_r, e1_i = torch.chunk(e1, 2, dim=1)
        r_r, r_i = torch.chunk(r, 2, dim=1)
        e2_r, e2_i = torch.chunk(e2, 2, dim=1)
        return (e1_r * r_r * e2_r + e1_r * r_i * e2_i + e1_i * r_r * e2_i - e1_i * r_i * e2_r).sum(dim=1)


class ConvE(torch.nn.Module):

    def __init__(self, feat_drop, dim, e_num):
        super(ConvE, self).__init__()
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.emb_dim1 = 10
        self.emb_dim2 = dim // self.emb_dim1
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.fc = torch.nn.Linear(4608, dim)

    def forward(self, e1, r, e2):
        e1 = e1.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r = r.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1, r], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = (x * e2).sum(dim=1)
        pred = x
        return pred

    def loss(self, pos_score, neg_score, target):
        return F.binary_cross_entropy_with_logits(torch.cat((pos_score, neg_score), dim=0), torch.cat((target, 1 - target), dim=0))


class DistMult(torch.nn.Module):

    def __init__(self, feat_drop):
        super(DistMult, self).__init__()
        self.feat_drop = feat_drop

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        pred = torch.sum(e1 * r * e2, dim=1)
        return pred


class HAKE(torch.nn.Module):

    def __init__(self, p, feat_drop, dim, params=None):
        super(HAKE, self).__init__()
        self.feat_drop = feat_drop
        self.margin = params
        self.rel_range = (self.margin + 2.0) / (dim / 2)
        self.pi = 3.141592653589793
        self.modulus_weight = nn.Parameter(torch.Tensor([1.0]))
        self.phase_weight = nn.Parameter(torch.Tensor([0.5 * self.rel_range]))

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        phase_head, mod_head = torch.chunk(e1, 2, dim=1)
        phase_relation, mod_relation, bias_relation = torch.chunk(r, 3, dim=1)
        phase_tail, mod_tail = torch.chunk(e2, 2, dim=1)
        phase_head = phase_head / (self.rel_range / self.pi)
        phase_relation = phase_relation / (self.rel_range / self.pi)
        phase_tail = phase_tail / (self.rel_range / self.pi)
        phase_score = phase_head + phase_relation - phase_tail
        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = bias_relation < -mod_relation
        bias_relation[indicator] = -mod_relation[indicator]
        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=1) * self.phase_weight
        r_score = torch.norm(r_score, dim=1) * self.modulus_weight
        return phase_score + r_score

    def loss(self, pos_score, neg_score, target):
        return -(F.logsigmoid(self.margin - pos_score) + F.logsigmoid(neg_score - self.margin)).mean()


class DistMA(torch.nn.Module):

    def __init__(self, feat_drop):
        super(DistMA, self).__init__()
        self.feat_drop = feat_drop

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        return (e1 * r + e1 * e2 + r * e2).sum(dim=1)


class MMEA(torch.nn.Module):

    def __init__(self, feat_drop):
        super(MMEA, self).__init__()
        self.distma = DistMA(feat_drop)
        self.complex = ComplEx(feat_drop)

    def forward(self, e1, r, e2):
        e1_1, e1_2 = torch.chunk(e1, 2, dim=1)
        r_1, r_2 = torch.chunk(r, 2, dim=1)
        e2_1, e2_2 = torch.chunk(e2, 2, dim=1)
        E1 = self.distma(e1_1, r_1, e2_1)
        E2 = self.complex(e1_2, r_2, e2_2)
        E = E1 + E2
        return torch.cat((E1.view(-1, 1), E2.view(-1, 1), E.view(-1, 1)), dim=1)

    def loss(self, pos_score, neg_score, target):
        E1_p_s, E2_p_s, E_p_s = torch.chunk(pos_score, 3, dim=1)
        E1_n_s, E2_n_s, E_n_s = torch.chunk(neg_score, 3, dim=1)
        return -F.logsigmoid(E1_p_s).sum() - F.logsigmoid(-1.0 * E1_n_s).sum() - F.logsigmoid(E2_p_s).sum() - F.logsigmoid(-1.0 * E2_n_s).sum() - F.logsigmoid(E_p_s).sum() - F.logsigmoid(-1.0 * E_n_s).sum()


class MTransE_Align(torch.nn.Module):

    def __init__(self, p, dim, mode='sa4'):
        super(MTransE_Align, self).__init__()
        self.p = p
        self.mode = mode
        if self.mode == 'sa1':
            pass
        elif self.mode == 'sa3':
            self.weight = Parameter(torch.Tensor(dim))
            nn.init.xavier_normal_(self.weight)
        elif self.mode == 'sa4':
            self.weight = Parameter(torch.Tensor(dim, dim))
            nn.init.orthogonal_(self.weight)
            self.I = Parameter(torch.eye(dim), requires_grad=False)
        else:
            raise NotImplementedError

    def forward(self, e1, e2):
        if self.mode == 'sa1':
            pred = -torch.norm(e1 - e2, p=self.p, dim=1)
        elif self.mode == 'sa3':
            pred = -torch.norm(e1 + self.weight - e2, p=self.p, dim=1)
        elif self.mode == 'sa4':
            pred = -torch.norm(torch.matmul(e1, self.weight) - e2, p=self.p, dim=1)
        else:
            raise NotImplementedError
        return pred

    def mapping(self, emb):
        return torch.matmul(emb, self.weight)

    def only_pos_loss(self, e1, e2):
        if self.p == 1:
            map_loss = torch.sum(torch.abs(torch.matmul(e1, self.weight) - e2), dim=1).sum()
        else:
            map_loss = torch.sum(torch.pow(torch.matmul(e1, self.weight) - e2, 2), dim=1).sum()
        orthogonal_loss = torch.pow(torch.matmul(self.weight, self.weight.t()) - self.I, 2).sum(dim=1).sum(dim=0)
        return map_loss + orthogonal_loss

    def __repr__(self):
        return '{}(mode={})'.format(self.__class__.__name__, self.mode)


class N_R_Align(torch.nn.Module):

    def __init__(self, params):
        super(N_R_Align, self).__init__()
        self.params = params
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-06)

    def forward(self, e1, e2, n1, n2):
        return self.params * torch.sigmoid(self.cos_sim(n1, n2)) + (1 - self.params) * torch.sigmoid(self.cos_sim(e1, e2))

    def loss(self, pos_score, neg_score, target):
        return -torch.log(pos_score).sum()


class N_TransE(torch.nn.Module):

    def __init__(self, p, params):
        super(N_TransE, self).__init__()
        self.p = p
        self.params = params

    def forward(self, e1, r, e2):
        pred = -torch.norm(e1 + r - e2, p=self.p, dim=1)
        return pred

    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score + self.params[0] - neg_score).sum() + self.params[1] * F.relu(pos_score - self.params[2]).sum()


class RotatE(torch.nn.Module):

    def __init__(self, p, feat_drop, dim, params=None):
        super(RotatE, self).__init__()
        self.feat_drop = feat_drop
        self.margin = params
        self.rel_range = (self.margin + 2.0) / (dim / 2)
        self.pi = 3.141592653589793

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        re_head, im_head = torch.chunk(e1, 2, dim=1)
        re_tail, im_tail = torch.chunk(e2, 2, dim=1)
        r = r / (self.rel_range / self.pi)
        re_relation = torch.cos(r)
        im_relation = torch.sin(r)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        pred = score.norm(dim=0).sum(dim=-1)
        return pred

    def loss(self, pos_score, neg_score, target):
        return -(F.logsigmoid(self.margin - pos_score) + F.logsigmoid(neg_score - self.margin)).mean()


class TransE(torch.nn.Module):

    def __init__(self, p, feat_drop, transe_sp=False):
        super(TransE, self).__init__()
        self.p = p
        self.feat_drop = feat_drop
        self.transe_sp = transe_sp

    def forward(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        if self.transe_sp:
            pred = -F.normalize(e1 + r - e2, p=2, dim=1).sum(dim=1)
        else:
            pred = -torch.norm(e1 + r - e2, p=self.p, dim=1)
        return pred

    def only_pos_loss(self, e1, r, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        if self.p == 1:
            return torch.sum(torch.abs(e1 + r - e2), dim=1).sum()
        else:
            return torch.sum(torch.pow(e1 + r - e2, 2), dim=1).sum()


class MLP(torch.nn.Module):

    def __init__(self, act=torch.relu, hiddens=[], l2_norm=False):
        super(MLP, self).__init__()
        self.hiddens = hiddens
        self.fc_layers = nn.ModuleList()
        self.num_layers = len(self.hiddens) - 1
        self.activation = act
        self.l2_norm = l2_norm
        for i in range(self.num_layers):
            self.fc_layers.append(nn.Linear(self.hiddens[i], self.hiddens[i + 1]))

    def forward(self, e):
        for i, fc in enumerate(self.fc_layers):
            if self.l2_norm:
                e = F.normalize(e, p=2, dim=1)
            e = fc(e)
            if i != self.num_layers - 1:
                e = self.activation(e)
        return e


class TransEdge(torch.nn.Module):

    def __init__(self, p, feat_drop, dim, mode, params):
        super(TransEdge, self).__init__()
        self.func = TransE(p, feat_drop)
        self.params = params
        self.mode = mode
        if self.mode == 'cc':
            self.mlp_1 = MLP(act=torch.tanh, hiddens=[2 * dim, dim], l2_norm=True)
            self.mlp_2 = MLP(act=torch.tanh, hiddens=[2 * dim, dim], l2_norm=True)
            self.mlp_3 = MLP(act=torch.tanh, hiddens=[2 * dim, dim], l2_norm=True)
        elif self.mode == 'cp':
            self.mlp = MLP(act=torch.tanh, hiddens=[2 * dim, dim], l2_norm=False)
        else:
            raise NotImplementedError

    def forward(self, e1, r, e2):
        if self.mode == 'cc':
            hr = torch.cat((e1, r), dim=1)
            rt = torch.cat((r, e2), dim=1)
            hr = F.normalize(self.mlp_2(hr), p=2, dim=1)
            rt = F.normalize(self.mlp_3(rt), p=2, dim=1)
            crs = F.normalize(torch.cat((hr, rt), dim=1), p=2, dim=1)
            psi = self.mlp_1(crs)
        elif self.mode == 'cp':
            ht = torch.cat((e1, e2), dim=1)
            ht = F.normalize(self.mlp(ht), p=2, dim=1)
            psi = r - torch.sum(r * ht, dim=1, keepdim=True) * ht
        else:
            raise NotImplementedError
        psi = torch.tanh(psi)
        return -self.func(e1, psi, e2)

    def loss(self, pos_score, neg_score, target):
        return F.relu(pos_score - self.params[0]).sum() + self.params[1] * F.relu(self.params[2] - neg_score).sum()


class TransH(torch.nn.Module):

    def __init__(self, p, feat_drop, l2_norm=True):
        super(TransH, self).__init__()
        self.p = p
        self.feat_drop = feat_drop
        self.l2_norm = l2_norm

    def forward(self, e1, r, e2):
        if self.l2_norm:
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        d_r, n_r = torch.chunk(r, 2, dim=1)
        if self.l2_norm:
            d_r = F.normalize(d_r, p=2, dim=1)
            n_r = F.normalize(n_r, p=2, dim=1)
        e1_ = e1 - torch.sum(e1 * n_r, dim=1, keepdim=True) * n_r
        e2_ = e2 - torch.sum(e2 * n_r, dim=1, keepdim=True) * n_r
        pred = -torch.norm(e1_ + d_r - e2_, p=self.p, dim=1)
        return pred


class TransR(torch.nn.Module):

    def __init__(self, p, feat_drop):
        super(TransR, self).__init__()
        self.p = p
        self.feat_drop = feat_drop

    def forward(self, e1, rM, e2):
        e1 = F.dropout(e1, p=self.feat_drop, training=self.training)
        r, M_r = rM[:, :e1.size(1)], rM[:, e1.size(1):]
        M_r = M_r.view(e1.size(0), e1.size(1), e1.size(1))
        hr = torch.matmul(e1.view(e1.size(0), 1, e1.size(1)), M_r).view(e1.size(0), -1)
        tr = torch.matmul(e2.view(e2.size(0), 1, e2.size(1)), M_r).view(e2.size(0), -1)
        hr = F.normalize(hr, p=2, dim=1)
        tr = F.normalize(tr, p=2, dim=1)
        pred = -torch.norm(hr + r - tr, p=self.p, dim=1)
        return pred


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1, keepdims=True)


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """
    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat


def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def nearest_neighbor_sampling(pos, triples, ills, ids, k, params):
    t_ = time.time()
    emb = params['emb']
    metric = params['metric']
    if len(pos[0]) == 3:
        sorted_id = [sorted(ids[0]), sorted(ids[1])]
        distance = [-sim(emb[sorted_id[0]], emb[sorted_id[0]], metric=metric, normalize=False, csls_k=0), -sim(emb[sorted_id[1]], emb[sorted_id[1]], metric=metric, normalize=False, csls_k=0)]
        cache_dict = {}
        neg = []
        triples = set(triples)
        for _ in range(k):
            for h, r, t in pos:
                base_h = 0 if h in ids[0] else 1
                base_t = 0 if t in ids[0] else 1
                while True:
                    h2, r2, t2 = h, r, t
                    choice = np.random.binomial(1, 0.5)
                    if choice:
                        if h not in cache_dict:
                            indices = np.argsort(distance[base_h][sorted_id[base_h].index(h), :])
                            cache_dict[h] = np.array(sorted_id[base_h])[indices[1:]].tolist()
                        h2 = random.sample(cache_dict[h][:k], 1)[0]
                    else:
                        if t not in cache_dict:
                            indices = np.argsort(distance[base_t][sorted_id[base_t].index(t), :])
                            cache_dict[t] = np.array(sorted_id[base_t])[indices[1:]].tolist()
                        t2 = random.sample(cache_dict[t][:k], 1)[0]
                    if (h2, r2, t2) not in triples:
                        break
                neg.append((h2, r2, t2))
    elif len(pos[0]) == 2:
        neg_left = []
        distance = -sim(emb[pos[:, 0]], emb[pos[:, 0]], metric=metric, normalize=False, csls_k=0)
        for idx in range(len(pos)):
            indices = np.argsort(distance[idx, :])
            neg_left.append(pos[:, 0][indices[1:k + 1]])
        neg_left = np.stack(neg_left, axis=1).reshape(-1, 1)
        neg_right = []
        distance = -sim(emb[pos[:, 1]], emb[pos[:, 1]], metric=metric, normalize=False, csls_k=0)
        for idx in range(len(pos)):
            indices = np.argsort(distance[idx, :])
            neg_right.append(pos[:, 1][indices[1:k + 1]])
        neg_right = np.stack(neg_right, axis=1).reshape(-1, 1)
        neg_left = np.concatenate((neg_left, np.tile(pos, (k, 1))[:, 1].reshape(-1, 1)), axis=1).tolist()
        neg_right = np.concatenate((np.tile(pos, (k, 1))[:, 0].reshape(-1, 1), neg_right), axis=1).tolist()
        neg = neg_left + neg_right
        del distance
        gc.collect()
    else:
        raise NotImplementedError
    return neg


def random_sampling(pos, triples, ills, ids, k, params):
    t_ = time.time()
    if len(pos[0]) == 3:
        neg = []
        triples = set(triples)
        for _ in range(k):
            for h, r, t in pos:
                ent_set = ids[0] if h in ids[0] else ids[1]
                while True:
                    h2, r2, t2 = h, r, t
                    choice = np.random.binomial(1, 0.5)
                    if choice:
                        h2 = random.sample(ent_set, 1)[0]
                    else:
                        t2 = random.sample(ent_set, 1)[0]
                    if (h2, r2, t2) not in triples:
                        break
                neg.append((h2, r2, t2))
    elif len(pos[0]) == 2:
        neg_left, neg_right = [], []
        ills = set([(e1, e2) for e1, e2 in ills])
        for _ in range(k):
            for e1, e2 in pos:
                e11 = random.sample(ids[0] - {e1}, 1)[0]
                neg_left.append((e11, e2))
                e22 = random.sample(ids[1] - {e2}, 1)[0]
                neg_right.append((e1, e22))
        neg = neg_left + neg_right
    else:
        raise NotImplementedError
    return neg


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def multi_cal_neg(pos, task, triples, r_hs_dict, r_ts_dict, ids, k):
    neg = []
    for _ in range(k):
        neg_part = []
        for idx, tas in enumerate(task):
            h, r, t = pos[tas]
            temp_scope, num = True, 0
            while True:
                h2, r2, t2 = h, r, t
                choice = np.random.binomial(1, 0.5)
                if choice:
                    if temp_scope:
                        h2 = random.sample(r_hs_dict[r], 1)[0]
                    else:
                        for id in ids:
                            if h2 in id:
                                h2 = random.sample(id, 1)[0]
                elif temp_scope:
                    t2 = random.sample(r_ts_dict[r], 1)[0]
                else:
                    for id in ids:
                        if t2 in id:
                            t2 = random.sample(id, 1)[0]
                if (h2, r2, t2) not in triples:
                    break
                else:
                    num += 1
                    if num > 10:
                        temp_scope = False
            neg_part.append((h2, r2, t2))
        neg.append(neg_part)
    return neg


def typed_sampling(pos, triples, ills, ids, k, params):
    t_ = time.time()
    if len(pos[0]) == 2:
        raise NotImplementedError('typed_sampling is not supported in ills sampling')
    triples = set(triples)
    r_hs_dict, r_ts_dict = {}, {}
    for h, r, t in triples:
        if r not in r_hs_dict:
            r_hs_dict[r] = set()
        if r not in r_ts_dict:
            r_ts_dict[r] = set()
        r_hs_dict[r].add(h)
        r_ts_dict[r].add(t)
    tasks = div_list(np.array(range(len(pos)), dtype=np.int32), 1)
    neg_part = multi_cal_neg(pos, tasks[0], triples, r_hs_dict, r_ts_dict, ids, k)
    neg = []
    for part in neg_part:
        neg.extend(part)
    return neg


class Decoder(torch.nn.Module):

    def __init__(self, name, params):
        super(Decoder, self).__init__()
        self.print_name = name
        if name.startswith('[') and name.endswith(']'):
            self.name = name[1:-1]
        else:
            self.name = name
        p = 1 if params['train_dist'] == 'manhattan' else 2
        transe_sp = True if params['train_dist'] == 'normalize_manhattan' else False
        self.feat_drop = params['feat_drop']
        self.k = params['k']
        self.alpha = params['alpha']
        self.margin = params['margin']
        self.boot = params['boot']
        if self.name == 'align':
            self.func = Align(p)
        elif self.name == 'n_transe':
            self.func = N_TransE(p=p, params=self.margin)
        elif self.name == 'n_r_align':
            self.func = N_R_Align(params=self.margin)
        elif self.name == 'mtranse_align':
            self.func = MTransE_Align(p=p, dim=params['dim'], mode='sa4')
        elif self.name == 'alignea':
            self.func = AlignEA(p=p, feat_drop=self.feat_drop, params=self.margin)
        elif self.name == 'transedge':
            self.func = TransEdge(p=p, feat_drop=self.feat_drop, dim=params['dim'], mode='cp', params=self.margin)
        elif self.name == 'mmea':
            self.func = MMEA(feat_drop=self.feat_drop)
        elif self.name == 'transe':
            self.func = TransE(p=p, feat_drop=self.feat_drop, transe_sp=transe_sp)
        elif self.name == 'transh':
            self.func = TransH(p=p, feat_drop=self.feat_drop)
        elif self.name == 'transr':
            self.func = TransR(p=p, feat_drop=self.feat_drop)
        elif self.name == 'distmult':
            self.func = DistMult(feat_drop=self.feat_drop)
        elif self.name == 'complex':
            self.func = ComplEx(feat_drop=self.feat_drop)
        elif self.name == 'rotate':
            self.func = RotatE(p=p, feat_drop=self.feat_drop, dim=params['dim'], params=self.margin)
        elif self.name == 'hake':
            self.func = HAKE(p=p, feat_drop=self.feat_drop, dim=params['dim'], params=self.margin)
        elif self.name == 'conve':
            self.func = ConvE(feat_drop=self.feat_drop, dim=params['dim'], e_num=params['e_num'])
        else:
            raise NotImplementedError('bad decoder name: ' + self.name)
        if params['sampling'] == 'T':
            self.sampling_method = typed_sampling
        elif params['sampling'] == 'N':
            self.sampling_method = nearest_neighbor_sampling
        elif params['sampling'] == 'R':
            self.sampling_method = random_sampling
        elif params['sampling'] == '.':
            self.sampling_method = None
        else:
            raise NotImplementedError('bad sampling method: ' + self.sampling_method)
        if hasattr(self.func, 'loss'):
            self.loss = self.func.loss
        else:
            self.loss = nn.MarginRankingLoss(margin=self.margin)
        if hasattr(self.func, 'mapping'):
            self.mapping = self.func.mapping

    def forward(self, ins_emb, rel_emb, sample):
        if type(ins_emb) == tuple:
            ins_emb, weight = ins_emb
            rel_emb_ = torch.matmul(rel_emb, weight)
        else:
            rel_emb_ = rel_emb
        func = self.func if self.sampling_method else self.func.only_pos_loss
        if self.name in ['align', 'mtranse_align']:
            return func(ins_emb[sample[:, 0]], ins_emb[sample[:, 1]])
        elif self.name == 'n_r_align':
            nei_emb, ins_emb = ins_emb, rel_emb
            return func(ins_emb[sample[:, 0]], ins_emb[sample[:, 1]], nei_emb[sample[:, 0]], nei_emb[sample[:, 1]])
        else:
            return func(ins_emb[sample[:, 0]], rel_emb_[sample[:, 1]], ins_emb[sample[:, 2]])

    def __repr__(self):
        return '{}(name={}): {}'.format(self.__class__.__name__, self.print_name, self.func.__repr__())


class HighWay(torch.nn.Module):

    def __init__(self, f_in, f_out, bias=True):
        super(HighWay, self).__init__()
        self.w = Parameter(torch.Tensor(f_in, f_out))
        nn.init.xavier_uniform_(self.w)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

    def forward(self, in_1, in_2):
        t = torch.mm(in_1, self.w)
        if self.bias is not None:
            t = t + self.bias
        gate = torch.sigmoid(t)
        return gate * in_2 + (1.0 - gate) * in_1


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Align,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AlignEA,
     lambda: ([], {'p': 4, 'feat_drop': 4, 'params': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HighWay,
     lambda: ([], {'f_in': 4, 'f_out': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MTransE_Align,
     lambda: ([], {'p': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (N_R_Align,
     lambda: ([], {'params': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (N_TransE,
     lambda: ([], {'p': 4, 'params': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_THU_KEG_EAkit(_paritybench_base):
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

