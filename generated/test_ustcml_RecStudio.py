import sys
_module = sys.modules[__name__]
del sys
sasrec_demo = _module
recstudio = _module
ann = _module
sampler = _module
data = _module
advance_dataset = _module
dataset = _module
eval = _module
model = _module
multidae = _module
multivae = _module
basemodel = _module
baseranker = _module
baseretriever = _module
recommender = _module
debias = _module
fm = _module
dcn = _module
deepfm = _module
fm = _module
lr = _module
nfm = _module
widedeep = _module
xdeepfm = _module
graph = _module
lightgcn = _module
ncl = _module
ngcf = _module
sgl = _module
simgcl = _module
init = _module
kg = _module
loss_func = _module
mf = _module
bpr = _module
cml = _module
dssm = _module
ease = _module
irgan = _module
itemknn = _module
logisticmf = _module
ncf = _module
pmf = _module
slim = _module
wrmf = _module
module = _module
ctr = _module
data_augmentation = _module
functional = _module
graphmodule = _module
gru = _module
layers = _module
ranker = _module
retriever = _module
scorer = _module
seq = _module
bert4rec = _module
caser = _module
cl4srec = _module
coserec = _module
din = _module
fpmc = _module
gru4rec = _module
hgn = _module
iclrec = _module
narm = _module
npe = _module
sasrec = _module
stamp = _module
transrec = _module
quickstart = _module
config_dataset = _module
run = _module
utils = _module
callbacks = _module
compress_file = _module
trainer = _module
utils = _module
setup = _module
test_config_dataset = _module
test_dataset = _module
test_ddp = _module
test_quickrun = _module
test_retriever = _module

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


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


from typing import Dict


import numpy as np


from torch import Tensor


import torch.nn.functional as F


import copy


import logging


from typing import Sized


from typing import Iterator


import pandas as pd


import scipy.sparse as ssp


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import Sampler


from torch.utils.data.distributed import DistributedSampler


from collections import defaultdict


import inspect


import abc


import time


import torch.optim as optim


import torch.distributed as dist


from torch.utils.tensorboard import SummaryWriter


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.multiprocessing as mp


from torch.nn.utils.clip_grad import clip_grad_norm_


from collections import OrderedDict


import torch.nn as nn


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from typing import OrderedDict


from torch import optim


import scipy.sparse as sp


from sklearn.linear_model import ElasticNet


from sklearn.exceptions import ConvergenceWarning


import warnings


from typing import Set


import math


import random


from torch import nn


from torch.nn.parameter import Parameter


import re


from torch import TensorType


class SASRecQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, bidirectional=False, training_pooling_type='last', eval_pooling_type='last') ->None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=hidden_size, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=False)
        self.transformer_layer = torch.nn.TransformerEncoder(encoder_layer=transformer_encoder, num_layers=n_layer)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.training_pooling_layer = module.SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = module.SeqPoolingLayer(pooling_type=self.eval_pooling_type)

    def forward(self, batch, need_pooling=True):
        user_hist = batch['in_' + self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)
        mask4padding = user_hist == 0
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(src=self.dropout(seq_embs + position_embs), mask=attention_mask, src_key_padding_mask=mask4padding)
        if need_pooling:
            if self.training:
                if self.training_pooling_type == 'mask':
                    return self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.training_pooling_layer(transformer_out, batch['seqlen'])
            elif self.eval_pooling_type == 'mask':
                return self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
            else:
                return self.eval_pooling_layer(transformer_out, batch['seqlen'])
        else:
            return transformer_out


class Sampler(torch.nn.Module):

    def __init__(self, num_items, scorer_fn=None):
        super(Sampler, self).__init__()
        self.num_items = num_items - 1
        self.scorer = scorer_fn

    def update(self, item_embs, max_iter=30):
        pass

    def compute_item_p(self, query, pos_items):
        pass


class RetrieverSampler(Sampler):

    def __init__(self, num_items, retriever=None, method='brute', t=1):
        super().__init__(num_items)
        self.retriever = retriever
        self.method = method
        self.T = t

    def update(self, item_embs):
        self.retriever._update_item_vector()

    def forward(self, batch, num_neg: Union[int, List, Tuple], pos_items=None, excluding_hist=False):
        (log_pos_prob, neg_id, log_neg_prob), query_out = self.retriever.sampling(batch=batch, num_neg=num_neg, excluding_hist=excluding_hist, method=self.method, return_query=False, t=self.T)
        return log_pos_prob.detach(), neg_id.detach(), log_neg_prob.detach()


class UniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def forward(self, query: Union[Tensor, int], num_neg: int, pos_items: Optional[Tensor]=None, device: Optional[torch.device]=None):
        if isinstance(query, int):
            num_queries = query
            device = pos_items.device if pos_items is not None else device
            shape = num_queries,
        elif isinstance(query, Tensor):
            query = query
            num_queries = np.prod(query.shape[:-1])
            device = query.device
            shape = query.shape[:-1]
        with torch.no_grad():
            neg_items = torch.randint(1, self.num_items + 1, size=(num_queries, num_neg), device=device)
            neg_items = neg_items.reshape(*shape, -1)
            neg_prob = self.compute_item_p(None, neg_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(None, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return torch.zeros_like(pos_items)


def uniform_sample_masked_hist(num_items: int, num_neg: int, user_hist: Tensor, num_query_per_user: int=None):
    """Sampling from ``1`` to ``num_items`` uniformly with masking items in user history.

    Args:
        num_items(int): number of total items.
        num_neg(int): number of negative samples.
        user_hist(torch.Tensor): items list in user interacted history. The shape are required to be ``[num_user(or batch_size),max_hist_seq_len]`` with padding item(with index ``0``).
        num_query_per_user(int, optimal): number of queries of each user. It will be ``None`` when there is only one query for one user.

    Returns:
        torch.Tensor: ``[num_user(or batch_size),num_neg]`` or ``[num_user(or batch_size),num_query_per_user,num_neg]``, negative item index. If ``num_query_per_user`` is ``None``,  the shape will be ``[num_user(or batch_size),num_neg]``.
    """
    n_q = 1 if num_query_per_user is None else num_query_per_user
    num_user, hist_len = user_hist.shape
    device = user_hist.device
    neg_float = torch.rand(num_user, n_q * num_neg, device=device)
    non_zero_count = torch.count_nonzero(user_hist, dim=-1)
    neg_items = torch.floor(neg_float * (num_items - non_zero_count).view(-1, 1)).long() + 1
    sorted_hist, _ = user_hist.sort(dim=-1)
    offset = torch.arange(hist_len, device=device).repeat(num_user, 1)
    offset = offset - (hist_len - non_zero_count).view(-1, 1)
    offset[offset < 0] = 0
    sorted_hist = sorted_hist - offset
    masked_offset = torch.searchsorted(sorted_hist, neg_items, right=True)
    padding_nums = hist_len - non_zero_count
    neg_items += masked_offset - padding_nums.view(-1, 1)
    if num_query_per_user is not None:
        neg_items = neg_items.reshape(num_user, num_query_per_user, num_neg)
    return neg_items


class MaskedUniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def forward(self, query, num_neg, pos_items=None, user_hist=None):
        with torch.no_grad():
            if query.dim() == 2:
                neg_items = uniform_sample_masked_hist(num_query_per_user=None, num_items=self.num_items, num_neg=num_neg, user_hist=user_hist)
            elif query.dim() == 3:
                neg_items = uniform_sample_masked_hist(num_query_per_user=query.size(1), num_items=self.num_items, num_neg=num_neg, user_hist=user_hist)
            else:
                raise ValueError('`query` need to be 2-dimensional or 3-dimensional.')
            neg_prob = self.compute_item_p(query, neg_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(query, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return -torch.log(torch.ones_like(pos_items))


class DatasetUniformSampler(Sampler):

    def forward(self, num_neg=1, user_hist=None):
        for hist in user_hist:
            for i in range(num_neg):
                neg = torch.randint()


class PopularSamplerModel(Sampler):

    def __init__(self, pop_count, scorer=None, mode=0):
        super(PopularSamplerModel, self).__init__(pop_count.shape[0], scorer)
        with torch.no_grad():
            pop_count = torch.tensor(pop_count, dtype=torch.float)
            if mode == 0:
                pop_count = torch.log(pop_count + 1)
            elif mode == 1:
                pop_count = torch.log(pop_count + 1) + 1e-06
            elif mode == 2:
                pop_count = pop_count ** 0.75
            pop_count[0] = 1
            self.register_buffer('pop_prob', pop_count / pop_count.sum())
            self.register_buffer('table', torch.cumsum(self.pop_prob, dim=0))
            self.pop_prob[-1] = 1.0

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            seeds = torch.rand(num_queries, num_neg, device=query.device)
            neg_items = torch.searchsorted(self.table, seeds)
            neg_items = neg_items.reshape(*query.shape[:-1], -1)
            neg_prob = self.compute_item_p(query, neg_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(query, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return torch.log(self.pop_prob[pos_items])


def construct_index(cd01, K):
    cd01, indices = torch.sort(cd01, stable=True)
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr


def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int):
        K = K_or_center
        C = X[torch.randperm(N)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = X.new_zeros(N, K)
        assign_m[range(N), assign] = 1
        loss = torch.sum(torch.square(X - C[assign, :])).item()
        if verbose:
            None
        if prev_loss - loss < prev_loss * 1e-06:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C = assign_m.T @ X / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count < 0.5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N)[:ndead]]
    return C, assign, assign_m, loss


class MIDXSamplerUniform(Sampler):
    """
    Uniform sampling for the final items
    """

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, scorer.MLPScorer)
        super(MIDXSamplerUniform, self).__init__(num_items, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, scorer.CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(embs1, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(embs2, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        self.c0_ = torch.cat([self.c0.new_zeros(1, self.c0.size(1)), self.c0], dim=0)
        self.c1_ = torch.cat([self.c1.new_zeros(1, self.c1.size(1)), self.c1], dim=0)
        self.cd0 = torch.cat([-cd0.new_ones(1), cd0], dim=0) + 1
        self.cd1 = torch.cat([-cd1.new_ones(1), cd1], dim=0) + 1
        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K ** 2)
        self._update(item_embs, cd0m, cd1m)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            self.wkk = cd0m.T @ cd1m
        else:
            norm = torch.exp(-0.5 * torch.sum(item_embs ** 2, dim=-1))
            self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K ** 2):
                start, end = self.indptr[c], self.indptr[c + 1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum(0)
                    self.cp[start:end] = cumsum / cumsum[-1]

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            if isinstance(self.scorer, scorer.CosineScorer):
                query = F.normalize(query, dim=-1)
            q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
            r1 = q1 @ self.c1.T
            r1s = torch.softmax(r1, dim=-1)
            r0 = q0 @ self.c0.T
            r0s = torch.softmax(r0, dim=-1)
            s0 = r1s @ self.wkk.T * r0s
            k0 = torch.multinomial(s0, num_neg, replacement=True)
            p0 = torch.gather(r0, -1, k0)
            subwkk = self.wkk[k0, :]
            s1 = subwkk * r1s.unsqueeze(1)
            k1 = torch.multinomial(s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1])
            p1 = torch.gather(r1, -1, k1)
            k01 = k0 * self.K + k1
            p01 = p0 + p1
            neg_items, neg_prob = self.sample_item(k01, p01)
            if pos_items is not None:
                pos_prob = None if pos_items is None else self.compute_item_p(query, pos_items)
                return pos_prob, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
            else:
                return neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01, pos=None):
        if not hasattr(self, 'cp'):
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(item_cnt * torch.rand_like(item_cnt.float())).int()
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)

    def _sample_item_with_pop(self, k01, p01):
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(maxlen, device=start.device).reshape(1, 1, maxlen)
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(start.float()).unsqueeze(-1)).squeeze(-1)
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        neg_probs = self.p[item_idx + self.indptr[k01] + 1]
        return neg_items, p01 + torch.log(neg_probs)

    def compute_item_p(self, query, pos_items):
        if pos_items.dim() == 1:
            pos_items_ = pos_items.unsqueeze(1)
        else:
            pos_items_ = pos_items
        k0 = self.cd0[pos_items_]
        k1 = self.cd1[pos_items_]
        c0 = self.c0_[k0, :]
        c1 = self.c1_[k1, :]
        q0, q1 = query.chunk(2, dim=-1)
        if query.dim() == pos_items_.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) + torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1)
        else:
            r = torch.bmm(q0, c0.transpose(1, 2)) + torch.bmm(q1, c1.transpose(1, 2))
            pos_items_ = pos_items_.unsqueeze(1)
        if not hasattr(self, 'p'):
            return r.view_as(pos_items)
        else:
            return (r + torch.log(self.p[pos_items_])).view_as(pos_items)


class MIDXSamplerPop(MIDXSamplerUniform):
    """
    Popularity sampling for the final items
    """

    def __init__(self, pop_count: torch.Tensor, num_clusters, scorer=None, mode=1):
        super(MIDXSamplerPop, self).__init__(pop_count.shape[0], num_clusters, scorer)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-06
        elif mode == 2:
            pop_count = pop_count ** 0.75
        self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            norm = self.pop_count
        else:
            norm = self.pop_count * torch.exp(-0.5 * torch.sum(item_embs ** 2, dim=-1))
        self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K ** 2):
            start, end = self.indptr[c], self.indptr[c + 1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]


class ClusterSamplerUniform(MIDXSamplerUniform):

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or not isinstance(scorer_fn, scorer.MLPScorer)
        super(ClusterSamplerUniform, self).__init__(num_items, num_clusters, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=30):
        if isinstance(self.scorer, scorer.CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        self.c, cd, cdm, _ = kmeans(item_embs, self.K, max_iter)
        self.c_ = torch.cat([self.c.new_zeros(1, self.c.size(1)), self.c], dim=0)
        self.cd = torch.cat([-cd.new_ones(1), cd], dim=0) + 1
        self.indices, self.indptr = construct_index(cd, self.K)
        self._update(item_embs, cdm)

    def _update(self, item_embs, cdm):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            self.wkk = cdm.sum(0)
        else:
            norm = torch.exp(-0.5 * torch.sum(item_embs ** 2, dim=-1))
            self.wkk = (cdm * norm.view(-1, 1)).sum(0)
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K):
                start, end = self.indptr[c], self.indptr[c + 1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum()
                    self.cp[start:end] = cumsum / cumsum[-1]

    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            if isinstance(self.scorer, scorer.CosineScorer):
                query = F.normalize(query, dim=-1)
            q = query.view(-1, query.size(-1))
            r = q @ self.c.T
            rs = torch.softmax(r, dim=-1)
            k = torch.multinomial(rs, num_neg, replacement=True)
            p = torch.gather(r, -1, k)
            neg_items, neg_prob = self.sample_item(k, p, pos_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(query, pos_items)
                return pos_prob, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
            else:
                return neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def compute_item_p(self, query, pos_items):
        shape = pos_items.shape
        if pos_items.dim() == 1:
            pos_items = pos_items.view(-1, 1)
        k = self.cd[pos_items]
        c = self.c_[k, :]
        if query.dim() == pos_items.dim():
            r = torch.bmm(c, query.unsqueeze(-1)).squeeze(-1)
        else:
            r = torch.bmm(query, c.transpose(1, 2))
            pos_items = pos_items.unsqueeze(1)
        r = r.reshape(*shape)
        if not hasattr(self, 'p'):
            return r
        else:
            return r + torch.log(self.p[pos_items])

    def sample_item(self, k01, p01, pos=None):
        if not hasattr(self, 'cp'):
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(item_cnt * torch.rand_like(item_cnt.float())).int()
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)

    def _sample_item_with_pop(self, k01, p01):
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(maxlen, device=start.device).reshape(1, 1, maxlen)
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(start.float()).unsqueeze(-1)).squeeze(-1)
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        neg_probs = self.p[item_idx + self.indptr[k01] + 1]
        return neg_items, p01 + torch.log(neg_probs)


class ClusterSamplerPop(ClusterSamplerUniform):

    def __init__(self, pop_count: torch.Tensor, num_clusters, scorer=None, mode=1):
        super(ClusterSamplerPop, self).__init__(pop_count.shape[0], num_clusters, scorer)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-06
        elif mode == 2:
            pop_count = pop_count ** 0.75
        self.pop_count = torch.nn.Parameter(pop_count, requires_grad=False)

    def _update(self, item_embs, cdm):
        if not isinstance(self.scorer, scorer.EuclideanScorer):
            norm = self.pop_count
        else:
            norm = self.pop_count * torch.exp(-0.5 * torch.sum(item_embs ** 2, dim=-1))
        self.wkk = (cdm * norm.view(-1, 1)).sum(0)
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K):
            start, end = self.indptr[c], self.indptr[c + 1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]


class DataSampler(Sampler):
    """Data sampler to return index for batch data.

    The datasampler generate batches of index in the `data_source`, which can be used in dataloader to sample data.

    Args:
        data_source(Sized): the dataset, which is required to have length.

        batch_size(int): batch size for each mini batch.

        shuffle(bool, optional): whether to shuffle the dataset each epoch. (default: `True`)

        drop_last(bool, optional): whether to drop the last mini batch when the size is smaller than the `batch_size`.(default: `False`)

        generator(optinal): generator to generate rand numbers. (default: `None`)
    """

    def __init__(self, data_source: Sized, batch_size, shuffle=True, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.shuffle:
            output = torch.randperm(n, generator=generator).split(self.batch_size)
        else:
            output = torch.arange(n).split(self.batch_size)
        if self.drop_last and len(output[-1]) < self.batch_size:
            yield from output[:-1]
        else:
            yield from output

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class SortedDataSampler(Sampler):
    """Data sampler to return index for batch data, aiming to collect data with similar lengths into one batch.

    In order to save memory in training producure, the data sampler collect data point with similar length into one batch.

    For example, in sequential recommendation, the interacted item sequence of different users may vary differently, which may cause
    a lot of padding. By considering the length of each sequence, gathering those sequence with similar lengths in the same batch can
    tackle the problem.

    If `shuffle` is `True`, length of sequence and the random index are combined together to reduce padding without randomness.

    Args:
        data_source(Sized): the dataset, which is required to have length.

        batch_size(int): batch size for each mini batch.

        shuffle(bool, optional): whether to shuffle the dataset each epoch. (default: `True`)

        drop_last(bool, optional): whether to drop the last mini batch when the size is smaller than the `batch_size`.(default: `False`)

        generator(optinal): generator to generate rand numbers. (default: `None`)
    """

    def __init__(self, data_source: Sized, batch_size, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            output = torch.div(torch.randperm(n), self.batch_size * 10, rounding_mode='floor')
            output = self.data_source.sample_length + output * (self.data_source.sample_length.max() + 1)
        else:
            output = self.data_source.sample_length
        output = torch.sort(output).indices
        output = output.split(self.batch_size)
        if self.drop_last and len(output[-1]) < self.batch_size:
            yield from output[:-1]
        else:
            yield from output

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class Dice(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_features: int

    def __init__(self, num_parameters, init: float=0.25, epsilon: float=1e-08):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = torch.nn.parameter.Parameter(torch.empty(num_parameters).fill_(init))
        self.epsilon = epsilon

    def forward(self, x):
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        var_x = torch.var(x, dim=-1, keepdim=True)
        x_std = (x - mean_x) / torch.sqrt(var_x + self.epsilon)
        p_x = torch.sigmoid(x_std)
        f_x = p_x * x + (1 - p_x) * x * self.weight.expand_as(x)
        return f_x

    def extra_repr(self) ->str:
        return 'num_parameters={}'.format(self.num_parameters)


def get_act(activation: str, dim=None):
    if activation == None or isinstance(activation, torch.nn.Module):
        return activation
    elif type(activation) == str:
        if activation.lower() == 'relu':
            return torch.nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return torch.nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        elif activation.lower() == 'identity':
            return lambda x: x
        elif activation.lower() == 'dice':
            return Dice(dim)
        elif activation.lower() == 'gelu':
            return torch.nn.GELU()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(f'activation function type "{activation}"  is not supported, check spelling or pass in a instance of torch.nn.Module.')
    else:
        raise ValueError('"activation_func" must be a str or a instance of torch.nn.Module. ')


class MLPModule(torch.nn.Module):
    """
    MLPModule
    Gets a MLP easily and quickly.

    Args:
        mlp_layers(list): the dimensions of every layer in the MLP.
        activation_func(torch.nn.Module,str,None): the activation function in each layer.
        dropout(float): the probability to be set in dropout module. Default: ``0.0``.
        bias(bool): whether to add batch normalization between layers. Default: ``False``.
        last_activation(bool): whether to add activation in the last layer. Default: ``True``.
        last_bn(bool): whether to add batch normalization in the last layer. Default: ``True``.

    Examples:
    >>> MLP = MLPModule([64, 64, 64], 'ReLU', 0.2)
    >>> MLP.model
    Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=64, out_features=64, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=64, out_features=64, bias=True)
        (5): ReLU()
    )
    >>> MLP.add_modules(torch.nn.Linear(64, 10, True), torch.nn.ReLU())
    >>> MLP.model
    Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=64, out_features=64, bias=True)
        (2): ReLU()
        (3): Dropout(p=0.2, inplace=False)
        (4): Linear(in_features=64, out_features=64, bias=True)
        (5): ReLU()
        (6): Linear(in_features=64, out_features=10, bias=True)
        (7): ReLU()
    )
    """

    def __init__(self, mlp_layers, activation_func='ReLU', dropout=0.0, bias=True, batch_norm=False, last_activation=True, last_bn=True):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.bias = bias
        self.dropout = dropout
        self.activation_func = activation_func
        self.model = []
        last_bn = self.batch_norm and last_bn
        for idx, layer in enumerate(zip(self.mlp_layers[:-1], self.mlp_layers[1:])):
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.Linear(*layer, bias=bias))
            if idx == len(mlp_layers) - 2 and last_bn or idx < len(mlp_layers) - 2 and batch_norm:
                self.model.append(torch.nn.BatchNorm1d(layer[-1]))
            if idx == len(mlp_layers) - 2 and last_activation and activation_func is not None or idx < len(mlp_layers) - 2 and activation_func is not None:
                activation = get_act(activation_func, dim=layer[-1])
                self.model.append(activation)
        self.model = torch.nn.Sequential(*self.model)

    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model.
        """
        for block in args:
            assert isinstance(block, torch.nn.Module)
        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, input):
        return self.model(input)


class MultiDAEQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, num_items, embed_dim, dropout_rate, encoder_dims, decoder_dims, activation='relu'):
        super().__init__()
        assert encoder_dims[-1] == decoder_dims[0], 'expecting the output size ofencoder is equal to the input size of decoder.'
        assert encoder_dims[0] == decoder_dims[-1], 'expecting the output size ofdecoder is equal to the input size of encoder.'
        self.fiid = fiid
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.encoder_decoder = torch.nn.Sequential(MLPModule([embed_dim] + encoder_dims + decoder_dims[1:], activation), torch.nn.Linear(decoder_dims[-1], embed_dim))

    def forward(self, batch):
        seq_emb = self.item_embedding(batch['in_' + self.fiid])
        non_zero_num = batch['in_' + self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)
        h = self.dropout(seq_emb)
        return self.encoder_decoder(h)


class MultiVAEQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, num_items, embed_dim, dropout_rate, encoder_dims, decoder_dims, activation='relu'):
        super().__init__()
        assert encoder_dims[-1] == decoder_dims[0], 'expecting the output size ofencoder is equal to the input size of decoder.'
        assert encoder_dims[0] == decoder_dims[-1], 'expecting the output size ofdecoder is equal to the input size of encoder.'
        self.fiid = fiid
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.encoders = torch.nn.Sequential(MLPModule([embed_dim] + encoder_dims[:-1], activation), torch.nn.Linear(([embed_dim] + encoder_dims[:-1])[-1], encoder_dims[-1] * 2))
        self.decoders = torch.nn.Sequential(MLPModule(decoder_dims, activation), torch.nn.Linear(decoder_dims[-1], embed_dim))
        self.kl_loss = 0.0

    def forward(self, batch):
        seq_emb = self.item_embedding(batch['in_' + self.fiid])
        non_zero_num = batch['in_' + self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)
        h = self.dropout(seq_emb)
        encoder_h = self.encoders(h)
        mu, logvar = encoder_h.tensor_split(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        decoder_z = self.decoders(z)
        if self.training:
            self.kl_loss = self.kl_loss_func(mu, logvar)
        return decoder_z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def kl_loss_func(self, mu, logvar):
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return KLD


class CombinedLoaders(object):

    def __init__(self, loaders) ->None:
        """
        The first loader is the main loader.
        """
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for i, l in enumerate(self.loaders):
            self.loaders[i] = iter(l)
        return self

    def __next__(self):
        batch = next(self.loaders[0])
        for i, l in enumerate(self.loaders[1:]):
            try:
                batch.update(next(l))
            except StopIteration:
                self.loaders[i + 1] = iter(self.loaders[i + 1])
                batch.update(next(self.loaders[i + 1]))
        return batch


DEFAULT_CACHE_DIR = './.recstudio/'


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) ->int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(self, sampler, num_replicas: Optional[int]=None, rank: Optional[int]=None, shuffle: bool=True):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.sampler = sampler

    def __iter__(self) ->Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class TensorFrame(Dataset):
    """The main data structure used to save interaction data in RecStudio dataset.

    TensorFrame class can be regarded as one enhanced dict, which contains several fields of data (like: ``user_id``, ``item_id``, ``rating`` and so on).
    And TensorFrame have some useful strengths:

    - Generated from pandas.DataFrame directly.

    - Easy to get/add/remove fields.

    - Easy to get each interaction information.

    - Compatible for torch.utils.data.DataLoader, which provides a loader method to return batch data.
    """

    @classmethod
    def fromPandasDF(cls, dataframe, dataset):
        """Get a TensorFrame from a pandas.DataFrame.

        Args:
            dataframe(pandas.DataFrame): Dataframe read from csv file.
            dataset(recstudio.data.MFDataset): target dataset where the TensorFrame is used.

        Return:
            recstudio.data.TensorFrame: the TensorFrame get from the dataframe.
        """
        data = {}
        fields = []
        length = len(dataframe.index)
        for field in dataframe:
            fields.append(field)
            ftype = dataset.field2type[field]
            value = dataframe[field]
            if ftype == 'token_seq':
                seq_data = [torch.from_numpy(d[:dataset.field2maxlen[field]]) for d in value]
                data[field] = pad_sequence(seq_data, batch_first=True)
            elif ftype == 'float_seq':
                seq_data = [torch.from_numpy(d[:dataset.field2maxlen[field]]) for d in value]
                data[field] = pad_sequence(seq_data, batch_first=True)
            elif ftype == 'token':
                data[field] = torch.from_numpy(dataframe[field].to_numpy(np.int64))
            else:
                data[field] = torch.from_numpy(dataframe[field].to_numpy(np.float32))
        return cls(data, length, fields)

    def __init__(self, data, length, fields):
        self.data = data
        self.length = length
        self.fields = fields

    def get_col(self, field):
        """Get data from the specific field.

        Args:
            field(str): field name.

        Returns:
            torch.Tensor: data of corresponding filed.
        """
        return self.data[field]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ret = {}
        for field, value in self.data.items():
            ret[field] = value[idx]
        return ret

    def del_fields(self, keep_fields):
        """Delete fields that are *not in* ``keep_fields``.

        Args:
            keep_fields(list[str],set[str] or dict[str]): the fields need to remain.
        """
        fields = copy.deepcopy(self.fields)
        for f in fields:
            if f not in keep_fields:
                self.fields.remove(f)
                del self.data[f]

    def loader(self, batch_size, shuffle=False, num_workers=1, drop_last=False):
        """Create dataloader.

        Args:
            batch_size(int): batch size for mini batch.

            shuffle(bool, optional): whether to shuffle the whole data. (default `False`).

            num_workers(int, optional): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process. (default: `1`).

            drop_last(bool, optinal): whether to drop the last mini batch when the size is smaller than the `batch_size`.

        Returns:
            torch.utils.data.DataLoader: the dataloader used to load all the data in the TensorFrame.
        """
        sampler = DataSampler(self, batch_size, shuffle, drop_last)
        output = DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=False)
        return output

    def add_field(self, field, value):
        """Add field to the TensorFrame.

        Args:
            field(str): the field name to be added.

            value(torch.Tensor): the value of the field.
        """
        self.data[field] = value

    def reindex(self, idx):
        """Shuffle the data according to the given `idx`.

        Args:
            idx(numpy.ndarray): the given data index.

        Returns:
            recstudio.data.TensorFrame: a copy of the TensorFrame after reindexing.
        """
        output = copy.deepcopy(self)
        for f in output.fields:
            output.data[f] = output.data[f][idx]
        return output


class CompressedFile(object):
    magic = None
    file_type = None
    mime_type = None

    def __init__(self, fname, save_dir):
        self.extract_all(fname, save_dir)

    @classmethod
    def is_magic(self, data):
        return data.startswith(self.magic)

    def extract_all(self, fname, save_dir):
        pass


class GZFile(CompressedFile):
    magic = b'\x1f\x8b\x08'
    file_type = 'gz'
    mime_type = 'compressed/gz'

    def extract_all(self, fname, save_dir):
        decompressed_fname = os.path.basename(fname)[:-3]
        with gzip.open(fname, 'rb') as f_in:
            with open(os.path.join(save_dir, decompressed_fname), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


class ZIPFile(CompressedFile):
    magic = b'PK\x03\x04'
    file_type = 'zip'
    mime_type = 'compressed/zip'

    def extract_all(self, fname, save_dir):
        with zipfile.ZipFile(fname) as f:
            for member in f.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = f.open(member)
                target = open(os.path.join(save_dir, filename), 'wb')
                with source, target:
                    shutil.copyfileobj(source, target)


def extract_compressed_file(filename, save_dir):
    with open(filename, 'rb') as f:
        start_of_file = f.read(1024)
        f.seek(0)
        if filename.endswith('csv'):
            pass
        else:
            for cls in (ZIPFile, GZFile):
                if cls.is_magic(start_of_file):
                    cls(filename, save_dir)
                    break
            os.remove(filename)


def get_download_url_from_recstore(share_number: str):
    headers = {'Host': 'recapi.ustc.edu.cn', 'Content-Type': 'application/json'}
    data_resource_list = {'share_number': share_number, 'share_resource_number': None, 'is_rec': 'false', 'share_constraint': {}}
    resource = requests.post('https://recapi.ustc.edu.cn/api/v2/share/target/resource/list', json=data_resource_list, headers=headers)
    resource = resource.text.encode('utf-8').decode('utf-8-sig')
    resource = json.loads(resource)
    resource = resource['entity'][0]['number']
    data = {'share_number': share_number, 'share_constraint': {}, 'share_resources_list': [resource]}
    res = requests.post('https://recapi.ustc.edu.cn/api/v2/share/download', json=data, headers=headers)
    res = res.text.encode('utf-8').decode('utf-8-sig')
    res = json.loads(res)
    download_url = res['entity'][resource] + '&download=download'
    return download_url


def download_dataset(url: str, name: str, save_dir: str):
    if url.startswith('http'):
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if 'rec.ustc.edu.cn' in url:
                url = get_download_url_from_recstore(share_number=url.split('/')[-1])
                zipped_file_name = f'{name}.zip'
            else:
                zipped_file_name = url.split('/')[-1]
            dataset_file_path = os.path.join(save_dir, zipped_file_name)
            response = requests.get(url, stream=True)
            content_length = int(response.headers.get('content-length', 0))
            with open(dataset_file_path, 'wb') as file, tqdm(desc='Downloading dataset', total=content_length, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            extract_compressed_file(dataset_file_path, save_dir)
            return save_dir
        except:
            None


def md5(config: dict):
    s = ''
    for k in sorted(config):
        s += f'{k}:{config[k]}\n'
    md = hashlib.md5(s.encode('utf8')).hexdigest()
    return md


def check_valid_dataset(name: str, config: Dict, default_dataset_path=DEFAULT_CACHE_DIR):
    """ Check existed dataset according to the md5 string.

    Args:
        name(str): the name of the dataset
        config(Dict): the config of the dataset
        default_data_set_path:(str, optional): the path of the local cache folder.

    Returns:
        str: download url of the dataset file or the local file path.
    """
    logger = logging.getLogger('recstudio')

    def get_files(vs):
        res = []
        for v in vs:
            if not isinstance(v, list):
                res.append(v)
            else:
                res = res + get_files(v)
        return res
    if not os.path.exists(default_dataset_path):
        os.makedirs(default_dataset_path)
    config_md5 = md5(config)
    cache_file_name = os.path.join(default_dataset_path, 'cache', config_md5)
    if os.path.exists(cache_file_name):
        return True, cache_file_name
    else:
        download_flag = False
        default_dir = os.path.join(default_dataset_path, name)
        for k, v in config.items():
            if k.endswith('feat_name'):
                if v is not None:
                    v = [v] if not isinstance(v, List) else v
                    files = get_files(v)
                    for f in files:
                        fpath = os.path.join(default_dir, f)
                        if not os.path.exists(fpath):
                            download_flag = True
                            break
            if download_flag == True:
                break
        if not download_flag:
            logger.info(f'dataset is read from {default_dir}.')
            return False, default_dir
        elif download_flag and config['url'] is not None:
            if config['url'].startswith('http'):
                logger.info(f"will download dataset {name} fron the url {config['url']}.")
                return False, download_dataset(config['url'], name, default_dir)
            elif config['url'].startswith('recstudio:'):
                dir = os.path.dirname(os.path.dirname(__file__))
                dir = os.path.join(dir, config['url'].split(':')[1])
                logger.info(f'dataset is read from {dir}.')
                return False, dir
            else:
                logger.info(f"dataset is read from {config['url']}.")
                return False, config['url']
        elif download_flag and config['url'] is None:
            raise FileNotFoundError('Sorry, the original dataset file can not be found due tothere is neither url provided or local file path provided in configuration fileswith the key `url`.')


def parser_yaml(config_path):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(u'tag:yaml.org,2002:float', re.compile(u"""^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""", re.X), list(u'-+0123456789.'))
    with open(config_path, 'r', encoding='utf-8') as f:
        ret = yaml.load(f.read(), Loader=loader)
    return ret


def get_dataset_default_config(dataset_name: str) ->Dict:
    logger = logging.getLogger('recstudio')
    dir = os.path.dirname(__file__)
    dataset_config_dir = os.path.join(dir, '../data/config')
    dataset_config_fname = os.path.join(dataset_config_dir, f'{dataset_name}.yaml')
    if os.path.exists(dataset_config_fname):
        config = parser_yaml(dataset_config_fname)
    else:
        logger.warning(f"There is no default configuration file for dataset {dataset_name}.Please make sure that all the configurations are setted in your provided file or theconfiguration dict you've assigned.")
        config = {}
    return config


def map(pred, target, k):
    """Calculate the mean Average Precision(mAP).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    count = (target > 0).sum(-1)
    pred = pred[:, :k].float()
    output = pred.cumsum(dim=-1) / torch.arange(1, k + 1).type_as(pred)
    output = (output * pred).sum(dim=-1) / torch.minimum(count, k * torch.ones_like(count))
    return output.mean()


def set_color(log, color, highlight=True, keep=False):
    """Set color for log string.

    Args:
        log(str): the
    """
    if keep:
        return log
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\x1b['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\x1b[0m'


def uniform_sampling(num_queries: int, num_items: int, num_neg: int, user_hist: Tensor=None, device='cpu', backend='multinomial'):
    if user_hist is None:
        neg_idx = torch.randint(1, num_items, size=(num_queries, num_neg), device=device)
        return neg_idx
    else:
        device = user_hist.device
        if backend == 'multinomial':
            weight = torch.ones(size=(num_queries, num_items), device=device)
            _idx = torch.arange(user_hist.size(0), device=device).view(-1, 1).expand_as(user_hist)
            weight[_idx, user_hist] = 0.0
            neg_idx = torch.multinomial(weight, num_neg, replacement=True)
        elif backend == 'numpy':
            user_hist_np = user_hist.cpu().numpy()
            neg_idx_np = np.zeros(shape=num_queries * num_neg)
            isin_id = np.arange(num_queries * num_neg)
            while len(isin_id) > 0:
                neg_idx[isin_id] = np.random.randint(1, num_items, len(isin_id))
                isin_id = torch.tensor([id for id in isin_id if neg_idx[id] in user_hist_np[id // num_neg]])
            neg_idx = torch.tensor(neg_idx_np, dtype=torch.long, device=device)
        elif backend == 'torch':
            neg_idx = user_hist.new_zeros(size=num_queries * num_neg)
            isin_id = torch.arange(neg_idx.size(0), device=device)
            while len(isin_id) > 0:
                neg_idx[isin_id] = torch.randint(1, num_items, size=len(isin_id), device=device)
                isin_id = torch.tensor([id for id in isin_id if neg_idx[id] in user_hist[id // num_neg]], device=device)
        return neg_idx


class MFDataset(Dataset):
    """ Dataset for Matrix Factorized Methods.

    The basic dataset class in RecStudio.
    """

    def __init__(self, name: str='ml-100k', config: Union[Dict, str]=None):
        """Load all data.

        Args:
            config(str): config file path or config dict for the dataset.

        Returns:
            recstudio.data.dataset.MFDataset: The ingredients list.
        """
        self.name = name
        self.logger = logging.getLogger('recstudio')
        self.config = get_dataset_default_config(name)
        if config is not None:
            if isinstance(config, str):
                self.config.update(parser_yaml(config))
            elif isinstance(config, Dict):
                self.config.update(config)
            else:
                raise TypeError(f'expecting `config` to be Dict or string,while get {type(config)} instead.')
        cache_flag, data_dir = check_valid_dataset(self.name, self.config)
        if cache_flag:
            self.logger.info('Load dataset from cache.')
            self._load_cache(data_dir)
        else:
            self._init_common_field()
            self._load_all_data(data_dir, self.config['field_separator'])
            self._filter(self.config['min_user_inter'], self.config['min_item_inter'])
            self._map_all_ids()
            self._post_preprocess()
            if self.config['save_cache']:
                self._save_cache(md5(self.config))
        self._use_field = set([self.fuid, self.fiid, self.frating])

    @property
    def field(self):
        return set(self.field2type.keys())

    @property
    def use_field(self):
        return self._use_field

    @use_field.setter
    def use_field(self, fields):
        self._use_field = set(fields)

    @property
    def drop_dup(self):
        if self.split_mode == 'entry':
            return False
        else:
            return True

    def _load_cache(self, path):
        with open(path, 'rb') as f:
            download_obj = pickle.load(f)
        for k in download_obj.__dict__:
            attr = getattr(download_obj, k)
            setattr(self, k, attr)

    def _save_cache(self, md: str):
        cache_dir = os.path.join(DEFAULT_CACHE_DIR, 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, md), 'wb') as f:
            pickle.dump(self, f)

    def _init_common_field(self):
        """Inits several attributes.
        """
        self.field2type = {}
        self.field2token2idx = {}
        self.field2tokens = {}
        self.field2maxlen = self.config['field_max_len'] or {}
        self.fuid = self.config['user_id_field'].split(':')[0]
        self.fiid = self.config['item_id_field'].split(':')[0]
        self.ftime = self.config['time_field'].split(':')[0]
        if self.config['rating_field'] is not None:
            self.frating = self.config['rating_field'].split(':')[0]
        else:
            self.frating = None

    def __test__(self):
        feat = self.network_feat[1][-10:]
        None
        self._map_all_ids()
        feat1 = self._recover_unmapped_feature(self.network_feat[1])
        None
        self._prepare_user_item_feat()
        feat2 = self._recover_unmapped_feature(self.network_feat[1])[-10:]
        None

    def __repr__(self):
        info = {'item': {}, 'user': {}, 'interaction': {}}
        feat = {'item': self.item_feat, 'user': self.user_feat, 'interaction': self.inter_feat}
        max_num_fields = 0
        max_len_field = max([len(f) for f in self.field] + [len('token_seq')]) + 1
        for k in info:
            info[k]['field'] = list(feat[k].fields)
            info[k]['type'] = [self.field2type[f] for f in info[k]['field']]
            info[k]['##'] = [(str(self.num_values(f)) if 'token' in t else '-') for f, t in zip(info[k]['field'], info[k]['type'])]
            max_num_fields = max(max_num_fields, len(info[k]['field'])) + 1
        info_str = f"\n{set_color('Dataset Info', 'green')}: \n"
        info_str += '\n' + '=' * (max_len_field * max_num_fields) + '\n'
        for k in info:
            info_str += set_color(k + ' information: \n', 'blue')
            for k, v in info[k].items():
                info_str += '{}'.format(set_color(k, 'yellow')) + ' ' * (max_len_field - len(k))
                info_str += ''.join([('{}'.format(i) + ' ' * (max_len_field - len(i))) for i in v])
                info_str += '\n'
            info_str += '=' * (max_len_field * max_num_fields) + '\n'
        info_str += '{}: {}\n'.format(set_color('Total Interactions', 'blue'), self.num_inters)
        info_str += '{}: {:.6f}\n'.format(set_color('Sparsity', 'blue'), 1 - self.num_inters / ((self.num_items - 1) * (self.num_users - 1)))
        info_str += '=' * (max_len_field * max_num_fields)
        return info_str

    def _filter_ratings(self):
        """Filter out the interactions whose rating is below `rating_threshold` in config."""
        if self.config['rating_threshold'] is not None:
            if not self.config['drop_low_rating']:
                self.inter_feat[self.frating] = (self.inter_feat[self.frating] >= self.config['rating_threshold']).astype(float)
            else:
                self.inter_feat = self.inter_feat[self.inter_feat[self.frating] >= self.config['rating_threshold']]
                self.inter_feat[self.frating] = 1.0

    def _load_all_data(self, data_dir, field_sep):
        """Load features for user, item, interaction and network."""
        inter_feat_path = os.path.join(data_dir, self.config['inter_feat_name'])
        self.inter_feat = self._load_feat(inter_feat_path, self.config['inter_feat_header'], field_sep, self.config['inter_feat_field'])
        self.inter_feat = self.inter_feat.dropna(how='any')
        if self.frating is None:
            self.frating = 'rating'
            self.inter_feat.insert(0, self.frating, 1)
            self.field2type[self.frating] = 'float'
            self.field2maxlen[self.frating] = 1
        self.user_feat = None
        if self.config['user_feat_name'] is not None:
            user_feat = []
            for _, user_feat_col in zip(self.config['user_feat_name'], self.config['user_feat_field']):
                user_feat_path = os.path.join(data_dir, _)
                user_f = self._load_feat(user_feat_path, self.config['user_feat_header'], field_sep, user_feat_col)
                user_f.set_index(self.fuid, inplace=True)
                user_feat.append(user_f)
            self.user_feat = pd.concat(user_feat, axis=1)
            self.user_feat.reset_index(inplace=True)
            self._fill_nan(self.user_feat)
        self.item_feat = None
        if self.config['item_feat_name'] is not None:
            item_feat = []
            for _, item_feat_col in zip(self.config['item_feat_name'], self.config['item_feat_field']):
                item_feat_path = os.path.join(data_dir, _)
                item_f = self._load_feat(item_feat_path, self.config['item_feat_header'], field_sep, item_feat_col)
                item_f.set_index(self.fiid, inplace=True)
                item_feat.append(item_f)
            self.item_feat = pd.concat(item_feat, axis=1)
            self.item_feat.reset_index(inplace=True)
            self._fill_nan(self.item_feat)
        if self.config['network_feat_name'] is not None:
            self.network_feat = [None] * len(self.config['network_feat_name'])
            self.node_link = [None] * len(self.config['network_feat_name'])
            self.node_relink = [None] * len(self.config['network_feat_name'])
            self.mapped_fields = [[(field.split(':')[0] if field != None else field) for field in fields] for fields in self.config['mapped_feat_field']]
            for i, (name, fields) in enumerate(zip(self.config['network_feat_name'], self.config['network_feat_field'])):
                if len(name) == 2:
                    net_name, link_name = name
                    net_field, link_field = fields
                    link = self._load_feat(os.path.join(data_dir, link_name), self.config['network_feat_header'][i][1], field_sep, link_field, update_dict=False).to_numpy()
                    self.node_link[i] = dict(link)
                    self.node_relink[i] = dict(link[:, [1, 0]])
                    feat = self._load_feat(os.path.join(data_dir, net_name), self.config['network_feat_header'][i][0], field_sep, net_field)
                    for j, col in enumerate(feat.columns):
                        if self.mapped_fields[i][j] != None:
                            feat[col] = [(self.node_relink[i][id] if id in self.node_relink[i] else id) for id in feat[col]]
                    self.network_feat[i] = feat
                else:
                    net_name, net_field = name[0], fields[0]
                    self.network_feat[i] = self._load_feat(os.path.join(data_dir, net_name), self.config['network_feat_header'][i][0], field_sep, net_field)

    def _fill_nan(self, feat, mapped=False):
        """Fill the missing data in the original data.

        For token type, `[PAD]` token is used.
        For float type, the mean value is used.
        For token_seq type and float_seq, the empty numpy array is used.
        """
        for field in feat:
            ftype = self.field2type[field]
            if ftype == 'float':
                feat[field].fillna(value=feat[field].mean(), inplace=True)
            elif ftype == 'token':
                feat[field].fillna(value=0 if mapped else '[PAD]', inplace=True)
            elif ftype == 'token_seq':
                dtype = np.int64 if mapped else str
                feat[field] = feat[field].map(lambda x: np.array([], dtype=dtype) if isinstance(x, float) else x)
            elif ftype == 'float_seq':
                feat[field] = feat[field].map(lambda x: np.array([], dtype=np.float64) if isinstance(x, float) else x)
            else:
                raise ValueError(f'field type {ftype} is not supported.                     Only supports float, token, token_seq, float_seq.')

    def _load_feat(self, feat_path, header, sep, feat_cols, update_dict=True):
        """Load the feature from a given a feature file."""
        fields = []
        types_of_fields = []
        seq_seperators = {}
        for feat in feat_cols:
            s = feat.split(':')
            fields.append(s[0])
            types_of_fields.append(s[1])
            if len(s) == 3:
                seq_seperators[s[0]] = s[2].split('"')[1]
        dtype = [(np.float64 if _ == 'float' else str) for _ in types_of_fields]
        if update_dict:
            self.field2type.update(dict(zip(fields, types_of_fields)))
        if not 'encoding_method' in self.config:
            self.config['encoding_method'] = 'utf-8'
        if self.config['encoding_method'] is None:
            self.config['encoding_method'] = 'utf-8'
        feat = pd.read_csv(feat_path, sep=sep, header=header, names=fields, dtype=dict(zip(fields, dtype)), engine='python', index_col=False, encoding=self.config['encoding_method'])[list(fields)]
        for i, (col, t) in enumerate(zip(fields, types_of_fields)):
            if not t.endswith('seq'):
                if update_dict and col not in self.field2maxlen:
                    self.field2maxlen[col] = 1
                continue
            feat[col].fillna(value='', inplace=True)
            cast = float if 'float' in t else str
            feat[col] = feat[col].map(lambda _: np.array(list(map(cast, filter(None, _.split(seq_seperators[col])))), dtype=cast))
            if update_dict and col not in self.field2maxlen:
                self.field2maxlen[col] = feat[col].map(len).max()
        return feat

    def _get_map_fields(self):
        if self.config['network_feat_name'] is not None:
            network_fields = {col: self.mapped_fields[i][j] for i, net in enumerate(self.network_feat) for j, col in enumerate(net.columns) if self.mapped_fields[i][j] != None}
        else:
            network_fields = {}
        fields_share_space = [[f] for f, t in self.field2type.items() if 'token' in t and f not in network_fields]
        for k, v in network_fields.items():
            for field_set in fields_share_space:
                if v in field_set:
                    field_set.append(k)
        return fields_share_space

    def _get_feat_list(self):
        feat_list = [self.inter_feat, self.user_feat, self.item_feat]
        if self.config['network_feat_name'] is not None:
            feat_list.extend(self.network_feat)
        return feat_list

    def _map_all_ids(self):
        """Map tokens to index."""
        fields_share_space = self._get_map_fields()
        feat_list = self._get_feat_list()
        for field_set in fields_share_space:
            flag = self.config['network_feat_name'] is not None and (self.fuid in field_set or self.fiid in field_set)
            token_list = []
            field_feat = [(field, feat, idx) for field in field_set for idx, feat in enumerate(feat_list) if feat is not None and field in feat]
            for field, feat, _ in field_feat:
                if 'seq' not in self.field2type[field]:
                    token_list.append(feat[field].values)
                else:
                    token_list.append(feat[field].agg(np.concatenate))
            count_inter_user_or_item = sum(1 for x in field_feat if x[-1] < 3)
            split_points = np.cumsum([len(_) for _ in token_list])
            token_list = np.concatenate(token_list)
            tid_list, tokens = pd.factorize(token_list)
            max_user_or_item_id = np.max(tid_list[:split_points[count_inter_user_or_item - 1]]) + 1 if flag else 0
            if '[PAD]' not in set(tokens):
                tokens = np.insert(tokens, 0, '[PAD]')
                tid_list = np.split(tid_list + 1, split_points[:-1])
                token2id = {tok: i for i, tok in enumerate(tokens)}
                max_user_or_item_id += 1
            else:
                token2id = {tok: i for i, tok in enumerate(tokens)}
                tid = token2id['[PAD]']
                tokens[tid] = tokens[0]
                token2id[tokens[0]] = tid
                tokens[0] = '[PAD]'
                token2id['[PAD]'] = 0
                idx_0, idx_1 = tid_list == 0, tid_list == tid
                tid_list[idx_0], tid_list[idx_1] = tid, 0
                tid_list = np.split(tid_list, split_points[:-1])
            for (field, feat, idx), _ in zip(field_feat, tid_list):
                if field not in self.field2tokens:
                    if flag:
                        if field in [self.fuid, self.fiid]:
                            self.field2tokens[field] = tokens[:max_user_or_item_id]
                            self.field2token2idx[field] = {tokens[i]: i for i in range(max_user_or_item_id)}
                        else:
                            tokens_ori = self._get_ori_token(idx - 3, tokens)
                            self.field2tokens[field] = tokens_ori
                            self.field2token2idx[field] = {t: i for i, t in enumerate(tokens_ori)}
                    else:
                        self.field2tokens[field] = tokens
                        self.field2token2idx[field] = token2id
                if 'seq' not in self.field2type[field]:
                    feat[field] = _
                    feat[field] = feat[field].astype('Int64')
                else:
                    sp_point = np.cumsum(feat[field].agg(len))[:-1]
                    feat[field] = np.split(_, sp_point)

    def _get_ori_token(self, idx, tokens):
        if self.node_link[idx] is not None:
            return [(self.node_link[idx][tok] if tok in self.node_link[idx] else tok) for tok in tokens]
        else:
            return tokens

    def _prepare_user_item_feat(self):
        if self.user_feat is not None:
            self.user_feat.set_index(self.fuid, inplace=True)
            self.user_feat = self.user_feat.reindex(np.arange(self.num_users))
            self.user_feat.reset_index(inplace=True)
            self._fill_nan(self.user_feat, mapped=True)
        else:
            self.user_feat = pd.DataFrame({self.fuid: np.arange(self.num_users)})
        if self.item_feat is not None:
            self.item_feat.set_index(self.fiid, inplace=True)
            self.item_feat = self.item_feat.reindex(np.arange(self.num_items))
            self.item_feat.reset_index(inplace=True)
            self._fill_nan(self.item_feat, mapped=True)
        else:
            self.item_feat = pd.DataFrame({self.fiid: np.arange(self.num_items)})

    def _post_preprocess(self):
        if self.ftime in self.inter_feat:
            if self.field2type[self.ftime] == 'str':
                assert 'time_format' in self.config, 'time_format is required when timestamp is string.'
                time_format = self.config['time_format']
                self.inter_feat[self.ftime] = pd.to_datetime(self.inter_feat[self.ftime], format=time_format)
            elif self.field2type[self.ftime] == 'float':
                pass
            else:
                raise ValueError(f'The field [{self.ftime}] should be float or str type')
        self._prepare_user_item_feat()

    def _recover_unmapped_feature(self, feat):
        feat = feat.copy()
        for field in feat:
            if field in self.field2tokens:
                feat[field] = feat[field].map(lambda x: self.field2tokens[field][x])
        return feat

    def _drop_duplicated_pairs(self):
        first_item_idx = ~self.inter_feat.duplicated(subset=[self.fuid, self.fiid], keep='first')
        self.inter_feat = self.inter_feat[first_item_idx]

    def _filter(self, min_user_inter, min_item_inter):
        self._filter_ratings()
        if self.drop_dup:
            self._drop_duplicated_pairs()
        item_list = self.inter_feat[self.fiid]
        item_idx_list, items = pd.factorize(item_list)
        user_list = self.inter_feat[self.fuid]
        user_idx_list, users = pd.factorize(user_list)
        user_item_mat = ssp.csc_matrix((np.ones_like(user_idx_list), (user_idx_list, item_idx_list)))
        cols = np.arange(items.size)
        rows = np.arange(users.size)
        while True:
            m, n = user_item_mat.shape
            col_sum = np.squeeze(user_item_mat.sum(axis=0).A)
            col_ind = col_sum >= min_item_inter
            col_count = np.count_nonzero(col_ind)
            if col_count > 0:
                cols = cols[col_ind]
                user_item_mat = user_item_mat[:, col_ind]
            row_sum = np.squeeze(user_item_mat.sum(axis=1).A)
            row_ind = row_sum >= min_user_inter
            row_count = np.count_nonzero(row_ind)
            if row_count > 0:
                rows = rows[row_ind]
                user_item_mat = user_item_mat[row_ind, :]
            if col_count == n and row_count == m:
                break
            else:
                pass
        keep_users = set(users[rows])
        keep_items = set(items[cols])
        keep = user_list.isin(keep_users)
        keep &= item_list.isin(keep_items)
        self.inter_feat = self.inter_feat[keep]
        self.inter_feat.reset_index(drop=True, inplace=True)
        if self.user_feat is not None:
            self.user_feat = self.user_feat[self.user_feat[self.fuid].isin(keep_users)]
            self.user_feat.reset_index(drop=True, inplace=True)
        if self.item_feat is not None:
            self.item_feat = self.item_feat[self.item_feat[self.fiid].isin(keep_items)]
            self.item_feat.reset_index(drop=True, inplace=True)

    def get_graph(self, idx, form='coo', value_fields=None, row_offset=0, col_offset=0, bidirectional=False, shape=None):
        """
        Returns a single graph or a graph composed of several networks. If more than one graph is passed into the methods, ``shape`` must be specified.

        Args:
            idx(int, list): the indices of the feat or networks. The index of ``inter_feat`` is set to ``0`` by default
            and the index of networks(such as knowledge graph and social network) is started by ``1`` corresponding to the dataset configuration file i.e. ``datasetname.yaml``.
            form(str): the form of the returned graph, can be 'coo', 'csr' or 'dgl'. Default: ``None``.
            value_fields(str, list): the value field in each graph. If value_field isn't ``None``, the values in this column will fill the adjacency matrix.
            row_offset(int, list): the offset of each row in corrresponding graph.
            col_offset(int, list): the offset of each column in corrresponding graph.
            bidirectional(bool, list): whether to turn the graph into bidirectional graph or not. Default: False
            shape(tuple): the shape of the returned graph. If more than one graph is passed into the methods, ``shape`` must be specified.

        Returns:
           graph(coo_matrix, csr_matrix or DGLGraph): a single graph or a graph composed of several networks in specified form.
           If the form is ``DGLGraph``, the relaiton type of the edges is stored in graph.edata['value'].
           num_relations(int): the number of relations in the combined graph.
           [ ['pad'], relation_0_0, relation_0_1, ..., relation_0_n, ['pad'], relation_1_0, relation_1_1, ..., relation_1_n]
        """
        if type(idx) == int:
            idx = [idx]
        if type(value_fields) == str or value_fields == None:
            value_fields = [value_fields] * len(idx)
        if type(bidirectional) == bool or bidirectional == None:
            bidirectional = [bidirectional] * len(idx)
        if type(row_offset) == int or row_offset == None:
            row_offset = [row_offset] * len(idx)
        if type(col_offset) == int or col_offset == None:
            col_offset = [col_offset] * len(idx)
        assert len(idx) == len(value_fields) and len(idx) == len(bidirectional)
        if shape is not None:
            assert type(shape) == list or type(shape) == tuple, 'the type of shape should be list or tuple'
        rows, cols, vals = [], [], []
        n, m, val_off = 0, 0, 0
        for id, value_field, bidirectional, row_off, col_off in zip(idx, value_fields, bidirectional, row_offset, col_offset):
            tmp_rows, tmp_cols, tmp_vals, val_off, tmp_n, tmp_m = self._get_one_graph(id, value_field, row_off, col_off, val_off, bidirectional)
            rows.append(tmp_rows)
            cols.append(tmp_cols)
            vals.append(tmp_vals)
            n += tmp_n
            m += tmp_m
        if shape == None or type(shape) != tuple and type(shape) != list:
            if len(idx) > 1:
                raise ValueError(f'If the length of idx is larger than 1, user should specify the shape of the combined graph.')
            else:
                shape = n, m
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals)
        if form == 'coo':
            from scipy.sparse import coo_matrix
            return coo_matrix((vals, (rows, cols)), shape), val_off
        elif form == 'csr':
            from scipy.sparse import csr_matrix
            return csr_matrix((vals, (rows, cols)), shape), val_off
        elif form == 'dgl':
            assert shape[0] == shape[1], 'only support homogeneous graph in form of dgl, shape[0] must epuals to shape[1].'
            graph = dgl.graph((rows, cols), num_nodes=shape[0])
            graph.edata['value'] = vals
            return graph, val_off

    def _get_one_graph(self, id, value_field=None, row_offset=0, col_offset=0, val_offset=0, bidirectional=False):
        """
        Gets rows, cols and values in one graph.
        If several graphs are to be combined into one, offset should be added on the edge value in each graph to avoid conflict.
        Then the edge value will be: .. math:: offset + vals. (.. math:: offset + 1 in user-item graph). The offset will be reset to ``offset + len(self.field2tokens[value_field])`` in next graph.
        If bidirectional is True, the inverse edge values in the graph will be set to ``offset + corresponding_canonical_values + len(self.field2tokens[value_field]) - 1``.
        If all edges in the graph are sorted by their values in a list, the list will be:
            ['[PAD]', canonical_edge_1, canonical_edge_2, ..., canonical_edge_n, inverse_edge_1, inverse_edge_2, ..., inverse_edge_n]

        Args:
            id(int): the indix of the feat or network. The index of ``inter_feat`` is set to ``0`` by default
            and the index of networks(such as knowledge graph and social network) is started by ``1`` corresponding to the dataset configuration file i.e. ``datasetname.yaml``.
            value_field(str): the value field in the graph. If value_field isn't ``None``, the values in this column will fill the adjacency matrix.
            row_offset(int): the offset of the row in the graph. Default: 0.
            col_offset(int): the offset of the column in the graph. Default: 0.
            val_offset(int): the offset of the edge value in the graph. If several graphs are to be combined into one,
            offset should be added on the edge value in each graph to avoid conflict. Default: 0.
            bidirectional(bool): whether to turn the graph into bidirectional graph or not. Default: False

        Returns:
            rows(torch.Tensor): source nodes in all edges in the graph.
            cols(torch.Tensor): destination nodes in all edges in the graph.
            values(torch.Tensor): values of all edges in the graph.
            num_rows(int): number of source nodes.
            num_cols(int): number of destination nodes.
        """
        if id == 0:
            source_field = self.fuid
            target_field = self.fiid
            feat = self.inter_feat[self.inter_feat_subset]
        elif self.network_feat is not None:
            if id - 1 < len(self.network_feat):
                feat = self.network_feat[id - 1]
                if len(feat.fields) == 2:
                    source_field, target_field = feat.fields[:2]
                elif len(feat.fields) == 3:
                    source_field, target_field = feat.fields[0], feat.fields[2]
            else:
                raise ValueError(f'idx [{id}] is larger than the number of network features [{len(self.network_feat)}] minus 1')
        else:
            raise ValueError(f'No network feature is input while idx [{id}] is larger than 1')
        if id == 0:
            source = feat[source_field] + row_offset
            target = feat[target_field] + col_offset
        else:
            source = feat.get_col(source_field) + row_offset
            target = feat.get_col(target_field) + col_offset
        if bidirectional:
            rows = torch.cat([source, target])
            cols = torch.cat([target, source])
        else:
            rows = source
            cols = target
        if value_field is not None:
            if id == 0 and value_field == 'inter':
                if bidirectional:
                    vals = torch.tensor([val_offset + 1] * len(source) + [val_offset + 2] * len(source))
                    val_offset += 1 + 2
                else:
                    vals = torch.tensor([val_offset + 1] * len(source))
                    val_offset += 1 + 1
            elif value_field in feat.fields:
                if bidirectional:
                    vals = feat.get_col(value_field) + val_offset
                    inv_vals = feat.get_col(value_field) + len(self.field2tokens[value_field]) - 1 + val_offset
                    vals = torch.cat([vals, inv_vals])
                    val_offset += 2 * len(self.field2tokens[value_field]) - 1
                else:
                    vals = feat.get_col(value_field) + val_offset
                    val_offset += len(self.field2tokens[value_field])
            else:
                raise ValueError(f'valued_field [{value_field}] does not exist')
        else:
            vals = torch.ones(len(rows))
        return rows, cols, vals, val_offset, self.num_values(source_field), self.num_values(target_field)

    def _split_by_ratio(self, ratio, data_count, user_mode):
        """Split dataset into train/valid/test by specific ratio."""
        m = len(data_count)
        if not user_mode:
            splits = np.outer(data_count, ratio).astype(np.int32)
            splits[:, 0] = data_count - splits[:, 1:].sum(axis=1)
            for i in range(1, len(ratio)):
                idx = (splits[:, -i] == 0) & (splits[:, 0] > 1)
                splits[idx, -i] += 1
                splits[idx, 0] -= 1
        else:
            idx = np.random.permutation(m)
            sp_ = (m * np.array(ratio)).astype(np.int32)
            sp_[0] = m - sp_[1:].sum()
            sp_ = sp_.cumsum()
            parts = np.split(idx, sp_[:-1])
            splits = np.zeros((m, len(ratio)), dtype=np.int32)
            for _, p in zip(range(len(ratio)), parts):
                splits[p, _] = data_count.iloc[p]
        splits = np.hstack([np.zeros((m, 1), dtype=np.int32), np.cumsum(splits, axis=1)])
        cumsum = np.hstack([[0], data_count.cumsum()[:-1]])
        splits = cumsum.reshape(-1, 1) + splits
        return splits, data_count.index if m > 1 else None

    def _split_by_num(self, num, data_count):
        """Split dataset into train/valid/test by specific ratio.
        num: list of int
        assert split_mode is entry                       
        """
        m = len(data_count)
        splits = np.hstack([0, num]).cumsum().reshape(1, -1)
        if splits[0][-1] == data_count.values.sum():
            return splits, data_count.index if m > 1 else None
        else:
            ValueError(f'Expecting the number of interactions             should be equal to the sum of {num}')

    def _split_by_leave_one_out(self, leave_one_num, data_count, rep=True):
        """Split dataset into train/valid/test by leave one out method.
        The split methods are usually used for sequential recommendation, where the last item of the item sequence will be used for test.

        Args:
            leave_one_num(int): the last ``leave_one_num`` items of the sequence will be splited out.
            data_count(pandas.DataFrame or numpy.ndarray):  entry range for each user or number of all entries.
            rep(bool, optional): whether there should be repititive items in the sequence.
        """
        m = len(data_count)
        cumsum = data_count.cumsum()[:-1]
        if rep:
            splits = np.ones((m, leave_one_num + 1), dtype=np.int32)
            splits[:, 0] = data_count - leave_one_num
            for _ in range(leave_one_num):
                idx = splits[:, 0] < 1
                splits[idx, 0] += 1
                splits[idx, _] -= 1
            splits = np.hstack([np.zeros((m, 1), dtype=np.int32), np.cumsum(splits, axis=1)])
        else:

            def get_splits(bool_index):
                idx = bool_index.values.nonzero()[0]
                if len(idx) > 2:
                    return [0, idx[-2], idx[-1], len(idx)]
                elif len(idx) == 2:
                    return [0, idx[-1], idx[-1], len(idx)]
                else:
                    return [0, len(idx), len(idx), len(idx)]
            splits = np.array([get_splits(bool_index) for bool_index in np.split(self.first_item_idx, cumsum)])
        cumsum = np.hstack([[0], cumsum])
        splits = cumsum.reshape(-1, 1) + splits
        return splits, data_count.index if m > 1 else None

    def _get_data_idx(self, splits):
        """ Return data index for train/valid/test dataset.
        """
        splits, uids = splits
        data_idx = [list(zip(splits[:, i - 1], splits[:, i])) for i in range(1, splits.shape[1])]
        if not getattr(self, 'fmeval', False):
            if uids is not None:
                d = [torch.from_numpy(np.hstack([np.arange(*e) for e in data_idx[0]]))]
                for _ in data_idx[1:]:
                    d.append(torch.tensor([[u, *e] for u, e in zip(uids, _) if e[1] > e[0]]))
                return d
            else:
                d = [torch.from_numpy(np.hstack([np.arange(*e) for e in data_idx[0]]))]
                for _ in data_idx[1:]:
                    start, end = _[0]
                    data = self.inter_feat.get_col(self.fuid)[start:end]
                    uids, counts = data.unique_consecutive(return_counts=True)
                    cumsum = torch.hstack([torch.tensor([0]), counts.cumsum(-1)]) + start
                    d.append(torch.tensor([[u, st, en] for u, st, en in zip(uids, cumsum[:-1], cumsum[1:])]))
                return d
        else:
            return [torch.from_numpy(np.hstack([np.arange(*e) for e in _])) for _ in data_idx]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data_index)

    def _get_pos_data(self, index):
        if self.data_index.dim() > 1:
            idx = self.data_index[index]
            data = {self.fuid: idx[:, 0]}
            data.update(self.user_feat[data[self.fuid]])
            start = idx[:, 1]
            end = idx[:, 2]
            lens = end - start
            l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            d = self.inter_feat.get_col(self.fiid)[l]
            rating = self.inter_feat.get_col(self.frating)[l]
            data[self.fiid] = pad_sequence(d.split(tuple(lens.numpy())), batch_first=True)
            data[self.frating] = pad_sequence(rating.split(tuple(lens.numpy())), batch_first=True)
        else:
            idx = self.data_index[index]
            data = self.inter_feat[idx]
            uid, iid = data[self.fuid], data[self.fiid]
            data.update(self.user_feat[uid])
            data.update(self.item_feat[iid])
        if 'user_hist' in data:
            user_count = self.user_count[data[self.fuid]].max()
            data['user_hist'] = data['user_hist'][:, 0:user_count]
        return data

    def _get_neg_data(self, data: Dict):
        if 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]].max()
            user_hist = self.user_hist[data[self.fuid]][:, 0:user_count]
        else:
            user_hist = data['user_hist']
        neg_id = uniform_sampling(data[self.frating.size(0)], self.num_items, self.neg_count, user_hist).long()
        neg_id = neg_id.transpose(0, 1).contiguous().view(-1)
        neg_item_feat = self.item_feat[neg_id]
        for k, v in data.items():
            if k in neg_item_feat:
                data[k] = torch.cat([v, neg_item_feat[k]], dim=0)
            elif k != self.frating:
                data[k] = v.tile((self.neg_count + 1,))
            else:
                neg_rating = torch.zeros_like(neg_id)
                data[k] = torch.cat((v, neg_rating), dim=0)
        return data

    def __getitem__(self, index):
        """Get data at specific index.

        Args:
            index(int): The data index.
        Returns:
            dict: A dict contains different feature.
        """
        data = self._get_pos_data(index)
        if self.eval_mode and 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]].max()
            data['user_hist'] = self.user_hist[data[self.fuid]][:, 0:user_count]
        elif getattr(self, 'neg_count', None) is not None:
            data = self._get_neg_data(data)
        return data

    def _copy(self, idx):
        d = copy.copy(self)
        d.data_index = idx
        return d

    def _init_sampler(self, dataset_sampler, dataset_neg_count):
        self.neg_count = dataset_neg_count
        self.sampler = dataset_sampler
        if self.sampler is not None:
            assert self.sampler == 'uniform', '`dataset_sampler` only support uniform sampler now.'
            assert self.neg_count is not None, '`dataset_neg_count` are required when `dataset_sampler` is used.'
            self.logger.warning('The rating of the sampled negatives will be set as 0.')
            if not self.config['drop_low_rating']:
                self.logger.warning('Please attention the `drop_low_rating` is False and the dataset is a rating dataset, the sampled negatives will be treated as interactions with rating 0.')
            self.logger.warning(f'With the sampled negatives, the batch size will be {self.neg_count + 1} times as the batch size set in the configuration file. For example, `batch_size=16` and `dataset_neg_count=2` will load batches with size 48.')

    def build(self, split_ratio: List=[0.8, 0.1, 0.1], shuffle: bool=True, split_mode: str='user_entry', fmeval: bool=False, dataset_sampler: str=None, dataset_neg_count: int=None, **kwargs):
        """Build dataset.

        Args:
            split_ratio(numeric): split ratio for data preparition. If given list of float, the dataset will be splited by ratio. If given a integer, leave-n method will be used.

            shuffle(bool, optional): set True to reshuffle the whole dataset each epoch. Default: ``True``

            split_mode(str, optional): controls the split mode. If set to ``user_entry``, then the interactions of each user will be splited into 3 cut.
            If ``entry``, then dataset is splited by interactions. If ``user``, all the users will be splited into 3 cut. Default: ``user_entry``

            fmeval(bool, optional): set True for MFDataset and ALSDataset when use TowerFreeRecommender. Default: ``False``

        Returns:
            list: A list contains train/valid/test data-[train, valid, test]
        """
        self.fmeval = fmeval
        self.split_mode = split_mode
        self._init_sampler(dataset_sampler, dataset_neg_count)
        return self._build(split_ratio, shuffle, split_mode, False)

    def _build(self, ratio_or_num, shuffle, split_mode, rep):
        if not hasattr(self, 'first_item_idx'):
            self.first_item_idx = ~self.inter_feat.duplicated(subset=[self.fuid, self.fiid], keep='first')
        if self.drop_dup:
            self.inter_feat = self.inter_feat[self.first_item_idx]
        if split_mode == 'user_entry' or split_mode == 'user':
            if self.ftime in self.inter_feat:
                self.inter_feat.sort_values(by=[self.fuid, self.ftime], inplace=True)
                self.inter_feat.reset_index(drop=True, inplace=True)
            else:
                self.inter_feat.sort_values(by=self.fuid, inplace=True)
                self.inter_feat.reset_index(drop=True, inplace=True)
        if split_mode == 'user_entry':
            user_count = self.inter_feat[self.fuid].groupby(self.inter_feat[self.fuid], sort=False).count()
            if shuffle:
                cumsum = np.hstack([[0], user_count.cumsum()[:-1]])
                idx = np.concatenate([(np.random.permutation(c) + start) for start, c in zip(cumsum, user_count)])
                self.inter_feat = self.inter_feat.iloc[idx].reset_index(drop=True)
        elif split_mode == 'entry':
            if isinstance(ratio_or_num, list) and isinstance(ratio_or_num[0], int):
                user_count = self.inter_feat[self.fuid].groupby(self.inter_feat[self.fuid], sort=True).count()
            else:
                if shuffle:
                    self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)
                user_count = np.array([len(self.inter_feat)])
        elif split_mode == 'user':
            user_count = self.inter_feat[self.fuid].groupby(self.inter_feat[self.fuid], sort=False).count()
        if isinstance(ratio_or_num, int):
            splits = self._split_by_leave_one_out(ratio_or_num, user_count, rep)
        elif isinstance(ratio_or_num, list) and isinstance(ratio_or_num[0], float):
            splits = self._split_by_ratio(ratio_or_num, user_count, split_mode == 'user')
        else:
            splits = self._split_by_num(ratio_or_num, user_count)
        splits_ = splits[0][0]
        if split_mode == 'entry':
            if isinstance(self, AEDataset) or isinstance(self, SeqDataset):
                ucnts = pd.DataFrame({self.fuid: splits[1]})
                for i, (start, end) in enumerate(zip(splits_[:-1], splits_[1:])):
                    self.inter_feat[start:end] = self.inter_feat[start:end].sort_values(by=[self.fuid, self.ftime] if self.ftime in self.inter_feat else self.fuid)
                    ucnts[i] = self.inter_feat[start:end][self.fuid].groupby(self.inter_feat[self.fuid], sort=True).count().values
                self.inter_feat.sort_values(by=[self.fuid], inplace=True, kind='mergesort')
                self.inter_feat.reset_index(drop=True, inplace=True)
                ucnts = ucnts.astype(int)
                ucnts = torch.from_numpy(ucnts.values)
                u_cumsum = ucnts[:, 1:].cumsum(dim=1)
                u_start = torch.hstack([torch.tensor(0), u_cumsum[:, -1][:-1]]).view(-1, 1).cumsum(dim=0)
                splits = torch.hstack([u_start, u_cumsum + u_start])
                uids = ucnts[:, 0]
                if isinstance(self, AEDataset):
                    splits = splits, uids.view(-1, 1)
                else:
                    splits = splits.numpy(), uids
            else:
                for start, end in zip(splits_[:-1], splits_[1:]):
                    self.inter_feat[start:end] = self.inter_feat[start:end].sort_values(by=[self.fuid, self.ftime] if self.ftime in self.inter_feat else self.fuid)
        self.dataframe2tensors()
        datasets = [self._copy(_) for _ in self._get_data_idx(splits)]
        user_hist, user_count = datasets[0].get_hist(True)
        for d in datasets[:2]:
            d.user_hist = user_hist
            d.user_count = user_count
        if len(datasets) > 2:
            assert len(datasets) == 3
            uh, uc = datasets[1].get_hist(True)
            uh = torch.cat((user_hist, uh), dim=-1).sort(dim=-1, descending=True).values
            uc = uc + user_count
            datasets[-1].user_hist = uh
            datasets[-1].user_count = uc
        return datasets

    def dataframe2tensors(self):
        """Convert the data type from TensorFrame to Tensor
        """
        self.inter_feat = TensorFrame.fromPandasDF(self.inter_feat, self)
        self.user_feat = TensorFrame.fromPandasDF(self.user_feat, self)
        self.item_feat = TensorFrame.fromPandasDF(self.item_feat, self)
        if hasattr(self, 'network_feat'):
            for i in range(len(self.network_feat)):
                self.network_feat[i] = TensorFrame.fromPandasDF(self.network_feat[i], self)

    def train_loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False, ddp=False):
        """Return a dataloader for training.

        Args:
            batch_size(int): the batch size for training data.

            shuffle(bool,optimal): set to True to have the data reshuffled at every epoch. Default:``True``.

            num_workers(int, optimal): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process. (default: ``1``)

            drop_last(bool, optimal): set to True to drop the last mini-batch if the size is smaller than given batch size. Default: ``False``

            load_combine(bool, optimal): set to True to combine multiple loaders as :doc:`ChainedDataLoader <chaineddataloader>`. Default: ``False``

        Returns:
            list or ChainedDataLoader: list of loaders if load_combine is True else ChainedDataLoader.

        .. note::
            Due to that index is used to shuffle the dataset and the data keeps remained, `num_workers > 0` may get slower speed.
        """
        self.eval_mode = False
        return self.loader(batch_size, shuffle, num_workers, drop_last, ddp)

    def loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False, ddp=False):
        if self.data_index.dim() > 1:
            sampler = SortedDataSampler(self, batch_size, shuffle, drop_last)
        else:
            sampler = DataSampler(self, batch_size, shuffle, drop_last)
        if ddp:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False)
        output = DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=False)
        return output

    @property
    def sample_length(self):
        if self.data_index.dim() > 1:
            return self.data_index[:, 2] - self.data_index[:, 1]
        else:
            raise ValueError('can not compute sample length for this dataset')

    def eval_loader(self, batch_size, num_workers=1, ddp=False):
        self.eval_mode = True
        if not getattr(self, 'fmeval', False):
            sampler = SortedDataSampler(self, batch_size)
            if ddp:
                sampler = DistributedSamplerWrapper(sampler, shuffle=False)
            output = DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=False)
            return output
        else:
            self.eval_mode = True
            return self.loader(batch_size, shuffle=False, num_workers=num_workers, ddp=ddp)

    def drop_feat(self, keep_fields):
        if keep_fields is not None and len(keep_fields) > 0:
            fields = set(keep_fields)
            fields.add(self.frating)
            for feat in self._get_feat_list():
                feat.del_fields(fields)
            if 'user_hist' in fields:
                self.user_feat.add_field('user_hist', self.user_hist)
            if 'item_hist' in fields:
                self.item_feat.add_field('item_hist', self.get_hist(False))

    def get_hist(self, isUser=True):
        """Get user or item interaction history.

        Args:
            isUser(bool, optional): Default: ``True``.

        Returns:
            torch.Tensor: padded user or item hisoty.

            torch.Tensor: length of the history sequence.
        """
        user_array = self.inter_feat.get_col(self.fuid)[self.inter_feat_subset]
        item_array = self.inter_feat.get_col(self.fiid)[self.inter_feat_subset]
        sorted, index = torch.sort(user_array if isUser else item_array)
        user_item, count = torch.unique_consecutive(sorted, return_counts=True)
        list_ = torch.split(item_array[index] if isUser else user_array[index], tuple(count.numpy()))
        tensors = [torch.tensor([], dtype=torch.int64) for _ in range(self.num_users if isUser else self.num_items)]
        for i, l in zip(user_item, list_):
            tensors[i] = l
        user_count = torch.tensor([len(e) for e in tensors])
        tensors = pad_sequence(tensors, batch_first=True)
        return tensors, user_count

    def get_network_field(self, network_id, feat_id, field_id):
        """
        Returns the specified field name in some network.
        For example, if the head id field is in the first feat of KG network and is the first column of the feat and the index of KG network is 1.
        To get the head id field, the method can be called like this ``train_data.get_network_field(1, 0, 0)``.

        Args:
            network_id(int) : the index of network corresponding to the dataset configuration file.
            feat_id(int): the index of the feat in the network.
            field_id(int): the index of the wanted field in above feat.

        Returns:
            field(str): the wanted field.
        """
        return self.config['network_feat_field'][network_id][feat_id][field_id].split(':')[0]

    @property
    def inter_feat_subset(self):
        """ Data index.
        """
        if self.data_index.dim() > 1:
            return torch.cat([torch.arange(s, e) for s, e in zip(self.data_index[:, 1], self.data_index[:, 2])])
        else:
            return self.data_index

    @property
    def item_freq(self):
        """ Item frequency (or popularity).

        Returns:
            torch.Tensor: ``[num_items,]``. The times of each item appears in the dataset.
        """
        if not hasattr(self, 'data_index'):
            raise ValueError('please build the dataset first by call the build method')
        l = self.inter_feat.get_col(self.fiid)[self.inter_feat_subset]
        it, count = torch.unique(l, return_counts=True)
        it_freq = torch.zeros(self.num_items, dtype=torch.int64)
        it_freq[it] = count
        return it_freq

    @property
    def num_users(self):
        """Number of users.

        Returns:
            int: number of users.
        """
        return self.num_values(self.fuid)

    @property
    def num_items(self):
        """Number of items.

        Returns:
            int: number of items.
        """
        return self.num_values(self.fiid)

    @property
    def num_inters(self):
        """Number of total interaction numbers.

        Returns:
            int: number of interactions in the dataset.
        """
        return len(self.inter_feat)

    def num_values(self, field):
        """Return number of values in specific field.

        Args:
            field(str): the field to be counted.

        Returns:
            int: number of values in the field.

        .. note::
            This method is used to return ``num_items``, ``num_users`` and ``num_inters``.
        """
        if 'token' not in self.field2type[field]:
            return self.field2maxlen[field]
        else:
            return len(self.field2tokens[field])


def color_dict(dict_, keep=True):
    key_color = 'blue'
    val_color = 'yellow'

    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) + '=' + set_color(v_f, val_color, keep=keep)) % (k, v)
        return info
    des = 4
    if 'epoch' in dict_:
        start = set_color('Training: ', 'green', keep=keep)
        start += color_kv('Epoch', dict_['epoch'], '%s', '%3d')
    else:
        start = set_color('Testing: ', 'green', keep=keep)
    info = ' '.join([color_kv(k, v, '%s', '%.' + str(des) + 'f') for k, v in dict_.items() if k != 'epoch'])
    return start + ' [' + info + ']'


def seed_everything(seed: Optional[int]=None, workers: bool=False) ->int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:
    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.
    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    logger = logging.getLogger('recstudio')
    if seed is None:
        env_seed = os.environ.get('PL_GLOBAL_SEED')
        if env_seed is None:
            seed = random.randint(min_seed_value, max_seed_value)
            logger.warning(f'No seed found, seed set to {seed}')
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = random.randint(min_seed_value, max_seed_value)
                logger.warning(f'Invalid seed found: {repr(env_seed)}, seed set to {seed}')
    elif not isinstance(seed, int):
        seed = int(seed)
    if not min_seed_value <= seed <= max_seed_value:
        logger.warning(f'{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}')
        seed = random.randint(min_seed_value, max_seed_value)
    logger.info(f'Global seed set to {seed}')
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PL_SEED_WORKERS'] = f'{int(workers)}'
    return seed


class Recommender(torch.nn.Module, abc.ABC):

    def __init__(self, config: Dict=None, **kwargs):
        super(Recommender, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = parser_yaml(os.path.join(os.path.dirname(__file__), 'basemodel.yaml'))
        if self.config['seed'] is not None:
            seed_everything(self.config['seed'], workers=True)
        self.embed_dim = self.config['embed_dim']
        self.logged_metrics = {}
        if 'retriever' in kwargs:
            assert isinstance(kwargs['retriever'], basemodel.BaseRetriever), 'sampler must be recstudio.model.basemodel.BaseRetriever'
            self.retriever = kwargs['retriever']
        else:
            self.retriever = None
        if 'loss' in kwargs:
            assert isinstance(kwargs['loss'], loss_func.FullScoreLoss) or isinstance(kwargs['loss'], loss_func.PairwiseLoss) or isinstance(kwargs['loss'], loss_func.PointwiseLoss), 'loss should be one of: [recstudio.loss_func.FullScoreLoss,                 recstudio.loss_func.PairwiseLoss, recstudio.loss_func.PointWiseLoss]'
            self.loss_fn = kwargs['loss']
        else:
            self.loss_fn = None
        self.ckpt_path = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group('Recommender')
        parent_parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        parent_parser.add_argument('--learner', type=str, default='adam', help='optimization algorithm')
        parent_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay coefficient')
        parent_parser.add_argument('--epochs', type=int, default=50, help='training epochs')
        parent_parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parent_parser.add_argument('--eval_batch_size', type=int, default=128, help='evaluation batch size')
        parent_parser.add_argument('--val_n_epoch', type=int, default=1, help='valid epoch interval')
        parent_parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
        parent_parser.add_argument('--early_stop_patience', type=int, default=10, help='early stop patience')
        parent_parser.add_argument('--gpu', type=int, action='append', default=None, help='gpu number')
        parent_parser.add_argument('--init_method', type=str, default='xavier_normal', help='init method for model')
        parent_parser.add_argument('--init_range', type=float, help='init range for some methods like normal')
        parent_parser.add_argument('--seed', type=int, default=2022, help='random seed')
        return parent_parser

    def _add_modules(self, train_data):
        pass

    def _set_data_field(self, data):
        pass

    def _init_model(self, train_data, drop_unused_field=True):
        self._set_data_field(train_data)
        self.fields = train_data.use_field
        self.frating = train_data.frating
        assert self.frating in self.fields, 'rating field is required.'
        if drop_unused_field:
            train_data.drop_feat(self.fields)
        self.item_feat = train_data.item_feat
        self.item_fields = set(train_data.item_feat.fields).intersection(self.fields)
        self.neg_count = self.config['negative_count']
        if self.loss_fn is None:
            if 'train_data' in inspect.signature(self._get_loss_func).parameters:
                self.loss_fn = self._get_loss_func(train_data)
            else:
                self.loss_fn = self._get_loss_func()

    def fit(self, train_data: MFDataset, val_data: Optional[MFDataset]=None, run_mode='light', config: Dict=None, **kwargs) ->None:
        """
        Fit the model with train data.
        """
        self.logger = logging.getLogger('recstudio')
        if len(self.logger.handlers) > 1:
            tb_log_name = os.path.basename(self.logger.handlers[1].baseFilename).split('.')[0]
        else:
            import time
            tb_log_name = time.strftime(f'{self.__class__.__name__}-{train_data.name}-%Y-%m-%d-%H-%M-%S.log', time.localtime())
        self.tensorboard_logger = SummaryWriter(f'tensorboard/{tb_log_name}')
        if config is not None:
            self.config.update(config)
        if kwargs is not None:
            self.config.update(kwargs)
        self._init_model(train_data)
        self._init_parameter()
        self.run_mode = run_mode
        self.val_check = val_data is not None and self.config['val_metrics'] is not None
        if val_data is not None:
            val_data.use_field = train_data.use_field
        if self.val_check:
            self.val_metric = next(iter(self.config['val_metrics'])) if isinstance(self.config['val_metrics'], list) else self.config['val_metrics']
            cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
            if len(eval.get_rank_metrics(self.val_metric)) > 0:
                self.val_metric += '@' + str(cutoffs[0])
        self.callback = self._get_callback(train_data.name)
        self.logger.info('save_dir:' + self.callback.save_dir)
        self.logger.info(self)
        self._accelerate()
        if self.config['accelerator'] == 'ddp':
            mp.spawn(self.parallel_training, args=(self.world_size, train_data, val_data), nprocs=self.world_size, join=True)
        else:
            self.trainloaders = self._get_train_loaders(train_data)
            if val_data:
                val_loader = val_data.eval_loader(batch_size=self.config['eval_batch_size'], num_workers=self.config['num_workers'])
            else:
                val_loader = None
            self.optimizers = self._get_optimizers()
            self.fit_loop(val_loader)
        return self.callback.best_ckpt['metric']

    def parallel_training(self, rank, world_size, train_data, val_data):
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        self.trainloaders = self._get_train_loaders(train_data, ddp=True)
        if val_data:
            val_loader = val_data.eval_loader(batch_size=self.config['eval_batch_size'], num_workers=self.config['num_workers'], ddp=True)
        else:
            val_loader = None
        self.device = self.device_list[rank]
        self = self
        self = DDP(self, device_ids=[self.device], output_device=self.device).module
        self.optimizers = self._get_optimizers()
        self.fit_loop(val_loader)
        dist.destroy_process_group()

    def evaluate(self, test_data, verbose=True, **kwargs) ->Dict:
        """ Predict for test data.

        Args:
            test_data(recstudio.data.Dataset): The dataset of test data, which is generated by RecStudio.

            verbose(bool, optimal): whether to show the detailed information.

        Returns:
            dict: dict of metrics. The key is the name of metrics.
        """
        test_data.drop_feat(self.fields)
        test_loader = test_data.eval_loader(batch_size=self.config['eval_batch_size'], num_workers=self.config['num_workers'])
        output = {}
        self.load_checkpoint(os.path.join(self.config['save_path'], self.ckpt_path))
        if 'config' in kwargs:
            self.config.update(kwargs['config'])
        self.eval()
        output_list = self.test_epoch(test_loader)
        output.update(self.test_epoch_end(output_list))
        if self.run_mode == 'tune':
            output['default'] = output[self.val_metric]
        if verbose:
            self.logger.info(color_dict(output, self.run_mode == 'tune'))
        return output

    def predict(self, batch, k, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, batch):
        pass

    def _get_callback(self, dataset_name):
        save_dir = self.config['save_path']
        if self.val_check:
            return callbacks.EarlyStopping(self, self.val_metric, dataset_name, save_dir=save_dir, patience=self.config['early_stop_patience'], mode=self.config['early_stop_mode'])
        else:
            return callbacks.SaveLastCallback(self, dataset_name, save_dir=save_dir)

    def current_epoch_trainloaders(self, nepoch) ->Tuple:
        """
        Returns:
            list or dict or Dataloader : the train loaders used in the current epoch
            bool : whether to combine the train loaders or use them alternately in one epoch.
        """
        combine = False
        return self.trainloaders, combine

    def current_epoch_optimizers(self, nepoch) ->List:
        return self.optimizers

    @abc.abstractmethod
    def build_index(self):
        pass

    @abc.abstractmethod
    def training_step(self, batch):
        pass

    @abc.abstractmethod
    def validation_step(self, batch):
        pass

    @abc.abstractmethod
    def test_step(self, batch):
        pass

    def training_epoch_end(self, output_list):
        output_list = [output_list] if not isinstance(output_list, list) else output_list
        for outputs in output_list:
            if isinstance(outputs, List):
                loss_metric = {('train_' + k): torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
            elif isinstance(outputs, torch.Tensor):
                loss_metric = {'train_loss': outputs.item()}
            elif isinstance(outputs, Dict):
                loss_metric = {('train_' + k): v for k, v in outputs}
            self.log_dict(loss_metric)
        if self.val_check and self.run_mode == 'tune':
            metric = self.logged_metrics[self.val_metric]
        if self.run_mode in ['light', 'tune'] or self.val_check:
            self.logger.info(color_dict(self.logged_metrics, self.run_mode == 'tune'))
        else:
            self.logger.info('\n' + color_dict(self.logged_metrics, self.run_mode == 'tune'))

    def validation_epoch_end(self, outputs):
        val_metric = self.config['val_metrics'] if isinstance(self.config['val_metrics'], list) else [self.config['val_metrics']]
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config.setdefault('cutoff', None)]
        val_metric = [(f'{m}@{cutoff}' if len(eval.get_rank_metrics(m)) > 0 else m) for cutoff in cutoffs[:1] for m in val_metric]
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end(outputs)
            out = dict(zip(val_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end(outputs)
        self.log_dict(out)
        return out

    def test_epoch_end(self, outputs):
        test_metric = self.config['test_metrics'] if isinstance(self.config['test_metrics'], list) else [self.config['test_metrics']]
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config.setdefault('cutoff', None)]
        test_metric = [(f'{m}@{cutoff}' if len(eval.get_rank_metrics(m)) > 0 else m) for cutoff in cutoffs for m in test_metric]
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end(outputs)
            out = dict(zip(test_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end(outputs)
        self.log_dict(out, tensorboard=False)
        return out

    def _test_epoch_end(self, outputs):
        if isinstance(outputs[0][0], List):
            metric, bs = zip(*outputs)
            metric = torch.tensor(metric)
            bs = torch.tensor(bs)
            out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        elif isinstance(outputs[0][0], Dict):
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = (metric * bs).sum() / bs.sum()
        return out

    def log_dict(self, metrics: Dict, tensorboard: bool=True):
        if tensorboard:
            for k, v in metrics.items():
                if 'train' in k:
                    self.tensorboard_logger.add_scalar(f'train/{k}', v, self.logged_metrics['epoch'] + 1)
                else:
                    self.tensorboard_logger.add_scalar(f'valid/{k}', v, self.logged_metrics['epoch'] + 1)
        self.logged_metrics.update(metrics)

    def _init_parameter(self):
        init_methods = {'xavier_normal': init.xavier_normal_initialization, 'xavier_uniform': init.xavier_uniform_initialization, 'normal': init.normal_initialization}
        for name, module in self.named_children():
            if isinstance(module, Recommender):
                module._init_parameter()
            else:
                if self.config['init_method'] == 'normal':
                    init_method = init.normal_initialization(self.config['init_range'])
                else:
                    init_method = init_methods[self.config['init_method']]
                module.apply(init_method)

    @staticmethod
    def _get_dataset_class():
        pass

    def _get_loss_func(self):
        return None

    def _get_item_feat(self, data):
        if isinstance(data, dict):
            if len(self.item_fields) == 1:
                return data[self.fiid]
            else:
                return dict((field, value) for field, value in data.items() if field in self.item_fields)
        elif len(self.item_fields) == 1:
            return data
        else:
            return self.item_feat[data]

    def _get_train_loaders(self, train_data, ddp=False) ->List:
        return [train_data.train_loader(batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], drop_last=False, ddp=ddp)]

    def _get_optimizers(self) ->List[Dict]:
        if 'learner' not in self.config:
            self.config['learner'] = 'adam'
            self.logger.warning('`learner` is not detected in the configuration, Adam optimizer is used.')
        if self.config['learner'] is None:
            self.logger.warning('There is no optimizers in the model due to `learner` isset as None in configurations.')
            return None
        if isinstance(self.config['learner'], list):
            self.logger.warning('If you want to use multi learner, please override `_get_optimizers` function. We will use the first learner for all the parameters.')
            opt_name = self.config['learner'][0]
            lr = self.config['learning_rate'][0]
            weight_decay = None if self.config['weight_decay'] is None else self.config['weight_decay'][0]
            scheduler_name = None if self.config['scheduler'] is None else self.config['scheduler'][0]
        else:
            opt_name = self.config['learner']
            if 'learning_rate' not in self.config:
                self.logger.warning('`learning_rate` is not detected in the configurations, the default learning is set as 0.001.')
                self.config['learning_rate'] = 0.001
            lr = self.config['learning_rate']
            if 'weight_decay' not in self.config:
                self.logger.warning('`weight_decay` is not detected in the configurations, the default weight_decay is set as 0.')
                self.config['weight_decay'] = 0
            weight_decay = self.config['weight_decay']
            scheduler_name = self.config['scheduler']
        params = self.parameters()
        optimizer = self._get_optimizer(opt_name, params, lr, weight_decay)
        scheduler = self._get_scheduler(scheduler_name, optimizer)
        m = self.val_metric if self.val_check else 'train_loss'
        if scheduler:
            return [{'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': m, 'interval': 'epoch', 'frequency': 1, 'strict': False}}]
        else:
            return [{'optimizer': optimizer}]

    def _get_optimizer(self, name, params, lr, weight_decay):
        """Return optimizer for specific parameters.

        The optimizer can be configured in the config file with the key ``learner``.
        Supported optimizer: ``Adam``, ``SGD``, ``AdaGrad``, ``RMSprop``, ``SparseAdam``.

        .. note::
            If no learner is assigned in the configuration file, then ``Adam`` will be user.

        Args:
            params: the parameters to be optimized.

        Returns:
            torch.optim.optimizer: optimizer according to the config.
        """
        learning_rate = lr
        decay = weight_decay
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self, name, optimizer):
        """Return learning rate scheduler for the optimizer.

        Args:
            optimizer(torch.optim.Optimizer): the optimizer which need a scheduler.

        Returns:
            torch.optim.lr_scheduler: the learning rate scheduler.
        """
        if name is not None:
            if name.lower() == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            elif name.lower() == 'onplateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            else:
                scheduler = None
        else:
            scheduler = None
        return scheduler

    def fit_loop(self, val_dataloader=None):
        try:
            nepoch = 0
            for e in range(self.config['epochs']):
                self.logged_metrics = {}
                self.logged_metrics['epoch'] = nepoch
                tik_train = time.time()
                self.train()
                training_output_list = self.training_epoch(nepoch)
                tok_train = time.time()
                tik_valid = time.time()
                if self.val_check:
                    self.eval()
                    if nepoch % self.config['val_n_epoch'] == 0:
                        validation_output_list = self.validation_epoch(nepoch, val_dataloader)
                        self.validation_epoch_end(validation_output_list)
                tok_valid = time.time()
                self.training_epoch_end(training_output_list)
                if self.config['gpu'] is not None:
                    mem_reversed = torch.cuda.max_memory_reserved(self._parameter_device) / 1024 ** 3
                    mem_total = torch.cuda.mem_get_info(self._parameter_device)[1] / 1024 ** 3
                else:
                    mem_reversed = mem_total = 0
                self.logger.info('{} {:.5f}s. {} {:.5f}s. {} {:.2f}/{:.2f} GB'.format(set_color('Train time:', 'white', False), tok_train - tik_train, set_color('Valid time:', 'white', False), tok_valid - tik_valid, set_color('GPU RAM:', 'white', False), mem_reversed, mem_total))
                optimizers = self.current_epoch_optimizers(e)
                if optimizers is not None:
                    for opt in optimizers:
                        if 'scheduler' in opt:
                            opt['scheduler'].step()
                if nepoch % self.config['val_n_epoch'] == 0:
                    stop_training = self.callback(self, nepoch, self.logged_metrics)
                    if stop_training:
                        break
                nepoch += 1
            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        except KeyboardInterrupt:
            if self.config['accelerator'] == 'ddp':
                if dist.get_rank() == 0:
                    self.callback.save_checkpoint(nepoch)
                    self.ckpt_path = self.callback.get_checkpoint_path()
            else:
                self.callback.save_checkpoint(nepoch)
                self.ckpt_path = self.callback.get_checkpoint_path()

    def training_epoch(self, nepoch):
        if hasattr(self, '_update_item_vector'):
            self._update_item_vector()
        if hasattr(self, 'sampler'):
            if hasattr(self.sampler, 'update'):
                if hasattr(self, 'item_vector'):
                    self.sampler.update(item_embs=self.item_vector)
                else:
                    self.sampler.update(item_embs=None)
        output_list = []
        optimizers = self.current_epoch_optimizers(nepoch)
        trn_dataloaders, combine = self.current_epoch_trainloaders(nepoch)
        if isinstance(trn_dataloaders, List) or isinstance(trn_dataloaders, Tuple):
            if combine:
                trn_dataloaders = [CombinedLoaders(list(trn_dataloaders))]
        else:
            trn_dataloaders = [trn_dataloaders]
        if not (isinstance(optimizers, List) or isinstance(optimizers, Tuple)):
            optimizers = [optimizers]
        for loader_idx, loader in enumerate(trn_dataloaders):
            outputs = []
            for batch_idx, batch in enumerate(loader):
                batch = self._to_device(batch, self._parameter_device)
                for opt in optimizers:
                    if opt is not None:
                        opt['optimizer'].zero_grad()
                training_step_args = {'batch': batch}
                if 'nepoch' in inspect.getargspec(self.training_step).args:
                    training_step_args['nepoch'] = nepoch
                if 'loader_idx' in inspect.getargspec(self.training_step).args:
                    training_step_args['loader_idx'] = loader_idx
                if 'batch_idx' in inspect.getargspec(self.training_step).args:
                    training_step_args['batch_idx'] = batch_idx
                loss = self.training_step(**training_step_args)
                if isinstance(loss, dict):
                    if loss['loss'].requires_grad:
                        if isinstance(loss['loss'], torch.Tensor):
                            loss['loss'].backward()
                        elif isinstance(loss['loss'], List):
                            for l in loss['loss']:
                                l.backward()
                        else:
                            raise TypeError('loss must be Tensor or List of Tensor')
                    loss_ = {}
                    for k, v in loss.items():
                        if k == 'loss':
                            if isinstance(v, torch.Tensor):
                                v = v.detach()
                            elif isinstance(v, List):
                                v = [_ for _ in v]
                        loss_[f'{k}_{loader_idx}'] = v
                    outputs.append(loss_)
                elif isinstance(loss, torch.Tensor):
                    if loss.requires_grad:
                        loss.backward()
                    outputs.append({f'loss_{loader_idx}': loss.detach()})
                for opt in optimizers:
                    if self.config['grad_clip_norm'] is not None:
                        clip_grad_norm_(opt['optimizer'].params, self.config['grad_clip_norm'])
                    if opt is not None:
                        opt['optimizer'].step()
                if len(outputs) > 0:
                    output_list.append(outputs)
        return output_list

    def validation_epoch(self, nepoch, dataloader):
        if hasattr(self, '_update_item_vector'):
            self._update_item_vector()
        output_list = []
        for batch in dataloader:
            batch = self._to_device(batch, self._parameter_device)
            output = self.validation_step(batch)
            output_list.append(output)
        return output_list

    def test_epoch(self, dataloader):
        if hasattr(self, '_update_item_vector'):
            self._update_item_vector()
        output_list = []
        for batch in dataloader:
            batch = self._to_device(batch, self._parameter_device)
            output = self.test_step(batch)
            output_list.append(output)
        return output_list

    @staticmethod
    def _set_device(gpus):
        if gpus is not None:
            gpus = [str(i) for i in gpus]
            os.environ['CUDA_VISIBLE_DEVICES'] = ' '.join(gpus)

    @staticmethod
    def _to_device(batch, device):
        if isinstance(batch, torch.Tensor) or isinstance(batch, torch.nn.Module):
            return batch
        elif isinstance(batch, Dict):
            for k in batch:
                batch[k] = Recommender._to_device(batch[k], device)
            return batch
        elif isinstance(batch, List) or isinstance(batch, Tuple):
            output = []
            for b in batch:
                output.append(Recommender._to_device(b, device))
            return output if isinstance(batch, List) else tuple(output)
        else:
            raise TypeError(f'`batch` is expected to be torch.Tensor, Dict,                 List or Tuple, but {type(batch)} given.')

    def _accelerate(self):
        gpu_list = get_gpus(self.config['gpu'])
        if gpu_list is not None:
            self.logger.info(f'GPU id {gpu_list} are selected.')
            if len(gpu_list) == 1:
                self.device = torch.device('cuda', gpu_list[0])
                self = self._to_device(self, self.device)
            elif self.config['accelerator'] == 'dp':
                self.device = torch.device('cuda')
                self = self
                self = torch.nn.DataParallel(self, device_ids=gpu_list, output_device=gpu_list[0])
            elif self.config['accelerator'] == 'ddp':
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '29500'
                self.device_list = [torch.device('cuda', i) for i in gpu_list]
                self.world_size = len(self.device_list)
                raise NotImplementedError("'ddp' not implemented.")
            else:
                raise ValueError(f"expecting accelerator to be 'dp' or 'ddp'while get {self.config['accelerator']} instead.")
        else:
            self.device = torch.device('cpu')
            self = self._to_device(self, self.device)

    @property
    def _parameter_device(self):
        if len(list(self.parameters())) == 0:
            return torch.device('cpu')
        else:
            return next(self.parameters()).device

    def _get_ckpt_param(self):
        """
        Returns:
            OrderedDict: the parameters to be saved as check point.
        """
        return self.state_dict()

    def save_checkpoint(self, ckpt: Dict) ->str:
        save_path = os.path.join(self.config['save_path'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        best_ckpt_path = os.path.join(save_path, self._best_ckpt_path)
        torch.save(ckpt, best_ckpt_path)
        self.logger.info('Best model checkpoint saved in {}.'.format(best_ckpt_path))

    def load_checkpoint(self, path: str) ->None:
        ckpt = torch.load(path)
        self.config = ckpt['config']
        if hasattr(self, '_update_item_vector') and not hasattr(self, 'item_vector'):
            self._update_item_vector()
        self.load_state_dict(ckpt['parameters'])


class FullScoreLoss(torch.nn.Module):
    """Calculate loss with positive scores and scores on all items.

    The loss need user's perference scores on positive items(ground truth) and all other items.
    However, due to the item numbers are very huge in real-world datasets, calculating scores on all items
    may be very time-consuming. So the loss is seldom used in large-scale dataset.
    """

    def forward(self, label, pos_score, all_score):
        """
        """
        pass


class PairwiseLoss(torch.nn.Module):

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        pass


class PointwiseLoss(torch.nn.Module):

    def forward(self, label, pos_score):
        raise NotImplementedError(f'{type(self).__name__} is an abstrat class,             this method would not be implemented')


class SquareLoss(PointwiseLoss):

    def forward(self, label, pos_score):
        if label.dim() > 1:
            return torch.mean(torch.mean(torch.square(label - pos_score), dim=-1))
        else:
            return torch.mean(torch.square(label - pos_score))


class SoftmaxLoss(FullScoreLoss):

    def forward(self, label, pos_score, all_score):
        if all_score.dim() > pos_score.dim():
            return torch.mean(torch.logsumexp(all_score, dim=-1) - pos_score)
        else:
            output = torch.logsumexp(all_score, dim=-1, keepdim=True) - pos_score
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
            return torch.mean(output)


class BPRLoss(PairwiseLoss):

    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return -torch.mean((loss * weight).sum(-1))
        else:
            loss = -torch.mean(F.logsigmoid(pos_score - torch.max(neg_score, dim=-1)))
            return loss


class Top1Loss(BPRLoss):

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = torch.sigmoid(neg_score - pos_score.view(*pos_score.shape, 1))
            loss += torch.sigmoid(neg_score ** 2)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return torch.mean((loss * weight).sum(-1))
        else:
            max_neg_score = torch.max(neg_score, dim=-1)
            loss = torch.sigmoid(max_neg_score - pos_score)
            loss = loss + torch.sigmoid(max_neg_score ** 2)
        return loss


class SampledSoftmaxLoss(PairwiseLoss):

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        if new_pos.dim() < new_neg.dim():
            new_pos.unsqueeze_(-1)
        new_neg = torch.cat([new_pos, new_neg], dim=-1)
        output = torch.logsumexp(new_neg, dim=-1, keepdim=True) - new_pos
        notpadnum = torch.logical_not(torch.isinf(new_pos)).float().sum(-1)
        output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        return torch.mean(output)


class WeightedBPRLoss(PairwiseLoss):

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        weight = F.softmax(neg_score - log_neg_prob, -1)
        return -torch.mean((loss * weight).sum(-1))


class BinaryCrossEntropyLoss(PairwiseLoss):

    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        assert pos_score.dim() == neg_score.dim() - 1 and pos_score.shape == neg_score.shape[:-1] or pos_score.dim() == neg_score.dim()
        if not self.dns:
            weight = self._cal_weight(neg_score, log_neg_prob)
            padding_mask = torch.isinf(pos_score)
            pos_loss = F.logsigmoid(pos_score)
            pos_loss.masked_fill_(padding_mask, 0.0)
            pos_loss = pos_loss.sum() / (~padding_mask).sum()
            neg_loss = F.softplus(neg_score) * weight
            neg_loss = neg_loss.sum(-1)
            if pos_score.dim() == neg_score.dim() - 1:
                neg_loss.masked_fill_(padding_mask, 0.0)
                neg_loss = neg_loss.sum() / (~padding_mask).sum()
            else:
                neg_loss = torch.mean(neg_loss)
            return -pos_loss + neg_loss
        else:
            return torch.mean(-F.logsigmoid(pos_score) + F.softplus(torch.max(neg_score, dim=-1)))

    def _cal_weight(self, neg_score, log_neg_prob):
        return torch.ones_like(neg_score) / neg_score.size(-1)


class WeightedBinaryCrossEntropyLoss(BinaryCrossEntropyLoss):

    def _cal_weight(self, neg_score, log_neg_prob):
        return F.softmax(neg_score - log_neg_prob, -1)


class HingeLoss(PairwiseLoss):

    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, label, pos_score, log_pos_prob, neg_score, neg_prob):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score + self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            return torch.mean(loss * torch.log(rank + 1))
        else:
            return torch.mean(loss)


class InfoNCELoss(torch.nn.Module):
    """
    Parameters: 
    neg_type(str): 'batch_both', 'batch_single', 'all'
    """

    def __init__(self, temperature: float=1.0, sim_method: str='inner_product', neg_type: str='batch_both') ->None:
        super().__init__()
        self.temperature = temperature
        self.sim_method = sim_method
        self.neg_type = neg_type

    def forward(self, augmented_rep_i: torch.Tensor, augmented_rep_j: torch.Tensor, instance_labels=None, all_reps: Optional[torch.Tensor]=None):
        if self.neg_type == 'batch_both':
            assert all_reps == None, "all_reps is used when the negative strategy is 'all'."
            batch_size = augmented_rep_i.size(0)
            if self.sim_method == 'inner_product':
                sim_ii = torch.matmul(augmented_rep_i, augmented_rep_i.T) / self.temperature
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature
            elif self.sim_method == 'cosine':
                augmented_rep_i = F.normalize(augmented_rep_i, p=2, dim=-1)
                augmented_rep_j = F.normalize(augmented_rep_j, p=2, dim=-1)
                sim_ii = torch.matmul(augmented_rep_i, augmented_rep_i.T) / self.temperature
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature
            if instance_labels is not None:
                """
                do de-noise as ICLRec, if data_1 and data_2 have the same label, 
                then (data_1_i, data_2_i) and (data_1_i, data_2_j) won't be treated as negative samples.
                """
                mask = torch.eq(instance_labels.unsqueeze(-1), instance_labels)
                sim_ii[mask == 1] = float('-inf')
                mask = mask.fill_diagonal_(False)
                sim_ij[mask == 1] = float('-inf')
            else:
                mask = torch.eye(batch_size, dtype=torch.long)
                sim_ii[mask == 1] = float('-inf')
            logits = torch.cat([sim_ij, sim_ii], dim=-1)
            labels = torch.arange(batch_size, dtype=torch.long, device=augmented_rep_i.device)
            loss = F.cross_entropy(logits, labels)
            return loss
        elif self.neg_type == 'batch_single':
            assert all_reps == None, "all_reps is used when the negative strategy is 'all'."
            batch_size = augmented_rep_i.size(0)
            if self.sim_method == 'inner_product':
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature
            elif self.sim_method == 'cosine':
                augmented_rep_i = F.normalize(augmented_rep_i, p=2, dim=-1)
                augmented_rep_j = F.normalize(augmented_rep_j, p=2, dim=-1)
                sim_ij = torch.matmul(augmented_rep_i, augmented_rep_j.T) / self.temperature
            if instance_labels is not None:
                """
                do de-noise as ICLRec, if data_1 and data_2 have the same label, 
                then (data_1_i, data_2_i) and (data_1_i, data_2_j) won't be treated as negative samples.
                """
                mask = torch.eq(instance_labels.unsqueeze(-1), instance_labels)
                mask = mask.fill_diagonal_(False)
                sim_ij[mask == 1] = float('-inf')
            labels = torch.arange(batch_size, dtype=torch.long, device=augmented_rep_i.device)
            loss = F.cross_entropy(sim_ij, labels)
            return loss
        elif self.neg_type == 'all':
            assert all_reps != None, "all_reps shouldn't be None."
            assert instance_labels == None, "instance_labels is used when the negative strategy is 'batch'."
            batch_size = augmented_rep_i.size(0)
            if self.sim_method == 'inner_product':
                sim_ij = torch.matmul(augmented_rep_i, all_reps.T) / self.temperature
                sim_ii = (augmented_rep_i * augmented_rep_j).sum(dim=-1) / self.temperature
            elif self.sim_method == 'cosine':
                augmented_rep_i = F.normalize(augmented_rep_i, p=2, dim=-1)
                augmented_rep_j = F.normalize(augmented_rep_j, p=2, dim=-1)
                all_reps = F.normalize(all_reps, p=2, dim=-1)
                sim_ij = torch.matmul(augmented_rep_i, all_reps.T) / self.temperature
                sim_ii = (augmented_rep_i * augmented_rep_j).sum(dim=-1) / self.temperature
            loss = torch.mean(torch.logsumexp(sim_ij, dim=-1) - sim_ii)
            return loss
        else:
            raise ValueError(f'{self.neg_type} is not supported, neg_type should be "batch" or "all".')


class NCELoss(PairwiseLoss):

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        loss = F.logsigmoid(new_pos) + (new_neg - F.softplus(new_neg)).sum(1)
        return -loss.mean()


class CCLLoss(PairwiseLoss):

    def __init__(self, margin=0.8, neg_weight=0.3) ->None:
        super().__init__()
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        pos_score = torch.sigmoid(pos_score)
        neg_score = torch.sigmoid(neg_score)
        neg_score_mean = torch.mean(torch.relu(neg_score - self.margin), dim=-1)
        notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum()
        loss = 1 - pos_score + self.neg_weight * neg_score_mean
        loss = torch.nan_to_num(loss, posinf=0.0)
        return loss.sum() / notpadnum


class BCEWithLogitLoss(PointwiseLoss):

    def __init__(self, threshold: float=3.0, reduction: str='mean') ->None:
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, label, pos_score):
        label = (label > self.threshold).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_score, label, reduction=self.reduction)
        return loss


class MSELoss(PointwiseLoss):

    def __init__(self, threshold: float=None, reduction: str='mean') ->None:
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, label, pos_score):
        if self.threshold is not None:
            label = (label > self.threshold).float()
        loss = torch.nn.functional.mse_loss(pos_score, label)
        return loss


class DenseEmbedding(torch.nn.Module):

    def __init__(self, embedding_dim, bias=False, batch_norm=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bias = bias
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layer = torch.nn.BatchNorm1d(1)
        self.weight = torch.nn.Linear(1, embedding_dim, bias=bias)

    def forward(self, input):
        input = input.view(-1, 1)
        if self.batch_norm:
            input = self.batch_norm_layer(input)
        emb = self.weight(input)
        return emb

    def extra_repr(self):
        return f'embedding_dim={self.embedding_dim}, bias={self.bias}, batch_norm={self.batch_norm}'


class SeqPoolingLayer(torch.nn.Module):

    def __init__(self, pooling_type='mean', keepdim=False) ->None:
        super().__init__()
        if not pooling_type in ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']:
            raise ValueError(f"pooling_type can only be one of ['origin', 'mask', 'concat', 'sum', 'mean', 'max', 'last']but {pooling_type} is given.")
        self.pooling_type = pooling_type
        self.keepdim = keepdim

    def forward(self, batch_seq_embeddings, seq_len, weight=None, mask_token=None):
        B = batch_seq_embeddings.size(0)
        _need_reshape = False
        if batch_seq_embeddings.dim() == 4:
            _need_reshape = True
            batch_seq_embeddings = batch_seq_embeddings.view(-1, *batch_seq_embeddings.shape[2:])
            seq_len = seq_len.view(-1)
            if weight is not None:
                weight = weight.view(-1, weight.size(-1))
        N, L, D = batch_seq_embeddings.shape
        if weight is not None:
            batch_seq_embeddings = weight.unsqueeze(-1) * batch_seq_embeddings
        if self.pooling_type == 'mask':
            assert mask_token != None, "mask_token can be None when pooling_type is 'mask'."
            result = batch_seq_embeddings[mask_token]
        elif self.pooling_type in ['origin', 'concat', 'mean', 'sum', 'max']:
            mask = torch.arange(L).unsqueeze(0).unsqueeze(2)
            mask = mask.expand(N, -1, D)
            seq_len = seq_len.unsqueeze(1).unsqueeze(2)
            seq_len_ = seq_len.expand(-1, mask.size(1), -1)
            mask = mask >= seq_len_
            batch_seq_embeddings = batch_seq_embeddings.masked_fill(mask, 0.0)
            if self.pooling_type == 'origin':
                return batch_seq_embeddings
            elif self.pooling_type in ['concat', 'max']:
                if not self.keepdim:
                    if self.pooling_type == 'concat':
                        result = batch_seq_embeddings.reshape(N, -1)
                    else:
                        result = batch_seq_embeddings.max(dim=1)
                elif self.pooling_type == 'concat':
                    result = batch_seq_embeddings.reshape(N, -1).unsqueeze(1)
                else:
                    result = batch_seq_embeddings.max(dim=1).unsqueeze(1)
            elif self.pooling_type in ['mean', 'sum']:
                batch_seq_embeddings_sum = batch_seq_embeddings.sum(dim=1, keepdim=self.keepdim)
                if self.pooling_type == 'sum':
                    result = batch_seq_embeddings_sum
                else:
                    result = batch_seq_embeddings_sum / (seq_len + torch.finfo(torch.float32).eps if self.keepdim else seq_len.squeeze(2))
        elif self.pooling_type == 'last':
            gather_index = (seq_len - 1).view(-1, 1, 1).expand(-1, -1, D)
            output = batch_seq_embeddings.gather(dim=1, index=gather_index).squeeze(1)
            result = output if not self.keepdim else output.unsqueeze(1)
        if _need_reshape:
            return result.reshape(B, N // B, *result.shape[1:])
        else:
            return result

    def extra_repr(self):
        return f'pooling_type={self.pooling_type}, keepdim={self.keepdim}'


class Embeddings(torch.nn.Module):

    def __init__(self, fields: Set, embed_dim, data, reduction='mean', share_dense_embedding=False, dense_emb_bias=False, dense_emb_norm=True):
        super(Embeddings, self).__init__()
        self.embed_dim = embed_dim
        self.field2types = {f: data.field2type[f] for f in fields if f != data.frating}
        self.reduction = reduction
        self.share_dense_embedding = share_dense_embedding
        self.dense_emb_bias = dense_emb_bias
        self.dense_emb_norm = dense_emb_norm
        self.embeddings = torch.nn.ModuleDict()
        self.num_features = len(self.field2types)
        _num_token_seq_feat = 0
        _num_dense_feat = 0
        _dense_feat = []
        for f, t in self.field2types.items():
            if t == 'token' or t == 'token_seq':
                if t == 'token_seq':
                    _num_token_seq_feat += 1
                self.embeddings[f] = torch.nn.Embedding(data.num_values(f), embed_dim, 0)
            elif t == 'float':
                if share_dense_embedding:
                    _num_dense_feat += 1
                    _dense_feat.append(f)
                else:
                    self.embeddings[f] = DenseEmbedding(embed_dim, dense_emb_bias, dense_emb_norm)
        if _num_dense_feat > 0:
            dense_emb = DenseEmbedding(embed_dim, dense_emb_bias, dense_emb_norm)
            for f in _dense_feat:
                self.embeddings[f] = dense_emb
        if _num_token_seq_feat > 0:
            self.seq_pooling_layer = SeqPoolingLayer(reduction, keepdim=False)

    def forward(self, batch):
        embs = []
        for f in self.embeddings:
            d = batch[f]
            t = self.field2types[f]
            if t == 'token' or t == 'float':
                e = self.embeddings[f](d)
            else:
                length = (d > 0).float().sum(dim=-1, keepdim=False)
                seq_emb = self.embeddings[f](d)
                e = self.seq_pooling_layer(seq_emb, length)
            embs.append(e)
        emb = torch.stack(embs, dim=-2)
        return emb

    def extra_repr(self):
        s = 'num_features={num_features}, embed_dim={embed_dim}, reduction={reduction}'
        if self.share_dense_embedding:
            s += ', share_dense_embedding={share_dense_embedding}'
        if self.dense_emb_bias:
            s += ', dense_emb_bias={dense_emb_bias}'
        if not self.dense_emb_norm:
            s += ', dense_emb_norm={dense_emb_norm}'
        return s.format(**self.__dict__)


class LinearLayer(Embeddings):

    def __init__(self, fields, data, bias=True):
        super(LinearLayer, self).__init__(fields, 1, data)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1))
        else:
            self.bias = None

    def forward(self, batch):
        embs = super().forward(batch).squeeze(-1)
        sum_of_embs = torch.sum(embs, dim=-1)
        return sum_of_embs + self.bias

    def extra_repr(self):
        if self.bias is None:
            bias = False
        else:
            bias = True
        return f'bias={bias}'


class FMLayer(nn.Module):

    def __init__(self, first_order=True, reduction=None):
        super(FMLayer, self).__init__()
        self.reduction = reduction
        if reduction is not None:
            if reduction not in {'sum', 'mean'}:
                raise ValueError(f'reduction only support `mean`|`sum`, but get {reduction}')

    def forward(self, inputs):
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        output = 0.5 * (square_of_sum - sum_of_square)
        if self.reduction is None:
            pass
        elif self.reduction == 'sum':
            output = output.sum(-1)
        else:
            output = output.mean(-1)
        return output

    def extra_repr(self):
        if self.reduction is None:
            reduction_repr = 'None'
        else:
            reduction_repr = self.reduction
        return f'reduction={reduction_repr}'


class CrossNetwork(torch.nn.Module):

    def __init__(self, embed_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.weight = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.embed_dim)) for _ in range(num_layers)])
        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.embed_dim)) for _ in range(num_layers)])

    def forward(self, input):
        x = input
        for i in range(self.num_layers):
            x_1 = torch.tensordot(x, self.weight[i], dims=([1], [0]))
            x_2 = (input.transpose(0, 1) * x_1).transpose(0, 1)
            x = x_2 + x + self.bias[i]
        return x

    def extra_repr(self) ->str:
        return f'embed_dim={self.embed_dim}, num_layers={self.num_layers}'


class AttentionLayer(torch.nn.Module):

    def __init__(self, q_dim, k_dim=None, v_dim=None, mlp_layers=[], activation='sigmoid', n_head=1, dropout=0.0, bias=True, attention_type='feedforward', batch_first=True) ->None:
        super().__init__()
        assert attention_type in set(['feedforward', 'multi-head', 'scaled-dot-product']), f'expecting attention_type to be one of [feedforeard, multi-head, scaled-dot-product]'
        self.attention_type = attention_type
        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = k_dim
        if attention_type == 'feedforward':
            mlp_layers = [q_dim + k_dim] + mlp_layers + [1]
            self.mlp = torch.nn.Sequential(MLPModule(mlp_layers=mlp_layers[:-1], activation_func=activation, bias=bias), torch.nn.Linear(mlp_layers[-2], mlp_layers[-1]))
            pass
        elif attention_type == 'multi-head':
            self.attn_layer = torch.nn.MultiheadAttention(embed_dim=q_dim, num_heads=n_head, dropout=dropout, bias=bias, kdim=k_dim, vdim=v_dim, batch_first=batch_first)
            pass
        elif attention_type == 'scaled-dot-product':
            assert q_dim == k_dim, 'expecting q_dim is equal to k_dim in scaled-dot-product attention'
            pass

    def forward(self, query, key, value, key_padding_mask=None, need_weight=False, attn_mask=None, softmax=False, average_attn_weights=True):
        if self.attention_type in ['feedforward', 'scaled-dot-product']:
            if self.attention_type == 'feedforward':
                query = query.unsqueeze(2).expand(-1, -1, key.size(1), -1)
                key = key.unsqueeze(1).expand(-1, query.size(1), -1, -1)
                attn_output_weight = self.mlp(torch.cat((query, key), dim=-1)).squeeze(-1)
            else:
                attn_output_weight = query @ key.transpose(1, 2)
            attn_output_weight = attn_output_weight / query.size(-1) ** 0.5
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, query.size(1), -1)
                filled_value = -torch.inf if softmax else 0.0
                attn_output_weight = attn_output_weight.masked_fill(key_padding_mask, filled_value)
            if softmax:
                attn_output_weight = torch.softmax(attn_output_weight, dim=-1)
            attn_output = attn_output_weight @ value
        elif self.attention_type == 'multi-head':
            attn_output, attn_output_weight = self.attn_layer(query, key, value, key_padding_mask, True, attn_mask, average_attn_weights)
        elif self.attention_type == 'scaled-dot-product':
            product = query @ key.transpose(1, 2)
            attn_output_weight = torch.softmax(product / torch.sqrt(query.size(-1)), dim=-1)
            attn_output = attn_output_weight @ value
        if need_weight:
            return attn_output, attn_output_weight
        else:
            return attn_output


class DINScorer(torch.nn.Module):

    def __init__(self, fuid, fiid, num_users, num_items, embed_dim, attention_mlp, dense_mlp, dropout=0.0, activation='sigmoid', batch_norm=False):
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.activation_unit = AttentionLayer(3 * embed_dim, embed_dim, mlp_layers=attention_mlp, activation=activation)
        norm = [torch.nn.BatchNorm1d(embed_dim)] if batch_norm else []
        norm.append(torch.nn.Linear(embed_dim, embed_dim))
        self.norm = torch.nn.Sequential(*norm)
        self.dense_mlp = MLPModule([3 * embed_dim] + dense_mlp, activation_func=activation, dropout=dropout, batch_norm=batch_norm)
        self.fc = torch.nn.Linear(dense_mlp[-1], 1)

    def forward(self, batch):
        seq_emb = self.item_embedding(batch['in_' + self.fiid])
        target_emb = self.item_embedding(batch[self.fiid])
        item_bias = self.item_bias(batch[self.fiid]).squeeze(-1)
        target_emb_ = target_emb.unsqueeze(1).repeat(1, seq_emb.size(1), 1)
        attn_seq = self.activation_unit(query=target_emb.unsqueeze(1), key=torch.cat((target_emb_, target_emb_ * seq_emb, target_emb_ - seq_emb), dim=-1), value=seq_emb, key_padding_mask=batch['in_' + self.fiid] == 0, softmax=False).squeeze(1)
        attn_seq = self.norm(attn_seq)
        cat_emb = torch.cat((attn_seq, target_emb, target_emb * attn_seq), dim=-1)
        score = self.fc(self.dense_mlp(cat_emb)).squeeze(-1)
        return score + item_bias


class BehaviorSequenceTransformer(nn.Module):

    def __init__(self, fuid, fiid, num_users, num_items, max_len, embed_dim, hidden_size, n_layer, n_head, dropout, mlp_layers=[1024, 512, 256], activation='leakyrelu', batch_first=True, norm_first=False):
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.max_len = max_len
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.position_embedding = torch.nn.Embedding(max_len + 2, embed_dim, 0)
        tfm_encoder = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=hidden_size, dropout=dropout, activation='relu', batch_first=batch_first, norm_first=norm_first)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer=tfm_encoder, num_layers=n_layer)
        self.mlp = MLPModule([(max_len + 1) * embed_dim] + mlp_layers, activation_func=activation, dropout=dropout)
        self.predict = torch.nn.Linear(mlp_layers[-1], 1)

    def forward(self, batch):
        hist = batch['in_' + self.fiid]
        target = batch[self.fiid]
        seq_len = batch['seqlen']
        hist = torch.cat((hist, torch.zeros_like(target.view(-1, 1))), dim=1)
        B, L = hist.shape
        idx_ = torch.arange(0, B, dtype=torch.long)
        hist[idx_, seq_len] = target.long()
        seq_emb = self.item_embedding(hist)
        positions = torch.arange(1, L + 1, device=seq_emb.device)
        positions = torch.tile(positions, (B,)).view(B, -1)
        padding_mask = hist == 0
        positions[padding_mask] = 0
        position_emb = self.position_embedding(positions)
        attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=hist.device), 1)
        tfm_out = self.transformer(src=seq_emb + position_emb, mask=attention_mask, src_key_padding_mask=padding_mask)
        padding_emb = tfm_out.new_zeros((B, self.max_len + 1 - L, tfm_out.size(-1)))
        tfm_out = torch.cat((tfm_out, padding_emb), dim=1)
        flatten_tfm_out = tfm_out.view(B, -1)
        logits = self.predict(self.mlp(flatten_tfm_out))
        return logits.squeeze(-1)


class DIENScorer(torch.nn.Module):

    def __init__(self, fuid, fiid, num_users, num_items, embed_dim, attention_mlp, fc_mlp, activation='sigmoid', batch_norm=False) ->None:
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.activation_unit = AttentionLayer(3 * embed_dim, embed_dim, mlp_layers=attention_mlp, activation=activation)
        self.norm = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim), torch.nn.Linear(embed_dim, embed_dim)) if batch_norm else torch.nn.Linear(embed_dim, embed_dim)
        self.fc = torch.nn.Sequential(torch.nn.BatchNorm1d(3 * embed_dim), MLPModule([3 * embed_dim] + fc_mlp, activation_func=activation), torch.nn.Linear(fc_mlp[-1], 1)) if batch_norm else torch.nn.Sequential(MLPModule([3 * embed_dim] + fc_mlp, activation_func=activation), torch.nn.Linear(fc_mlp[-1], 1))

    def forward(self, batch):
        pass


class CIN(torch.nn.Module):

    def __init__(self, embed_dim, num_features, cin_layer_size, activation='relu', direct=True):
        super(CIN, self).__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.cin_layer_size = _temp = cin_layer_size
        self.activation = get_act(activation)
        self.direct = direct
        self.weight = torch.nn.ModuleList()
        if not self.direct:
            self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), _temp))
            if self.cin_layer_size[:-1] != _temp[:-1]:
                self.logger.warning('Layer size of CIN should be even except for the last layer when direct is True.It is changed to {}'.format(self.cin_layer_size))
        self.weight_list = nn.ModuleList()
        self.field_num_list = [self.num_features]
        for i, layer_size in enumerate(self.cin_layer_size):
            conv1d = nn.Conv1d(self.field_num_list[-1] * self.field_num_list[0], layer_size, 1)
            self.weight_list.append(conv1d)
            if self.direct:
                self.field_num_list.append(layer_size)
            else:
                self.field_num_list.append(layer_size // 2)
        if self.direct:
            output_dim = sum(self.cin_layer_size)
        else:
            output_dim = sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
        self.linear = torch.nn.Linear(output_dim, 1)

    def forward(self, input):
        B, _, D = input.shape
        hidden_nn_layers = [input]
        final_result = []
        for i, layer_size in enumerate(self.cin_layer_size):
            z_i = torch.einsum('bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            z_i = z_i.view(B, self.field_num_list[0] * self.field_num_list[i], D)
            z_i = self.weight_list[i](z_i)
            output = self.activation(z_i)
            if self.direct:
                direct_connect = output
                next_hidden = output
            elif i != len(self.cin_layer_size) - 1:
                next_hidden, direct_connect = torch.split(output, 2 * [layer_size // 2], 1)
            else:
                direct_connect = output
                next_hidden = 0
            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        score = self.linear(result).squeeze(-1)
        return score


class Item_Crop(torch.nn.Module):

    def __init__(self, tao=0.2):
        super().__init__()
        self.tao = tao

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        croped_sequences = []
        croped_seq_lens = torch.zeros(batch_size, dtype=seq_lens.dtype, device=seq_lens.device)
        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            sub_len = max(1, int(self.tao * seq_len))
            croped_seq_lens[i] = sub_len
            start_index = torch.randint(low=0, high=seq_len - sub_len + 1, size=(1,)).item()
            croped_sequence = sequences[i][start_index:start_index + sub_len]
            croped_sequences.append(croped_sequence)
        return pad_sequence(croped_sequences, batch_first=True), croped_seq_lens


class Item_Mask(torch.nn.Module):

    def __init__(self, mask_id, gamma=0.7):
        super().__init__()
        self.gamma = gamma
        self.mask_id = mask_id

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        copied_sequence = copy.deepcopy(sequences)
        for i in range(batch_size):
            seq_len = seq_lens[i]
            sub_len = int(self.gamma * seq_len)
            mask_idx = np.random.choice(seq_len.item(), size=sub_len, replace=False).astype(np.int64)
            copied_sequence[i][mask_idx] = self.mask_id
        return copied_sequence, seq_lens


class Item_Reorder(torch.nn.Module):

    def __init__(self, beta=0.2) ->None:
        super().__init__()
        self.beta = beta

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        reordered_sequences = []
        for i in range(batch_size):
            seq = sequences[i]
            seq_len = seq_lens[i]
            sub_len = int(self.beta * seq_len)
            start_index = random.randint(a=0, b=seq_len - sub_len)
            reordered_index = list(range(sub_len))
            random.shuffle(reordered_index)
            sub_seq = seq[start_index:start_index + sub_len][reordered_index]
            reordered_sequences.append(torch.cat([seq[:start_index], sub_seq, seq[start_index + sub_len:]]))
        return torch.stack(reordered_sequences, dim=0), seq_lens


class Item_Random(torch.nn.Module):

    def __init__(self, mask_id, tao=0.2, gamma=0.7, beta=0.2) ->None:
        super().__init__()
        self.mask_id = mask_id
        self.augmentation_methods = [Item_Crop(tao=tao), Item_Mask(mask_id, gamma=gamma), Item_Reorder(beta=beta)]

    def forward(self, sequences, seq_lens):
        return self.augmentation_methods[random.randint(0, len(self.augmentation_methods) - 1)](sequences, seq_lens)


class Item_Substitute(torch.nn.Module):

    def __init__(self, item_similarity_model, substitute_rate=0.1) ->None:
        super().__init__()
        if isinstance(item_similarity_model, list):
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate

    def _ensemble_sim_models(self, top_1_one, top_1_two):
        indicator = top_1_one[1] >= top_1_two[1]
        top_1 = torch.zeros_like(top_1_one[0])
        top_1[indicator] = top_1_one[0][indicator]
        top_1[~indicator] = top_1_two[0][~indicator]
        return top_1

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        substituted_sequences = []
        for i in range(batch_size):
            seq = sequences[i]
            seq_len = seq_lens[i]
            sub_len = max(1, int(self.substitute_rate * seq_len))
            substitute_idx = np.random.choice(seq_len.item(), size=sub_len, replace=False)
            substituted_sequence = copy.deepcopy(seq)
            selected_items = substituted_sequence[substitute_idx]
            if self.ensemble:
                top_1_one = self.item_sim_model_1(selected_items, with_score=True)
                top_1_two = self.item_sim_model_2(selected_items, with_score=True)
                substitute_items = self._ensemble_sim_models(top_1_one, top_1_two)
                substituted_sequence[substitute_idx] = substitute_items
            else:
                substitute_items = self.item_similarity_model(selected_items, with_score=False)
                substituted_sequence[substitute_idx] = substitute_items
            substituted_sequences.append(substituted_sequence)
        return torch.stack(substituted_sequences, dim=0), seq_lens


class Item_Insert(torch.nn.Module):

    def __init__(self, item_similarity_model, insert_rate=0.4) ->None:
        super().__init__()
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate

    def _ensemble_sim_models(self, top_1_one, top_1_two):
        if top_1_one[1] >= top_1_two[1]:
            return top_1_one[0]
        else:
            return top_1_two[0]

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        inserted_sequences = []
        new_seq_lens = torch.zeros_like(seq_lens, device=seq_lens.device)
        for i in range(batch_size):
            seq = sequences[i]
            seq_len = seq_lens[i]
            sub_len = max(1, int(self.insert_rate * seq_len))
            insert_idx = np.random.choice(seq_len.item(), size=sub_len, replace=False)
            inserted_sequence = []
            new_seq_lens[i] = seq_len + sub_len
            for j in range(seq_len):
                if j in insert_idx:
                    if self.ensemble:
                        top_1_one = self.item_sim_model_1(seq[j].item(), with_score=True)
                        top_1_two = self.item_sim_model_2(seq[j].item(), with_score=True)
                        insert_item = self._ensemble_sim_models(top_1_one, top_1_two)
                        inserted_sequence.append(insert_item)
                    else:
                        insert_item = self.item_similarity_model(seq[j].item(), with_score=False)
                        inserted_sequence.append(insert_item)
                inserted_sequence.append(seq[j].item())
            inserted_sequences.append(torch.tensor(inserted_sequence, device=seq.device))
        return pad_sequence(inserted_sequences, batch_first=True), new_seq_lens


class Random_Augmentation(torch.nn.Module):

    def __init__(self, augment_threshold, short_seq_aug_methods: list, long_seq_aug_methods: list) ->None:
        super().__init__()
        self.augment_threshold = augment_threshold
        self.short_seq_aug_methods = short_seq_aug_methods
        self.long_seq_aug_methods = long_seq_aug_methods

    def forward(self, sequences, seq_lens):
        batch_size = sequences.size(0)
        new_seqs = []
        new_seq_lens = []
        for i in range(batch_size):
            seq = sequences[i].unsqueeze(0)
            seq_len = seq_lens[i].unsqueeze(0)
            if seq_len > self.augment_threshold:
                aug_method = random.choice(self.long_seq_aug_methods)
            else:
                aug_method = random.choice(self.short_seq_aug_methods)
            seq_, seq_len_ = aug_method(seq, seq_len)
            new_seqs.append(seq_.squeeze(0))
            new_seq_lens.append(seq_len_)
        new_seqs = pad_sequence(new_seqs, batch_first=True)
        new_seq_lens = torch.cat(new_seq_lens, dim=0)
        return new_seqs, new_seq_lens


class EdgeDropout(torch.nn.Module):
    """
    Out-place operation. 
    Dropout some edges in the graph in sparse COO or dgl format. It is used in GNN-based models.
    Parameters:
        dropout_prob(float): probability of a node to be zeroed.
    """

    def __init__(self, dropout_prob) ->None:
        super().__init__()
        self.keep_prob = 1.0 - dropout_prob
        self.edge_dropout_dgl = None

    def forward(self, X):
        """
        Returns:
            (torch.Tensor or dgl.DGLGraph): the graph after dropout in sparse COO or dgl.DGLGraph format.
        """
        if not self.training:
            return X
        if isinstance(X, torch.Tensor) and X.is_sparse and not X.is_sparse_csr:
            X = X.coalesce()
            random_tensor = torch.rand(X._nnz(), device=X.device) + self.keep_prob
            random_tensor = torch.floor(random_tensor).type(torch.bool)
            indices = X.indices()[:, random_tensor]
            values = X.values()[random_tensor] * (1.0 / self.keep_prob)
            return torch.sparse_coo_tensor(indices, values, X.shape, dtype=X.dtype)
        elif isinstance(X, dgl.DGLGraph):
            if self.edge_dropout_dgl == None:
                self.edge_dropout_dgl = dgl.DropEdge(p=1.0 - self.keep_prob)
            new_X = copy.deepcopy(X)
            new_X = self.edge_dropout_dgl(new_X)
            return new_X


class NodeDropout(torch.nn.Module):
    """
    Drop some nodes and the edges connected to them in the graph in sparse COO or dgl format. 
    It is a out-place operation.
    Parameters:
        dropout_prob(float): probability of a node to be droped.
    """

    def __init__(self, dropout_prob: float, num_users: int, num_items: int) ->None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dropout_prob = dropout_prob

    def forward(self, X):
        if not self.training:
            return X
        nodes_flag = torch.tensor([False] * (self.num_users + self.num_items), device=X.device)
        user_drop_indices = torch.randperm(self.num_users)[:int(self.num_users * self.dropout_prob)]
        item_drop_indices = torch.randperm(self.num_items)[:int(self.num_items * self.dropout_prob)]
        nodes_flag[user_drop_indices] = True
        nodes_flag[item_drop_indices + self.num_users] = True
        if isinstance(X, torch.Tensor) and X.is_sparse and not X.is_sparse_csr:
            nodes_flag = nodes_flag
            nodes_indices = torch.arange(self.num_users + self.num_items, device=X.device, dtype=torch.long)[~nodes_flag]
            diag = torch.sparse_coo_tensor(torch.stack([nodes_indices, nodes_indices]), torch.ones(len(nodes_indices), device=X.device, dtype=X.dtype), size=(self.num_users + self.num_items, self.num_users + self.num_items))
            new_X = torch.sparse.mm(X, diag)
            new_X = torch.sparse.mm(diag, new_X)
            return new_X
        elif 'dgl' in str(type(X)):

            def edges_with_droped_nodes(edges):
                src, dst, _ = edges.edges()
                return torch.logical_or(nodes_flag[src], nodes_flag[dst])
            droped_edges = X.filter_edges(edges_with_droped_nodes)
            new_X = copy.deepcopy(X)
            new_X.remove_edges(droped_edges)
            return new_X
        else:
            raise ValueError(f"NodeDropout doesn't support graph with type {type(X)}")


class SGLAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) ->None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        if self.config['aug_type'] == 'ED':
            self.augmentation = EdgeDropout(self.config['ssl_ratio'], self.num_users, self.num_items)
        elif self.config['aug_type'] == 'ND':
            self.augmentation = NodeDropout(self.config['ssl_ratio'], self.num_users, self.num_items)
        elif self.config['aug_type'] == 'RW':
            self.augmentation = EdgeDropout(self.config['ssl_ratio'], self.num_users, self.num_items)
        self.InfoNCELoss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='cosine', neg_type='all')

    def get_gnn_embeddings(self, user_emb, item_emb, adj_mat, gnn_net):
        adj_mat = adj_mat
        if self.config['aug_type'] in ['ED', 'ND']:
            adj_mat_aug = self.augmentation(adj_mat)
        elif self.config['aug_type'] == 'RW':
            adj_mat_aug = []
            for i in range(len(gnn_net.combiners)):
                adj_mat_aug.append(self.augmentation(adj_mat))
        embeddings = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        all_embeddings = gnn_net(adj_mat_aug, embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        return torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

    def forward(self, batch, user_emb: torch.nn.Embedding, item_emb: torch.nn.Embedding, adj_mat, gnn_net: torch.nn.Module):
        output_dict = {}
        user_all_vec1, item_all_vec1 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)
        user_all_vec2, item_all_vec2 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)
        user_cl_loss = self.InfoNCELoss_fn(user_all_vec1[batch[self.fuid]], user_all_vec2[batch[self.fuid]], all_reps=user_all_vec2[1:])
        item_cl_loss = self.InfoNCELoss_fn(item_all_vec1[batch[self.fiid]], item_all_vec2[batch[self.fiid]], all_reps=item_all_vec2[1:])
        output_dict['cl_loss'] = user_cl_loss + item_cl_loss
        return output_dict


class NCLAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) ->None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.hyper_layers = self.config['hyper_layers']
        self.InfoNCELoss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='cosine', neg_type='all')
        self.cluster = faiss.Kmeans(d=self.config['embed_dim'], k=self.config['num_clusters'], gpu=False)
        self.user_centroids, self.user_2cluster = None, None
        self.item_centroids, self.item_2cluster = None, None

    def forward(self, batch, all_embeddings_list: list):
        output_dict = {}
        center_embeddings = all_embeddings_list[0]
        context_embeddings = all_embeddings_list[self.hyper_layers * 2]
        user_center_embeddings, item_center_embeddings = torch.split(center_embeddings, [self.num_users, self.num_items], dim=0)
        user_context_embeddings, item_context_embeddings = torch.split(context_embeddings, [self.num_users, self.num_items], dim=0)
        user_structure_loss = self.InfoNCELoss_fn(user_context_embeddings[batch[self.fuid]], user_center_embeddings[batch[self.fuid]], all_reps=user_center_embeddings[1:])
        item_structure_loss = self.InfoNCELoss_fn(item_context_embeddings[batch[self.fiid]], item_center_embeddings[batch[self.fiid]], all_reps=item_center_embeddings[1:])
        output_dict['structure_cl_loss'] = user_structure_loss + self.config['alpha'] * item_structure_loss
        user2cluster = self.user_2cluster[batch[self.fuid]]
        user_proto_loss = self.InfoNCELoss_fn(user_center_embeddings[batch[self.fuid]], self.user_centroids[user2cluster], all_reps=self.user_centroids)
        item2cluster = self.item_2cluster[batch[self.fiid]]
        item_proto_loss = self.InfoNCELoss_fn(item_center_embeddings[batch[self.fiid]], self.item_centroids[item2cluster], all_reps=self.item_centroids)
        output_dict['semantic_cl_loss'] = user_proto_loss + self.config['alpha'] * item_proto_loss
        return output_dict

    @torch.no_grad()
    def e_step(self, user_emb: torch.nn.Embedding, item_emb: torch.nn.Embedding):
        user_embeddings = user_emb.weight.detach().cpu().numpy()
        item_embeddings = item_emb.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)
        self.user_centroids = self.user_centroids
        self.user_2cluster = self.user_2cluster
        self.item_centroids = self.item_centroids
        self.item_2cluster = self.item_2cluster

    @torch.no_grad()
    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        self.cluster.train(x[1:])
        cluster_cents = self.cluster.centroids
        _, I = self.cluster.index.search(x, 1)
        centroids = torch.from_numpy(cluster_cents)
        node2cluster = torch.LongTensor(I).squeeze(dim=-1)
        return centroids, node2cluster


class SimGCLAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) ->None:
        super().__init__()
        self.config = config
        self.fuid = train_data.fuid
        self.fiid = train_data.fiid
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        self.InfoNCELoss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='cosine', neg_type=self.config['cl_neg_type'])

    def get_gnn_embeddings(self, user_emb, item_emb, adj_mat, gnn_net):
        adj_mat = adj_mat
        embeddings = torch.cat([user_emb.weight, item_emb.weight], dim=0)
        all_embeddings = gnn_net(adj_mat, embeddings, perturbed=True)
        all_embeddings = torch.stack(all_embeddings, dim=-2)
        all_embeddings = torch.mean(all_embeddings, dim=-2, keepdim=False)
        return torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

    def forward(self, batch, user_emb: torch.nn.Embedding, item_emb: torch.nn.Embedding, adj_mat, gnn_net: torch.nn.Module):
        output_dict = {}
        device = user_emb.weight.device
        u_idx = torch.unique(batch[self.fuid])
        i_idx = torch.unique(batch[self.fiid])
        user_all_vec1, item_all_vec1 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)
        user_all_vec2, item_all_vec2 = self.get_gnn_embeddings(user_emb, item_emb, adj_mat, gnn_net)
        if self.config['cl_neg_type'] == 'all':
            user_cl_loss = self.InfoNCELoss_fn(user_all_vec1[u_idx], user_all_vec2[u_idx], all_reps=user_all_vec2[1:])
        else:
            user_cl_loss = self.InfoNCELoss_fn(user_all_vec1[u_idx], user_all_vec2[u_idx])
        if self.config['cl_neg_type'] == 'all':
            item_cl_loss = self.InfoNCELoss_fn(item_all_vec1[i_idx], item_all_vec2[i_idx], all_reps=item_all_vec2[1:])
        else:
            item_cl_loss = self.InfoNCELoss_fn(item_all_vec1[i_idx], item_all_vec2[i_idx])
        output_dict['cl_loss'] = user_cl_loss + item_cl_loss
        return output_dict


class CL4SRecAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) ->None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        if self.config['augment_type'] == 'item_crop':
            self.augmentation = Item_Crop(self.config['tao'])
        elif self.config['augment_type'] == 'item_mask':
            self.augmentation = Item_Mask(mask_id=train_data.num_items)
        elif self.config['augment_type'] == 'item_reorder':
            self.augmentation = Item_Reorder()
        elif self.config['augment_type'] == 'item_random':
            self.augmentation = Item_Random(mask_id=train_data.num_items)
        else:
            raise ValueError(f"augmentation type: '{self.config['augment_type']}' is invalided")
        self.InfoNCE_loss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='inner_product', neg_type='batch_both')

    def forward(self, batch, query_encoder: torch.nn.Module):
        output_dict = {}
        seq_augmented_i, seq_augmented_i_len = self.augmentation(batch['in_' + self.fiid], batch['seqlen'])
        seq_augmented_j, seq_augmented_j_len = self.augmentation(batch['in_' + self.fiid], batch['seqlen'])
        seq_augmented_i_out = query_encoder({('in_' + self.fiid): seq_augmented_i, 'seqlen': seq_augmented_i_len}, need_pooling=False)
        seq_augmented_i_out = recfn.seq_pooling_function(seq_augmented_i_out, seq_augmented_i_len, pooling_type='mean')
        seq_augmented_j_out = query_encoder({('in_' + self.fiid): seq_augmented_j, 'seqlen': seq_augmented_j_len}, need_pooling=False)
        seq_augmented_j_out = recfn.seq_pooling_function(seq_augmented_j_out, seq_augmented_j_len, pooling_type='mean')
        cl_loss = self.InfoNCE_loss_fn(seq_augmented_i_out, seq_augmented_j_out)
        output_dict['cl_loss'] = cl_loss
        return output_dict


class ICLRecAugmentation(torch.nn.Module):

    def __init__(self, config, train_data) ->None:
        super().__init__()
        self.config = config
        self.fiid = train_data.fiid
        if self.config['augment_type'] == 'item_crop':
            self.augmentation = Item_Crop()
        elif self.config['augment_type'] == 'item_mask':
            self.augmentation = Item_Mask(mask_id=train_data.num_items)
        elif self.config['augment_type'] == 'item_reorder':
            self.augmentation = Item_Reorder()
        elif self.config['augment_type'] == 'item_random':
            self.augmentation = Item_Random(mask_id=train_data.num_items)
        else:
            raise ValueError(f"augmentation type: '{self.config['augment_type']}' is invalided")
        self.InfoNCE_loss_fn = InfoNCELoss(temperature=self.config['temperature'], sim_method='inner_product', neg_type='batch_both')
        if self.config['intent_seq_representation_type'] == 'concat':
            self.cluster = faiss.Kmeans(d=self.config['embed_dim'] * self.config['max_seq_len'], k=self.config['num_intent_clusters'], gpu=False)
        else:
            self.cluster = faiss.Kmeans(d=self.config['embed_dim'], k=self.config['num_intent_clusters'], gpu=False)
        self.centroids = None

    @torch.no_grad()
    def train_kmeans(self, query_encoder, trainloader, device):
        kmeans_training_data = []
        for batch_idx, batch in enumerate(trainloader):
            batch = Recommender._to_device(batch, device)
            seq_out = query_encoder(batch, need_pooling=False)
            seq_out = recfn.seq_pooling_function(seq_out, batch['seqlen'], pooling_type=self.config['intent_seq_representation_type'])
            kmeans_training_data.append(seq_out)
        kmeans_training_data = torch.cat(kmeans_training_data, dim=0)
        self.cluster.train(kmeans_training_data.cpu().numpy())
        self.centroids = torch.from_numpy(self.cluster.centroids)

    def forward(self, batch, seq_out: torch.Tensor, query_encoder: torch.nn.Module):
        output_dict = {}
        seq_augmented_i, seq_augmented_i_len = self.augmentation(batch['in_' + self.fiid], batch['seqlen'])
        seq_augmented_j, seq_augmented_j_len = self.augmentation(batch['in_' + self.fiid], batch['seqlen'])
        seq_augmented_i_out = query_encoder({('in_' + self.fiid): seq_augmented_i, 'seqlen': seq_augmented_i_len}, need_pooling=False)
        seq_augmented_j_out = query_encoder({('in_' + self.fiid): seq_augmented_j, 'seqlen': seq_augmented_j_len}, need_pooling=False)
        instance_seq_i_out = recfn.seq_pooling_function(seq_augmented_i_out, seq_augmented_i_len, pooling_type=self.config['instance_seq_representation_type'])
        instance_seq_j_out = recfn.seq_pooling_function(seq_augmented_j_out, seq_augmented_j_len, pooling_type=self.config['instance_seq_representation_type'])
        instance_loss = self.InfoNCE_loss_fn(instance_seq_i_out, instance_seq_j_out)
        instance_loss_rev = self.InfoNCE_loss_fn(instance_seq_j_out, instance_seq_i_out)
        seq_out = recfn.seq_pooling_function(seq_out, batch['seqlen'], pooling_type=self.config['intent_seq_representation_type'])
        _, intent_ids = self.cluster.index.search(seq_out.cpu().detach().numpy(), 1)
        seq2intents = self.centroids[intent_ids.squeeze(-1)]
        intent_seq_i_out = recfn.seq_pooling_function(seq_augmented_i_out, seq_augmented_i_len, pooling_type=self.config['intent_seq_representation_type'])
        intent_seq_j_out = recfn.seq_pooling_function(seq_augmented_j_out, seq_augmented_j_len, pooling_type=self.config['intent_seq_representation_type'])
        intent_ids = torch.from_numpy(intent_ids.squeeze(-1))
        intent_loss_i = self.InfoNCE_loss_fn(intent_seq_i_out, seq2intents, instance_labels=intent_ids)
        intent_loss_j = self.InfoNCE_loss_fn(intent_seq_j_out, seq2intents, instance_labels=intent_ids)
        output_dict['instance_cl_loss'] = 0.5 * (instance_loss + instance_loss_rev)
        output_dict['intent_cl_loss'] = 0.5 * (intent_loss_i + intent_loss_j)
        return output_dict


class OnlineItemSimilarity(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.item_embeddings = None

    def update_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings.weight[1:].detach().clone()
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -100.0, 100.0
        for item_idx in range(self.item_embeddings.size(0)):
            item_vector = self.item_embeddings[torch.tensor(item_idx)].view(-1, 1)
            item_similarity = torch.mm(self.item_embeddings, item_vector).view(-1)
            max_score = max(torch.max(item_similarity), max_score)
            min_score = min(torch.min(item_similarity), min_score)
        return max_score, min_score

    def forward(self, item_idx: Union[torch.Tensor, int], top_k=1, with_score=False):
        """
        Args: 
        item_idx (torch.Tensor or int)
        top_k (int)
        with_score (bool)
        Return:
        indices (torch.Tensor or int): [batch_size] or int 
        values (torch.Tensor or float): [batch_size] or float 
        """
        item_idx = item_idx - 1
        assert top_k == 1, 'only support top 1'
        if type(item_idx) == int:
            item_vector = self.item_embeddings[item_idx]
            item_similarity = torch.mv(self.item_embeddings, item_vector)
            item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
            item_similarity[item_idx] = -float('inf')
            scores, indices = item_similarity.topk(top_k)
            indices = indices + 1
            if with_score:
                return indices[0].item(), scores[0].item()
            else:
                return indices[0].item()
        else:
            item_vector = self.item_embeddings[item_idx]
            item_similarity = torch.mm(item_vector, self.item_embeddings.T)
            item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
            item_similarity[torch.arange(item_idx.size(0), device=item_idx.device), item_idx] = -float('inf')
            scores, indices = item_similarity.topk(top_k)
            indices = indices + 1
            if with_score:
                return indices.squeeze(-1), scores.squeeze(-1)
            else:
                return indices.squeeze(-1)


class Combiner(nn.Module):
    """
    The base class for combiner in GNN. 

    Args:
        input_size(int): size of input representations
        output_size(int): size of output representations
        dropout(float): the probability to be set in dropout module.
        act(torch.nn.Module): the activation function.
    """

    def __init__(self, input_size: Optional[float], output_size: Optional[float], dropout: Optional[float]=0.0, act=nn.ReLU()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mess_out = dropout
        self.act = act
        if dropout != None:
            self.dropout = nn.Dropout(dropout)


class GCNCombiner(Combiner):

    def __init__(self, input_size: int, output_size: int, dropout: float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Applies nonlinear transformation on the summation of two representation vectors
        """
        embeddings = self.act(self.linear(embeddings + side_embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings


class GraphSageCombiner(Combiner):

    def __init__(self, input_size: int, output_size: int, dropout: float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size * 2, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Concatenates the two representation vectors and the applies nonlinear transformation
        """
        embeddings = self.act(self.linear(torch.cat([embeddings, side_embeddings], dim=-1)))
        embeddings = self.dropout(embeddings)
        return embeddings


class NeighborCombiner(Combiner):

    def __init__(self, input_size: int, output_size: int, dropout: float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Applies nonlinear transformation on neighborhood representation.
        """
        embeddings = self.act(self.linear(side_embeddings))
        embeddings = self.dropout(embeddings)
        return embeddings


class BiCombiner(Combiner):

    def __init__(self, input_size: int, output_size: int, dropout: float=0.0, act=nn.ReLU()):
        super().__init__(input_size, output_size, dropout, act)
        self.linear_sum = nn.Linear(input_size, output_size)
        self.linear_product = nn.Linear(input_size, output_size)

    def forward(self, embeddings, side_embeddings):
        """
        Applies the following transformation on two representations.
        .. math::
            	ext{output} = act(W_{1}(V + V_{side})+b) + act(W_{2}(V \\odot V_{side})+b)
        """
        sum_embeddings = self.act(self.linear_sum(embeddings + side_embeddings))
        bi_embeddings = self.act(self.linear_product(embeddings * side_embeddings))
        embeddings = sum_embeddings + bi_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class LightGCNCombiner(Combiner):

    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, None, None)

    def forward(self, embeddings, side_embeddings):
        return side_embeddings


class GraphItemEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.item_embeddings = None

    def forward(self, batch_data):
        return self.item_embeddings[batch_data]


class GraphUserEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.user_embeddings = None

    def forward(self, batch_data):
        return self.user_embeddings[batch_data]


class AIGRU(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int=1, bias: bool=True, batch_first: bool=True, dropout: float=0.0, bidirectional: bool=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, input, weight, h_0=None):
        assert input.shape[:-1] == weight.shape, '`input` must have the same shape with `weight` at dimension 0and 1, but get {} and {}'.format(input.shape, weight.shape)
        weighted_input = input * weight.unsqueeze(2)
        if h_0 is not None:
            gru_out, _ = self.gru(weighted_input, h_0)
        else:
            gru_out, _ = self.gru(weighted_input)
        return gru_out

    def extra_repr(self) ->str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


class AGRUCell(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gru_cell = torch.nn.GRUCell(self.input_size, self.hidden_size, self.bias)

    def forward(self, input, hidden, weight):
        weight = weight.view(-1, 1)
        hidden_o = self.gru_cell(input, hidden)
        hidden = (1 - weight) * hidden + weight * hidden_o
        return hidden


class AUGRUCell(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.w_ir = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_hr = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_iz = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_hz = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_in = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.w_hn = torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.sigma = torch.nn.Sigmoid()

    def forward(self, input, hidden, weight):
        weight = weight.view(-1, 1)
        r = self.sigma(self.w_ir(input) + self.w_hr(hidden))
        z = self.sigma(self.w_iz(input) + self.w_hz(hidden))
        z = weight * z
        n = torch.tanh(self.w_in(input) + r * self.w_hn(hidden))
        hidden = (1 - z) * n + z * hidden
        return hidden


class AGRU(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bias: bool=True, batch_first: bool=True, dropout: float=0.0, bidirectional: bool=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.cell = self._set_gru_cell(self.input_size, self.hidden_size, self.bias)

    def _set_gru_cell(self, input_size, hidden_size, bias):
        return AGRUCell(input_size, hidden_size, bias)

    def forward(self, input, weight, h_0=None):
        if self.batch_first:
            input = input.contiguous().transpose(0, 1)
        L = input.size(0)
        if h_0 is None:
            num_directions = 2 if self.bidirectional else 1
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            hx = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = h_0
        output = [None] * L
        for i in range(L):
            input_ = input[i]
            weight_ = weight[i]
            hx = self.cell(input_, hx, weight_)
            output[i] = hx
        if self.batch_first:
            output = torch.stack(output, dim=1)
        else:
            output = torch.stack(output, dim=0)
        return output, hx

    def extra_repr(self) ->str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


class AUGRU(AGRU):

    def _set_gru_cell(self, input_size, hidden_size, bias):
        return AUGRUCell(input_size, hidden_size, bias)


class CrossCompressUnit(torch.nn.Module):
    """
    Cross & Compress unit.
    Performs feature interaction as below:
        .. math::
            C_{l}=v_{l}e_{l}^	op=egin{bmatrix}
            v_{l}^{(1)}e_{l}^{(1)} & ...  & v_{l}^{(1)}e_{l}^{(d)} \\
            ... &  & ... \\
            v_{l}^{(d)}e_{l}^{(1)} & ... & v_{l}^{(d)}e_{l}^{(d)}
            \\end{bmatrix}
            \\
            v_{l+1}=C_{l}W_{l}^{VV}+C_{l}^	op W_{l}^{EV}+b_{l}^{V}
            \\
            e_{l+1}=C_{l}W_{l}^{VE}+C_{l}^	op W_{l}^{EE}+b_{l}^{E}

    Parameters:
        embed_dim(int): dimensions of embeddings.
        weight_vv(torch.nn.Linear): transformation weights.
        weight_ev(torch.nn.Linear): transformation weights.
        weight_ve(torch.nn.Linear): transformation weights.
        weight_ee(torch.nn.Linear): transformation weights.
        bias_v(Parameter): bias on v.
        bias_e(Parameter): bias on e.

    Returns:
        v_output(torch.Tensor): the first embeddings after feature interaction.
        e_output(torch.Tensor): the second embeddings after feature interaction.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight_vv = torch.nn.Linear(self.embed_dim, 1, False)
        self.weight_ev = torch.nn.Linear(self.embed_dim, 1, False)
        self.weight_ve = torch.nn.Linear(self.embed_dim, 1, False)
        self.weight_ee = torch.nn.Linear(self.embed_dim, 1, False)
        self.bias_v = Parameter(data=torch.zeros(self.embed_dim), requires_grad=True)
        self.bias_e = Parameter(data=torch.zeros(self.embed_dim), requires_grad=True)

    def forward(self, inputs):
        v_input = inputs[0].unsqueeze(-1)
        e_input = inputs[1].unsqueeze(-2)
        c_matrix = torch.matmul(v_input, e_input)
        c_matrix_transpose = c_matrix.transpose(-1, -2)
        v_output = (self.weight_vv(c_matrix) + self.weight_ev(c_matrix_transpose)).squeeze(-1)
        v_output = v_output + self.bias_v
        e_output = (self.weight_ve(c_matrix) + self.weight_ee(c_matrix_transpose)).squeeze(-1)
        e_output = e_output + self.bias_e
        return v_output, e_output


class FeatInterLayers(torch.nn.Module):
    """
    Feature interaction layers with varied feature interaction units.

    Args:
        dim(int): the dimensions of the feature.
        num_layers(int): the number of stacked units in the layers.
        unit(torch.nn.Module): the feature interaction used in the layer.

    Examples:
    >>> featInter = FeatInterLayers(64, 2, CrossCompressUnit)
    >>> featInter.model
    Sequential(
        (unit[0]): CrossCompressUnit(
            (weight_vv): Linear(in_features=64, out_features=1, bias=False)
            (weight_ev): Linear(in_features=64, out_features=1, bias=False)
            (weight_ve): Linear(in_features=64, out_features=1, bias=False)
            (weight_ee): Linear(in_features=64, out_features=1, bias=False)
        )
        (unit[1]): CrossCompressUnit(
            (weight_vv): Linear(in_features=64, out_features=1, bias=False)
            (weight_ev): Linear(in_features=64, out_features=1, bias=False)
            (weight_ve): Linear(in_features=64, out_features=1, bias=False)
            (weight_ee): Linear(in_features=64, out_features=1, bias=False)
        )
    )
    """

    def __init__(self, dim, num_units, unit) ->None:
        super().__init__()
        self.model = torch.nn.Sequential()
        for id in range(num_units):
            self.model.add_module(f'unit[{id}]', unit(dim))

    def forward(self, v_input, e_input):
        return self.model((v_input, e_input))


class GRULayer(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_layer=1, bias=False, batch_first=True, bidirectional=False, return_hidden=False) ->None:
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=num_layer, bias=bias, batch_first=batch_first, bidirectional=bidirectional)
        self.return_hidden = return_hidden

    def forward(self, input):
        out, hidden = self.gru(input)
        if self.return_hidden:
            return out, hidden
        else:
            return out


class LambdaLayer(torch.nn.Module):

    def __init__(self, lambda_func) ->None:
        super().__init__()
        self.lambda_func = lambda_func

    def forward(self, *args):
        if len(args) == 1:
            return self.lambda_func(args[0])
        else:
            return self.lambda_func(args)


class HStackLayer(torch.nn.Sequential):

    def forward(self, *input):
        output = []
        assert len(input) == 1 or len(input) == len(self.module_list)
        for i, module in enumerate(self):
            if len(input) == 1:
                output.append(module(input[0]))
            else:
                output.append(module(input[i]))
        return tuple(output)


class VStackLayer(torch.nn.Sequential):

    def forward(self, input):
        for module in self:
            if isinstance(input, Tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class InnerProductScorer(torch.nn.Module):

    def forward(self, query, items):
        if query.size(0) == items.size(0):
            if query.dim() < items.dim():
                output = torch.matmul(items, query.view(*query.shape, 1))
                output = output.view(output.shape[:-1])
            else:
                output = torch.sum(query * items, dim=-1)
        else:
            output = torch.matmul(query, items.T)
        return output


class CosineScorer(InnerProductScorer):

    def forward(self, query, items):
        output = super().forward(query, items)
        output /= torch.norm(items, dim=-1)
        output /= torch.norm(query, dim=-1, keepdim=query.dim() != items.dim() or query.size(0) != items.size(0))
        return output


class EuclideanScorer(InnerProductScorer):

    def forward(self, query, items):
        output = -2 * super().forward(query, items)
        output += torch.sum(torch.square(items), dim=-1)
        output += torch.sum(torch.square(query), dim=-1, keepdim=query.dim() != items.dim() or query.size(0) != items.size(0))
        return -output


class MLPScorer(InnerProductScorer):

    def __init__(self, transform):
        super().__init__()
        self.trans = transform

    def forward(self, query: torch.Tensor, items: torch.Tensor):
        if query.size(0) == items.size(0):
            if query.dim() < items.dim():
                input = torch.cat((query.unsqueeze(-2).expand_as(items), items), dim=-1)
            else:
                input = torch.cat((query, items), dim=-1)
        else:
            query = query.unsqueeze(1).repeat(1, items.size(0), 1)
            items = items.expand(query.size(0), -1, -1)
            input = torch.cat((query, items), dim=-1)
        return self.trans(input).squeeze(-1)


class NormScorer(InnerProductScorer):

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, query, items):
        if query.dim() < items.dim() or query.size(0) != items.size(0):
            query.unsqueeze_(-2)
        output = torch.norm(query - items, p=self.p, dim=-1)
        return -output


class GMFScorer(InnerProductScorer):

    def __init__(self, emb_dim, bias=False, activation='relu') ->None:
        super().__init__()
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, 1, bias=bias)
        self.activation = module.get_act(activation)

    def forward(self, query, key):
        assert query.dim() <= key.dim(), 'query dim must be smaller than or euqal to key dim'
        if query.dim() < key.dim():
            query = query.unsqueeze(1)
        elif query.size(0) != key.size(0):
            query = query.unsqueeze(1).repeat(1, key.size(0), 1)
            key = key.unsqueeze(0).repeat(query.size(0), 1, 1)
        h = query * key
        return self.activation(self.W(h)).squeeze(-1)


class FusionMFMLPScorer(InnerProductScorer):

    def __init__(self, emb_dim, hidden_size, mlp, bias=False, activation='relu') ->None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.bias = bias
        self.W = torch.nn.Linear(self.emb_dim + self.hidden_size, 1, bias=False)
        self.activation = module.get_act(activation)
        self.mlp = mlp

    def forward(self, query, key):
        assert query.dim() <= key.dim(), 'query dim must be smaller than or euqal to key dim'
        if query.dim() < key.dim():
            query = query.unsqueeze(1).repeat(1, key.shape[1], 1)
        elif query.size(0) != key.size(0):
            query = query.unsqueeze(1).repeat(1, key.size(0), 1)
            key = key.unsqueeze(0).repeat(query.size(0), 1, 1)
        h_mf = query * key
        h_mlp = self.mlp(torch.cat([query, key], dim=-1))
        h = self.activation(self.W(torch.cat([h_mf, h_mlp], dim=-1)).squeeze(-1))
        return h


class CaserQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, fuid, num_users, num_items, embed_dim, max_seq_len, n_v, n_h, dropout=0.2) ->None:
        super().__init__()
        self.fiid = fiid
        self.fuid = fuid
        self.max_seq_len = max_seq_len
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.vertical_filter = torch.nn.Conv2d(in_channels=1, out_channels=n_v, kernel_size=(self.max_seq_len, 1))
        height = range(1, max_seq_len + 1)
        self.horizontal_filter = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=1, out_channels=n_h, kernel_size=(h, embed_dim)) for h in height])
        self.fc = torch.nn.Linear(n_v * embed_dim + n_h * max_seq_len, embed_dim, bias=True)

    def forward(self, batch):
        P_u = self.user_embedding(batch[self.fuid])
        item_seq = batch['in_' + self.fiid]
        item_seq = torch.nn.functional.pad(item_seq, (0, self.max_seq_len - item_seq.size(1)))
        E_ut = self.item_embedding(item_seq)
        E_ut_ = E_ut.view(-1, 1, *E_ut.shape[1:])
        o_v = self.vertical_filter(E_ut_)
        o_v = o_v.reshape(o_v.size(0), -1)
        o_h = []
        for i in range(E_ut.size(1)):
            conv_out = torch.relu(self.horizontal_filter[i](E_ut_).squeeze(3))
            pool_out = torch.nn.functional.max_pool1d(conv_out, conv_out.size(2))
            o_h.append(pool_out.squeeze(2))
        o_h = torch.cat(o_h, dim=1)
        o = torch.cat((o_v, o_h), dim=1)
        o = self.dropout(o)
        z = torch.relu(self.fc(o))
        return torch.cat((z, P_u), dim=1)


class HGNQueryEncoder(torch.nn.Module):

    def __init__(self, fuid, fiid, num_users, embed_dim, max_seq_len, item_encoder, pooling_type='mean') ->None:
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.pooling_type = pooling_type
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.W_g_1 = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_g_2 = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.b_g = torch.nn.Parameter(torch.empty(embed_dim), requires_grad=True)
        self.w_g_3 = torch.nn.Linear(embed_dim, 1, bias=False)
        self.W_g_4 = torch.nn.Linear(embed_dim, max_seq_len)

    def forward(self, batch):
        U = self.user_embedding(batch[self.fuid])
        S = self.item_encoder(batch['in_' + self.fiid])
        S_F = S * torch.sigmoid(self.W_g_1(S) + self.W_g_2(U).view(U.size(0), 1, -1) + self.b_g)
        weight = torch.sigmoid(self.w_g_3(S_F) + (U @ self.W_g_4.weight[:S.size(1)].T).view(U.size(0), -1, 1))
        S_I = S_F * weight
        if self.pooling_type == 'mean':
            s = S_I.sum(1) / weight.sum(1)
        elif self.pooling_type == 'max':
            s = torch.max(S_I, dim=1).values
        else:
            raise ValueError('`pooling_type` only support `avg` and `max`')
        query = U + s + S.sum(1)
        return query


class NARMQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, embed_dim, hidden_size, layer_num, dropout_rate: List, item_encoder=None) ->None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.gru_layer = torch.nn.Sequential(self.item_encoder, torch.nn.Dropout(dropout_rate[0]), module.GRULayer(input_dim=embed_dim, output_dim=hidden_size, num_layer=layer_num))
        self.gather_layer = module.SeqPoolingLayer(pooling_type='last')
        self.attn_layer = module.AttentionLayer(q_dim=hidden_size, mlp_layers=[hidden_size], bias=False)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(dropout_rate[1]), torch.nn.Linear(hidden_size * 2, embed_dim, bias=False))

    def forward(self, batch):
        gru_vec = self.gru_layer(batch['in_' + self.fiid])
        c_global = h_t = self.gather_layer(gru_vec, batch['seqlen'])
        c_local = self.attn_layer(query=h_t.unsqueeze(1), key=gru_vec, value=gru_vec, key_padding_mask=batch['in_' + self.fiid] == 0, need_weight=False).squeeze(1)
        c = torch.cat((c_global, c_local), dim=1)
        query = self.fc(c)
        return query


class STAMPQueryEncoder(torch.nn.Module):

    def __init__(self, fiid, embed_dim, item_encoder) ->None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.gather_layer = module.SeqPoolingLayer(pooling_type='last')
        self.attention_layer = module.AttentionLayer(q_dim=2 * embed_dim, k_dim=embed_dim, mlp_layers=[embed_dim])
        self.mlpA = module.MLPModule([embed_dim, embed_dim], torch.nn.Tanh())
        self.mlpB = module.MLPModule([embed_dim, embed_dim], torch.nn.Tanh())

    def forward(self, batch):
        user_hist = batch['in_' + self.fiid]
        seq_emb = self.item_encoder(user_hist)
        m_t = self.gather_layer(seq_emb, batch['seqlen'])
        m_s = seq_emb.sum(dim=1) / batch['seqlen'].unsqueeze(1).float()
        query = torch.cat((m_t, m_s), dim=1)
        m_a = self.attention_layer(query.unsqueeze(1), seq_emb, seq_emb, key_padding_mask=user_hist == 0).squeeze(1)
        h_s = self.mlpA(m_a)
        h_t = self.mlpB(m_t)
        return h_s * h_t


class TransRecQueryEncoder(torch.nn.Module):

    def __init__(self, fuid, fiid, num_users, embed_dim, item_encoder):
        super().__init__()
        self.fuid = fuid
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim, 0)
        self.global_user_emb = torch.nn.Parameter(torch.zeros(embed_dim))

    def forward(self, batch):
        user_hist = batch['in_' + self.fiid]
        seq_len = batch['seqlen'] - 1
        local_user_emb = self.user_embedding(batch[self.fuid])
        user_emb = local_user_emb + self.global_user_emb.expand_as(local_user_emb)
        last_item_id = torch.gather(user_hist, dim=-1, index=seq_len.unsqueeze(1))
        last_item_emb = self.item_encoder(last_item_id).squeeze(1)
        query = user_emb + last_item_emb
        return query


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionLayer,
     lambda: ([], {'q_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BCEWithLogitLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BPRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BiCombiner,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BinaryCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CCLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CIN,
     lambda: ([], {'embed_dim': 4, 'num_features': 4, 'cin_layer_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CosineScorer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CrossCompressUnit,
     lambda: ([], {'embed_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossNetwork,
     lambda: ([], {'embed_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseEmbedding,
     lambda: ([], {'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dice,
     lambda: ([], {'num_parameters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EdgeDropout,
     lambda: ([], {'dropout_prob': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EuclideanScorer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FMLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FullScoreLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCNCombiner,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GMFScorer,
     lambda: ([], {'emb_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GRULayer,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GraphSageCombiner,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (HingeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (InfoNCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (InnerProductScorer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LightGCNCombiner,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPModule,
     lambda: ([], {'mlp_layers': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPScorer,
     lambda: ([], {'transform': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NeighborCombiner,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NodeDropout,
     lambda: ([], {'dropout_prob': 0.5, 'num_users': 4, 'num_items': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormScorer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PairwiseLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SampledSoftmaxLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftmaxLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SquareLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Top1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UniformSampler,
     lambda: ([], {'num_items': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 4], {}),
     False),
    (VStackLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedBPRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (WeightedBinaryCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ustcml_RecStudio(_paritybench_base):
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

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

