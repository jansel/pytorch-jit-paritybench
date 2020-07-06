import sys
_module = sys.modules[__name__]
del sys
data = _module
evaluation = _module
loss = _module
model = _module
projection = _module
transD_Bernoulli_pytorch = _module
transD_pytorch = _module
transE_Bernoulli_pytorch = _module
transE_pytorch = _module
transH_Bernoulli_pytorch = _module
transH_pytorch = _module
transR_Bernoulli_pytorch = _module
transR_pytorch = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import time


import random


import math


from itertools import groupby


import torch


import torch.autograd as autograd


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from sklearn.metrics.pairwise import pairwise_distances


class marginLoss(nn.Module):

    def __init__(self):
        super(marginLoss, self).__init__()

    def forward(self, pos, neg, margin):
        zero_tensor = floatTensor(pos.size())
        zero_tensor.zero_()
        zero_tensor = autograd.Variable(zero_tensor)
        return torch.sum(torch.max(pos - neg + margin, zero_tensor))


class TransEModel(nn.Module):

    def __init__(self, config):
        super(TransEModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        ent_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_weight = floatTensor(self.relation_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg


def projection_transH_pytorch(original, norm):
    return original - torch.sum(original * norm, dim=1, keepdim=True) * norm


class TransHModel(nn.Module):

    def __init__(self, config):
        super(TransHModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        ent_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_weight = floatTensor(self.relation_total, self.embedding_size)
        norm_weight = floatTensor(self.relation_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(norm_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.norm_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)
        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb
        self.norm_embeddings.weight.data = normalize_norm_emb

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_norm = self.norm_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_norm = self.norm_embeddings(neg_r)
        pos_h_e = projection_transH_pytorch(pos_h_e, pos_norm)
        pos_t_e = projection_transH_pytorch(pos_t_e, pos_norm)
        neg_h_e = projection_transH_pytorch(neg_h_e, neg_norm)
        neg_t_e = projection_transH_pytorch(neg_t_e, neg_norm)
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg


def projection_transR_pytorch(original, proj_matrix):
    ent_embedding_size = original.shape[1]
    rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
    original = original.view(-1, ent_embedding_size, 1)
    proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
    return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)


class TransRModel(nn.Module):

    def __init__(self, config):
        super(TransRModel, self).__init__()
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.ent_embedding_size = config.ent_embedding_size
        self.rel_embedding_size = config.rel_embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        ent_weight = floatTensor(self.entity_total, self.ent_embedding_size)
        rel_weight = floatTensor(self.relation_total, self.rel_embedding_size)
        proj_weight = floatTensor(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(proj_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.ent_embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size)
        self.proj_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)
        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_proj = self.proj_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_proj = self.proj_embeddings(neg_r)
        pos_h_e = projection_transR_pytorch(pos_h_e, pos_proj)
        pos_t_e = projection_transR_pytorch(pos_t_e, pos_proj)
        neg_h_e = projection_transR_pytorch(neg_h_e, neg_proj)
        neg_t_e = projection_transR_pytorch(neg_t_e, neg_proj)
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e


class TransRPretrainModel(nn.Module):

    def __init__(self, config):
        super(TransRPretrainModel, self).__init__()
        self.dataset = config.dataset
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.ent_embedding_size = config.ent_embedding_size
        self.rel_embedding_size = config.rel_embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        with open('./transE_%s_%s_best.pkl' % (config.dataset, str(config.ent_embedding_size)), 'rb') as fr:
            ent_embeddings_list = pickle.load(fr)
            rel_embeddings_list = pickle.load(fr)
        ent_weight = floatTensor(ent_embeddings_list)
        rel_weight = floatTensor(rel_embeddings_list)
        proj_weight = floatTensor(self.rel_embedding_size, self.ent_embedding_size)
        nn.init.eye(proj_weight)
        proj_weight = proj_weight.view(-1).expand(self.relation_total, -1)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.ent_embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size)
        self.proj_embeddings = nn.Embedding(self.relation_total, self.rel_embedding_size * self.ent_embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_proj = self.proj_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_proj = self.proj_embeddings(neg_r)
        pos_h_e = projection_transR_pytorch(pos_h_e, pos_proj)
        pos_t_e = projection_transR_pytorch(pos_t_e, pos_proj)
        neg_h_e = projection_transR_pytorch(neg_h_e, neg_proj)
        neg_t_e = projection_transR_pytorch(neg_t_e, neg_proj)
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e


def projection_transD_pytorch_samesize(entity_embedding, entity_projection, relation_projection):
    return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=1, keepdim=True) * relation_projection


class TransDPretrainModelSameSize(nn.Module):

    def __init__(self, config):
        super(TransDPretrainModelSameSize, self).__init__()
        self.dataset = config.dataset
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size
        with open('./transE_%s_%s_best.pkl' % (config.dataset, str(config.embedding_size)), 'rb') as fr:
            ent_embeddings_list = pickle.load(fr)
            rel_embeddings_list = pickle.load(fr)
        ent_weight = floatTensor(ent_embeddings_list)
        rel_weight = floatTensor(rel_embeddings_list)
        ent_proj_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_proj_weight = floatTensor(self.relation_total, self.embedding_size)
        ent_proj_weight.zero_()
        rel_proj_weight.zero_()
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_proj_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_proj_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.ent_proj_embeddings.weight = nn.Parameter(ent_proj_weight)
        self.rel_proj_embeddings.weight = nn.Parameter(rel_proj_weight)

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_h_proj = self.ent_proj_embeddings(pos_h)
        pos_t_proj = self.ent_proj_embeddings(pos_t)
        pos_r_proj = self.rel_proj_embeddings(pos_r)
        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_h_proj = self.ent_proj_embeddings(neg_h)
        neg_t_proj = self.ent_proj_embeddings(neg_t)
        neg_r_proj = self.rel_proj_embeddings(neg_r)
        pos_h_e = projection_transD_pytorch_samesize(pos_h_e, pos_h_proj, pos_r_proj)
        pos_t_e = projection_transD_pytorch_samesize(pos_t_e, pos_t_proj, pos_r_proj)
        neg_h_e = projection_transD_pytorch_samesize(neg_h_e, neg_h_proj, neg_r_proj)
        neg_t_e = projection_transD_pytorch_samesize(neg_t_e, neg_t_proj, neg_r_proj)
        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e

