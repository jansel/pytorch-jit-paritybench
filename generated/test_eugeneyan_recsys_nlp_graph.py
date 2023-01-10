import sys
_module = sys.modules[__name__]
del sys
src = _module
config = _module
ml = _module
data_loader = _module
data_loader_edges = _module
data_loader_with_meta = _module
mf = _module
mf_bias = _module
mf_bias_continuous = _module
mf_continuous = _module
skipgram = _module
skipgram_with_meta = _module
skipgram_with_meta_weighted = _module
train_gensim_embedding = _module
train_node2vec_embeddings = _module
train_torch_embedding = _module
train_torch_embedding_with_meta = _module
train_torch_mf = _module
train_torch_mf_bias = _module
train_torch_mf_bias_continuous_edges = _module
train_torch_mf_bias_edges = _module
train_torch_mf_bias_edges_parallel = _module
train_torch_mf_continuous_edges = _module
train_torch_mf_edges = _module
parse = _module
parse_json = _module
prep = _module
prep_edges = _module
prep_graph_samples = _module
prep_meta = _module
prep_node_relationship = _module
train_val_split = _module
utils = _module
io_utils = _module
logger = _module
viz = _module
plot_results = _module
prep_results = _module

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


import itertools


from collections import Counter


from typing import Dict


from typing import List


from typing import Tuple


import numpy as np


import pandas as pd


import torch


from torch.utils.data import Dataset


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


from sklearn.metrics import roc_auc_score


from torch import optim


from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def regularize_l2(array):
    loss = torch.sum(array ** 2.0)
    return loss


class MF(nn.Module):

    def __init__(self, emb_size, emb_dim, c_vector=1e-06):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.c_vector = c_vector
        self.embedding = nn.Embedding(emb_size, emb_dim)
        self.sig = nn.Sigmoid()
        self.bce = nn.BCELoss()
        logger.info('Model initialized: {}'.format(self))

    def forward(self, product1, product2):
        emb_product1 = self.embedding(product1)
        emb_product2 = self.embedding(product2)
        interaction = self.sig(torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float))
        return interaction

    def loss(self, pred, label):
        mf_loss = self.bce(pred, label)
        product_prior = regularize_l2(self.embedding.weight) * self.c_vector
        loss_total = mf_loss + product_prior
        return loss_total


class MFBias(nn.Module):

    def __init__(self, emb_size, emb_dim, c_vector=1e-06, c_bias=1e-06):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.c_vector = c_vector
        self.c_bias = c_bias
        self.product_embedding = nn.Embedding(emb_size, emb_dim)
        self.sig = nn.Sigmoid()
        self.product_bias = nn.Embedding(emb_size, 1)
        self.bias = nn.Parameter(torch.ones(1))
        self.bce = nn.BCELoss()
        logger.info('Model initialized: {}'.format(self))

    def forward(self, product1, product2):
        emb_product1 = self.product_embedding(product1)
        emb_product2 = self.product_embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)
        bias_product1 = self.product_bias(product1).squeeze()
        bias_product2 = self.product_bias(product2).squeeze()
        biases = self.bias + bias_product1 + bias_product2
        prediction = self.sig(interaction + biases)
        return prediction

    def loss(self, pred, label):
        mf_loss = self.bce(pred, label)
        product_prior = regularize_l2(self.product_embedding.weight) * self.c_vector
        product_bias_prior = regularize_l2(self.product_bias.weight) * self.c_bias
        loss_total = mf_loss + product_prior + product_bias_prior
        return loss_total


class MFBiasContinuous(nn.Module):

    def __init__(self, emb_size, emb_dim, c_vector=1e-06, c_bias=1e-06):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.c_vector = c_vector
        self.c_bias = c_bias
        self.product_embedding = nn.Embedding(emb_size, emb_dim)
        self.sig = nn.Sigmoid()
        self.product_bias = nn.Embedding(emb_size, 1)
        self.bias = nn.Parameter(torch.ones(1))
        self.mse = nn.MSELoss()
        logger.info('Model initialized: {}'.format(self))

    def forward(self, product1, product2):
        emb_product1 = self.product_embedding(product1)
        emb_product2 = self.product_embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)
        bias_product1 = self.product_bias(product1).squeeze()
        bias_product2 = self.product_bias(product2).squeeze()
        biases = self.bias + bias_product1 + bias_product2
        prediction = interaction + biases
        return prediction

    def predict(self, product1, product2):
        emb_product1 = self.product_embedding(product1)
        emb_product2 = self.product_embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)
        bias_product1 = self.product_bias(product1).squeeze()
        bias_product2 = self.product_bias(product2).squeeze()
        biases = self.bias + bias_product1 + bias_product2
        prediction = self.sig(interaction + biases)
        return prediction

    def loss(self, pred, label):
        mf_loss = self.mse(pred, label)
        product_prior = regularize_l2(self.product_embedding.weight) * self.c_vector
        product_bias_prior = regularize_l2(self.product_bias.weight) * self.c_bias
        loss_total = mf_loss + product_prior + product_bias_prior
        return loss_total


class MFContinuous(nn.Module):

    def __init__(self, emb_size, emb_dim, c_vector=1e-06):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.c_vector = c_vector
        self.embedding = nn.Embedding(emb_size, emb_dim)
        self.sig = nn.Sigmoid()
        self.mse = nn.MSELoss()
        logger.info('Model initialized: {}'.format(self))

    def forward(self, product1, product2):
        emb_product1 = self.embedding(product1)
        emb_product2 = self.embedding(product2)
        interaction = torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float)
        return interaction

    def predict(self, product1, product2):
        emb_product1 = self.embedding(product1)
        emb_product2 = self.embedding(product2)
        interaction = self.sig(torch.sum(emb_product1 * emb_product2, dim=1, dtype=torch.float))
        return interaction

    def loss(self, pred, label):
        mf_loss = self.mse(pred, label)
        product_prior = regularize_l2(self.embedding.weight) * self.c_vector
        loss_total = mf_loss + product_prior
        return loss_total


class SkipGram(nn.Module):

    def __init__(self, emb_sizes, emb_dim):
        super().__init__()
        self.emb_sizes = emb_sizes
        self.emb_dim = emb_dim
        self.center_embeddings = nn.ModuleList()
        for k, v in self.emb_sizes.items():
            self.center_embeddings.append(nn.Embedding(v, emb_dim, sparse=True))
        self.context_embeddings = nn.ModuleList()
        for k, v in self.emb_sizes.items():
            self.context_embeddings.append(nn.Embedding(v, emb_dim, sparse=True))
        self.emb_weights = nn.Embedding(emb_sizes['product'], len(emb_sizes), sparse=True)
        self.emb_weights_softmax = nn.Softmax(dim=1)
        self.init_emb()
        logger.info('Model initialized: {}'.format(self))

    def init_emb(self):
        """
        Init embeddings like word2vec

        Center embeddings have uniform distribution in [-0.5/emb_dim , 0.5/emb_dim].
        Context embeddings are initialized with 0s.

        Returns:

        """
        emb_range = 0.5 / self.emb_dim
        for emb in self.center_embeddings:
            emb.weight.data.uniform_(-emb_range, emb_range)
        for emb in self.context_embeddings:
            emb.weight.data.uniform_(0, 0)
        emb_weights_init = 1 / len(self.emb_sizes)
        self.emb_weights.weight.data.uniform_(emb_weights_init)

    def get_embedding(self, nodes):
        embs = []
        emb_weight = self.emb_weights(nodes[:, 0])
        emb_weight_norm = self.emb_weights_softmax(emb_weight)
        for i in range(nodes.shape[1]):
            logger.debug('center i: {}'.format(i))
            embs.append(self.center_embeddings[i](nodes[:, i]))
        emb_stack = torch.stack(embs)
        embs_weighted = emb_stack * emb_weight_norm.T.unsqueeze(2).expand_as(emb_stack)
        emb = torch.sum(embs_weighted, axis=0)
        return emb

    def forward(self, centers, contexts, neg_contexts):
        """

        Args:
            center: List of center words
            context: List of context words
            neg_context: List of list of negative context words

        Returns:

        """
        emb_center = self.get_embedding(centers)
        emb_context = self.get_embedding(contexts)
        neg_contexts = neg_contexts.view(-1, len(self.context_embeddings))
        emb_neg_context = self.get_embedding(neg_contexts)
        score = torch.mul(emb_center, emb_context)
        score = torch.sum(score, dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        neg_score = torch.bmm(emb_neg_context.view(emb_center.shape[0], -1, emb_center.shape[1]), emb_center.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        return torch.mean(score + neg_score)

    def get_center_emb(self, centers):
        emb_centers = []
        for row_idx, center in enumerate(centers):
            emb_center = []
            for col_idx, center_ in enumerate(center):
                emb_center.append(self.center_embeddings[col_idx](center_))
            emb_centers.append(torch.mean(torch.stack(emb_center), axis=0))
        return torch.stack(emb_centers)

    def save_embeddings(self, file_name):
        embedding = self.center_embeddings.weight.cpu().data.numpy()
        np.save(file_name, embedding)

