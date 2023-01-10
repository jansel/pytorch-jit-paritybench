import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
data_batcher = _module
data_loader = _module
dataset = _module
generators = _module
preprocess = _module
evaluation = _module
backend = _module
cython = _module
holdout = _module
loo = _module
python = _module
func = _module
evaluator = _module
early_stop = _module
hparam_search = _module
loggers = _module
base = _module
console_logger = _module
csv_logger = _module
file_logger = _module
neptune = _module
tensorboard = _module
main = _module
BaseModel = _module
CDAE = _module
DAE = _module
EASE = _module
ItemKNN = _module
LightGCN = _module
MF = _module
MultVAE = _module
NGCF = _module
P3a = _module
PureSVD = _module
RP3b = _module
SLIMElastic = _module
models = _module
setup = _module
trainer = _module
helper_func = _module
general = _module
result_table = _module
stats = _module
types = _module

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


import numpy as np


import abc


from typing import MutableMapping


from torch.utils.tensorboard import SummaryWriter


from torch.utils.tensorboard.summary import hparams as hparams_tb


import torch.nn as nn


from collections import OrderedDict


import torch.nn.functional as F


import scipy.sparse as sp


import math


import time


from sklearn.preprocessing import normalize


from sklearn.utils.extmath import randomized_svd


from sklearn.linear_model import ElasticNet


import random


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        pass

    def fit(self, *input):
        pass

    def predict(self, eval_users, eval_pos, test_batch_size):
        pass


class MatrixGenerator:

    def __init__(self, input_matrix, return_index=False, batch_size=32, shuffle=True, matrix_as_numpy=False, index_as_numpy=False, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.return_index = return_index
        self._num_data = self.input_matrix.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.matrix_as_numpy = matrix_as_numpy
        self.index_as_numpy = index_as_numpy
        self.device = device

    def __len__(self):
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data, dtype=np.int32)
        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]
            if self.matrix_as_numpy:
                batch_input = self.input_matrix[batch_idx].toarray()
            else:
                batch_input = torch.tensor(self.input_matrix[batch_idx].toarray(), dtype=torch.float32, device=self.device)
            if self.return_index:
                if not self.index_as_numpy:
                    batch_idx = torch.tensor(batch_idx, dtype=torch.int64, device=self.device)
                yield batch_input, batch_idx
            else:
                yield batch_input


class CDAE(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(CDAE, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.hidden_dim = hparams['hidden_dim']
        self.act = hparams['act']
        self.corruption_ratio = hparams['corruption_ratio']
        self.device = device
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        self
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, user_id, rating_matrix):
        rating_matrix = F.dropout(rating_matrix, self.corruption_ratio, training=self.training)
        enc = torch.tanh(self.encoder(rating_matrix) + self.user_embedding(user_id))
        dec = self.decoder(enc)
        return torch.sigmoid(dec)

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / exp_config.batch_size))
        batch_generator = MatrixGenerator(train_matrix, return_index=True, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, (batch_matrix, batch_users) in enumerate(batch_generator):
                self.optimizer.zero_grad()
                pred_matrix = self.forward(batch_users, batch_matrix)
                batch_loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='none').sum(1).mean()
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss
                if exp_config.verbose and b % 50 == 0:
                    None
            epoch_summary = {'loss': epoch_loss}
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            elif loggers is not None:
                for logger in loggers:
                    logger.log_metrics(epoch_summary, epoch=epoch)
        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}

    def predict(self, eval_users, eval_pos, test_batch_size):
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray())
            preds = np.zeros(shape=input_matrix.shape)
            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
                test_batch_matrix = input_matrix[batch_idx]
                batch_idx_tensor = torch.LongTensor(batch_idx)
                batch_pred_matrix = self.forward(batch_idx_tensor, test_batch_matrix)
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


class DAE(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(DAE, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.hidden_dim = hparams['hidden_dim']
        self.act = hparams['act']
        self.corruption_ratio = hparams['corruption_ratio']
        self.device = device
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        self
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, rating_matrix):
        rating_matrix = F.dropout(rating_matrix, self.corruption_ratio, training=self.training)
        enc = torch.tanh(self.encoder(rating_matrix))
        dec = self.decoder(enc)
        return torch.sigmoid(dec)

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / exp_config.batch_size))
        batch_generator = MatrixGenerator(train_matrix, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, batch_matrix in enumerate(batch_generator):
                self.optimizer.zero_grad()
                pred_matrix = self.forward(batch_matrix)
                batch_loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='none').sum(1).mean()
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss
                if exp_config.verbose and b % 50 == 0:
                    None
            epoch_summary = {'loss': epoch_loss}
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            elif loggers is not None:
                for logger in loggers:
                    logger.log_metrics(epoch_summary, epoch=epoch)
        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}

    def predict(self, eval_users, eval_pos, test_batch_size):
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray())
            preds = np.zeros(eval_pos.shape)
            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
                test_batch_matrix = input_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix)
                preds[batch_idx] += batch_pred_matrix.detach().cpu().numpy()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


class EASE(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(EASE, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.reg = hparams['reg']
        self.device = device
        self

    def forward(self, rating_matrix):
        G = (rating_matrix.T @ rating_matrix).toarray()
        diag = np.diag_indices(G.shape[0])
        G[diag] += self.reg
        P = np.linalg.inv(G)
        self.enc_w = P / -np.diag(P)
        self.enc_w[diag] = 0
        output = rating_matrix @ self.enc_w
        return output

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        self.train()
        train_matrix = dataset.train_data
        output = self.forward(train_matrix)
        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output))
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)
        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = test_batch_matrix @ self.enc_w
            preds[batch_idx] = batch_pred_matrix
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


class ItemKNN(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(ItemKNN, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.topk = hparams['topk']
        self.shrink = hparams['shrink']
        self.feature_weighting = hparams['feature_weighting']
        assert self.feature_weighting in ['tf-idf', 'bm25', 'none']

    def fit_knn(self, train_matrix, block_size=500):
        if self.feature_weighting == 'tf-idf':
            train_matrix = self.TF_IDF(train_matrix.T).T
        elif self.feature_weighting == 'bm25':
            train_matrix = self.okapi_BM25(train_matrix.T).T
        train_matrix = train_matrix.tocsc()
        num_items = train_matrix.shape[1]
        start_col_local = 0
        end_col_local = num_items
        start_col_block = start_col_local
        this_block_size = 0
        block_size = 500
        sumOfSquared = np.array(train_matrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(sumOfSquared)
        values = []
        rows = []
        cols = []
        while start_col_block < end_col_local:
            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block
            item_data = train_matrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)
            this_block_weights = train_matrix.T.dot(item_data)
            for col_index_in_block in range(this_block_size):
                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]
                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0
                denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-06
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)
                relevant_items_partition = (-this_column_weights).argpartition(self.topk - 1)[0:self.topk]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)
                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)
            start_col_block += block_size
        self.W_sparse = sp.csr_matrix((values, (rows, cols)), shape=(num_items, num_items), dtype=np.float32)

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.fit_knn(train_matrix)
        output = train_matrix @ self.W_sparse
        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()))
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)
        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = test_batch_matrix @ self.W_sparse
            preds[batch_idx] = batch_pred_matrix
        preds[eval_pos.nonzero()] = float('-inf')
        return preds

    def okapi_BM25(self, rating_matrix, K1=1.2, B=0.75):
        assert B > 0 and B < 1, 'okapi_BM_25: B must be in (0,1)'
        assert K1 > 0, 'okapi_BM_25: K1 must be > 0'
        rating_matrix = sp.coo_matrix(rating_matrix)
        N = float(rating_matrix.shape[0])
        idf = np.log(N / (1 + np.bincount(rating_matrix.col)))
        row_sums = np.ravel(rating_matrix.sum(axis=1))
        average_length = row_sums.mean()
        length_norm = 1.0 - B + B * row_sums / average_length
        rating_matrix.data = rating_matrix.data * (K1 + 1.0) / (K1 * length_norm[rating_matrix.row] + rating_matrix.data) * idf[rating_matrix.col]
        return rating_matrix.tocsr()

    def TF_IDF(self, rating_matrix):
        """
        Items are assumed to be on rows
        :param dataMatrix:
        :return:
        """
        rating_matrix = sp.coo_matrix(rating_matrix)
        N = float(rating_matrix.shape[0])
        idf = np.log(N / (1 + np.bincount(rating_matrix.col)))
        rating_matrix.data = np.sqrt(rating_matrix.data) * idf[rating_matrix.col]
        return rating_matrix.tocsr()


class PairwiseGenerator:

    def __init__(self, input_matrix, as_numpy=False, num_positives_per_user=-1, num_negatives=1, batch_size=32, shuffle=True, device=None):
        self.input_matrix = input_matrix
        self.num_positives_per_user = num_positives_per_user
        self.num_negatives = num_negatives
        self.as_numpy = as_numpy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self._construct()

    def _construct(self):
        num_users, num_items = self.input_matrix.shape
        self._data = self.sample_negatives()
        self._num_data = len(self._data[0])

    def sample_negatives(self):
        num_users, num_items = self.input_matrix.shape
        users = []
        positives = []
        negatives = []
        for u in range(num_users):
            u_pos_items = self.input_matrix[u].indices
            num_pos_user = len(u_pos_items)
            prob = np.ones(num_items)
            prob[u_pos_items] = 0.0
            prob = prob / sum(prob)
            if self.num_positives_per_user > 0 and self.num_positives_per_user < num_pos_user:
                pos_sampled = np.random.choice(num_items, size=self.num_positives_per_user, replace=False)
                neg_sampled = np.random.choice(num_items, size=self.num_positives_per_user, replace=False, p=prob)
            else:
                pos_sampled = u_pos_items
                neg_sampled = np.random.choice(num_items, size=num_pos_user, replace=False, p=prob)
            assert len(pos_sampled) == len(neg_sampled)
            users += [u] * len(neg_sampled)
            positives += pos_sampled.tolist()
            negatives += neg_sampled.tolist()
        users = np.array(users)
        positives = np.array(positives)
        negatives = np.array(negatives)
        return users, positives, negatives

    def __len__(self):
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data)
        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]
            batch_users = self._data[0][batch_idx]
            batch_pos = self._data[1][batch_idx]
            batch_neg = self._data[2][batch_idx]
            if not self.as_numpy:
                batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                batch_pos = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
                batch_neg = torch.tensor(batch_neg, dtype=torch.long, device=self.device)
            yield batch_users, batch_pos, batch_neg


class LightGCN(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(LightGCN, self).__init__()
        self.data_name = dataset.dataname
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.emb_dim = hparams['emb_dim']
        self.num_layers = hparams['num_layers']
        self.node_dropout = hparams['node_dropout']
        self.split = hparams['split']
        self.num_folds = hparams['num_folds']
        self.reg = hparams['reg']
        self.Graph = None
        self.data_loader = None
        self.path = hparams['graph_dir']
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.device = device
        self.build_graph()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def build_graph(self):
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        self.user_embedding_pred = None
        self.item_embedding_pred = None
        self

    def update_lightgcn_embedding(self):
        self.user_embeddings, self.item_embeddings = self._lightgcn_embedding(self.Graph)

    def forward(self, user_ids, item_ids):
        user_emb = F.embedding(user_ids, self.user_embeddings)
        item_emb = F.embedding(item_ids, self.item_embeddings)
        pred_rating = torch.sum(torch.mul(user_emb, item_emb), 1)
        return pred_rating

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.Graph = self.getSparseGraph(train_matrix)
        batch_generator = PairwiseGenerator(train_matrix, num_negatives=1, num_positives_per_user=1, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        num_batches = len(batch_generator)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, (batch_users, batch_pos, batch_neg) in enumerate(batch_generator):
                self.optimizer.zero_grad()
                batch_loss = self.process_one_batch(batch_users, batch_pos, batch_neg)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss
                if exp_config.verbose and b % 50 == 0:
                    None
            epoch_summary = {'loss': epoch_loss}
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            elif loggers is not None:
                for logger in loggers:
                    logger.log_metrics(epoch_summary, epoch=epoch)
        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}

    def process_one_batch(self, users, pos_items, neg_items):
        self.update_lightgcn_embedding()
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -F.sigmoid(pos_scores - neg_scores).log().mean()
        return loss

    def predict_batch_users(self, user_ids):
        user_embeddings = F.embedding(user_ids, self.user_embeddings)
        item_embeddings = self.item_embeddings
        return user_embeddings @ item_embeddings.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        self.update_lightgcn_embedding()
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()
        pred_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_matrix

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def _lightgcn_embedding(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.node_dropout > 0:
            if self.training:
                g_droped = self.__dropout(graph, self.node_dropout)
            else:
                g_droped = graph
        else:
            g_droped = graph
        for layer in range(self.num_layers):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def make_train_matrix(self):
        train_matrix_arr = self.dataset.train_matrix.toarray()
        self.train_matrix = sp.csr_matrix(train_matrix_arr)

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_users + self.num_items) // self.num_folds
        for i_fold in range(self.num_folds):
            start = i_fold * fold_len
            if i_fold == self.num_folds - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, rating_matrix):
        n_users, n_items = rating_matrix.shape
        None
        filename = f'{self.data_name}_s_pre_adj_mat.npz'
        try:
            pre_adj_mat = sp.load_npz(os.path.join(self.path, filename))
            None
            norm_adj = pre_adj_mat
        except:
            None
            s = time.time()
            adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = rating_matrix.tolil()
            adj_mat[:n_users, n_users:] = R
            adj_mat[n_users:, :n_users] = R.T
            adj_mat = adj_mat.todok()
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            None
            sp.save_npz(os.path.join(self.path, filename), norm_adj)
        if self.split == True:
            Graph = self._split_A_hat(norm_adj)
            None
        else:
            Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            Graph = Graph.coalesce()
            None
        return Graph


class PointwiseGenerator:

    def __init__(self, input_matrix, return_rating=True, as_numpy=False, negative_sample=True, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.return_rating = return_rating
        self.negative_sample = negative_sample
        self.num_negatives = num_negatives
        self.as_numpy = as_numpy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self._construct()

    def _construct(self):
        num_users, num_items = self.input_matrix.shape
        self.users = []
        self.items = []
        self.ratings = []
        for u in range(num_users):
            u_items = self.input_matrix[u].indices
            u_ratings = self.input_matrix[u].data
            self.users += [u] * len(u_items)
            self.items += u_items.tolist()
            if self.return_rating:
                self.ratings += u_ratings.tolist()
        self.users = np.array(self.users)
        self.items = np.array(self.items)
        self.ratings = np.array(self.ratings)
        self._num_data = len(self.users)

    def sample_negatives(self, users):
        num_users, num_items = self.input_matrix.shape
        users = []
        negatives = []
        for u in range(num_users):
            u_pos_items = self.input_matrix[u].indices
            prob = np.ones(num_items)
            prob[u_pos_items] = 0.0
            prob = prob / sum(prob)
            neg_samples = np.random.choice(num_items, size=self.num_negatives, replace=False, p=prob)
            users += [u] * len(neg_samples)
            negatives += neg_samples.tolist()
        users = np.array(users)
        negatives = np.array(negatives)
        ratings = np.zeros_like(users)
        return users, negatives, ratings

    def __len__(self):
        return int(np.ceil(self._num_data / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self._num_data)
        else:
            perm = np.arange(self._num_data)
        for b, st in enumerate(range(0, self._num_data, self.batch_size)):
            ed = min(st + self.batch_size, self._num_data)
            batch_idx = perm[st:ed]
            batch_users = self.users[batch_idx]
            batch_items = self.items[batch_idx]
            if self.return_rating:
                batch_ratings = self.ratings[batch_idx]
                if self.negative_sample and self.num_negatives > 0:
                    neg_users, neg_items, neg_ratings = self.sample_negatives(batch_users)
                    batch_users = np.concatenate((batch_users, neg_users))
                    batch_items = np.concatenate((batch_items, neg_items))
                    batch_ratings = np.concatenate((batch_ratings, neg_ratings))
                if not self.as_numpy:
                    batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                    batch_items = torch.tensor(batch_items, dtype=torch.long, device=self.device)
                    batch_ratings = torch.tensor(batch_ratings, dtype=torch.float32, device=self.device)
                yield batch_users, batch_items, batch_ratings
            else:
                if not self.as_numpy:
                    batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
                    batch_items = torch.tensor(batch_items, dtype=torch.long, device=self.device)
                yield batch_users, batch_items


class MF(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(MF, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.hidden_dim = hparams['hidden_dim']
        self.pointwise = hparams['pointwise']
        self.loss_func = F.mse_loss if hparams['loss_func'] == 'mse' else F.binary_cross_entropy_with_logits
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)
        self.device = device
        self
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def embeddings(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return user_emb, item_emb

    def forward(self, user_ids, item_ids):
        user_emb, item_emb = self.embeddings(user_ids, item_ids)
        pred_rating = torch.sum(torch.mul(user_emb, item_emb), 1)
        return pred_rating

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        if self.pointwise:
            batch_generator = PointwiseGenerator(train_matrix, return_rating=True, num_negatives=1, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        else:
            batch_generator = PairwiseGenerator(train_matrix, num_negatives=1, num_positives_per_user=1, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        num_batches = len(batch_generator)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, (batch_users, batch_pos, batch_ratings) in enumerate(batch_generator):
                self.optimizer.zero_grad()
                batch_loss = self.process_one_batch(batch_users, batch_pos, batch_ratings)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss
                if exp_config.verbose and b % 50 == 0:
                    None
            epoch_summary = {'loss': epoch_loss}
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            elif loggers is not None:
                for logger in loggers:
                    logger.log_metrics(epoch_summary, epoch=epoch)
        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}

    def process_one_batch(self, users, items, ratings):
        pos_ratings = self.forward(users, items)
        if self.pointwise:
            loss = self.loss_func(pos_ratings, ratings)
        else:
            neg_ratings = self.forward(users, ratings)
            loss = -F.sigmoid(pos_ratings - neg_ratings).log().mean()
        return loss

    def predict_batch_users(self, user_ids):
        user_latent = self.user_embedding(user_ids)
        all_item_latent = self.item_embedding.weight.data
        return user_latent @ all_item_latent.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()
        pred_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_matrix


class MultVAE(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(MultVAE, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        if isinstance(hparams['enc_dims'], str):
            hparams['enc_dims'] = eval(hparams['enc_dims'])
        self.enc_dims = [self.num_items] + list(hparams['enc_dims'])
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]
        self.total_anneal_steps = hparams['total_anneal_steps']
        self.anneal_cap = hparams['anneal_cap']
        self.dropout = hparams['dropout']
        self.eps = 1e-06
        self.anneal = 0.0
        self.update_count = 0
        self.device = device
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if i == len(self.enc_dims[:-1]) - 1:
                d_out *= 2
            self.encoder.append(nn.Linear(d_in, d_out))
            if i != len(self.enc_dims[:-1]) - 1:
                self.encoder.append(nn.Tanh())
        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())
        self
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, rating_matrix):
        h = F.dropout(F.normalize(rating_matrix), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)
        mu_q = h[:, :self.enc_dims[-1]]
        logvar_q = h[:, self.enc_dims[-1]:]
        std_q = torch.exp(0.5 * logvar_q)
        epsilon = torch.zeros_like(std_q).normal_(mean=0, std=0.01)
        sampled_z = mu_q + self.training * epsilon * std_q
        output = sampled_z
        for layer in self.decoder:
            output = layer(output)
        if self.training:
            kl_loss = (0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(1).mean()
            return output, kl_loss
        else:
            return output

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / exp_config.batch_size))
        batch_generator = MatrixGenerator(train_matrix, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, batch_matrix in enumerate(batch_generator):
                self.optimizer.zero_grad()
                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1.0 * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap
                pred_matrix, kl_loss = self.forward(batch_matrix)
                ce_loss = F.binary_cross_entropy_with_logits(pred_matrix, batch_matrix, reduction='none').sum(1).mean()
                batch_loss = ce_loss + kl_loss * self.anneal
                batch_loss.backward()
                self.optimizer.step()
                self.update_count += 1
                epoch_loss += batch_loss
                if exp_config.verbose and b % 50 == 0:
                    None
            epoch_summary = {'loss': epoch_loss}
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            elif loggers is not None:
                for logger in loggers:
                    logger.log_metrics(epoch_summary, epoch=epoch)
        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}

    def predict(self, eval_users, eval_pos, test_batch_size):
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray())
            preds = np.zeros(eval_pos.shape)
            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
                test_batch_matrix = input_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix)
                preds[batch_idx] = batch_pred_matrix.detach().cpu().numpy()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


class NGCF(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(NGCF, self).__init__()
        self.data_name = dataset.dataname
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.emb_dim = hparams['emb_dim']
        self.num_layers = hparams['num_layers']
        self.node_dropout = hparams['node_dropout']
        self.mess_dropout = hparams['mess_dropout']
        self.split = hparams['split']
        self.num_folds = hparams['num_folds']
        self.reg = hparams['reg']
        self.Graph = None
        self.data_loader = None
        self.path = hparams['graph_dir']
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.device = device
        self.build_graph()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def build_graph(self):
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)
        self.weight_dict = nn.ParameterDict()
        layers = [self.emb_dim] * (self.num_layers + 1)
        for k in range(len(layers) - 1):
            self.weight_dict.update({('W_gc_%d' % k): nn.Parameter(nn.init.normal_(torch.empty(layers[k], layers[k + 1])))})
            self.weight_dict.update({('b_gc_%d' % k): nn.Parameter(nn.init.normal_(torch.empty(1, layers[k + 1])))})
            self.weight_dict.update({('W_bi_%d' % k): nn.Parameter(nn.init.normal_(torch.empty(layers[k], layers[k + 1])))})
            self.weight_dict.update({('b_bi_%d' % k): nn.Parameter(nn.init.normal_(torch.empty(1, layers[k + 1])))})
        self

    def update_ngcf_embedding(self):
        self.user_embeddings, self.item_embeddings = self._ngcf_embedding(self.Graph)

    def forward(self, user_ids, item_ids):
        user_emb = F.embedding(user_ids, self.user_embeddings)
        item_emb = F.embedding(item_ids, self.item_embeddings)
        pred_rating = torch.sum(torch.mul(user_emb, item_emb), 1)
        return pred_rating

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.Graph = self.getSparseGraph(train_matrix)
        batch_generator = PairwiseGenerator(train_matrix, num_negatives=1, num_positives_per_user=1, batch_size=exp_config.batch_size, shuffle=True, device=self.device)
        num_batches = len(batch_generator)
        for epoch in range(1, exp_config.num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            for b, (batch_users, batch_pos, batch_neg) in enumerate(batch_generator):
                self.optimizer.zero_grad()
                batch_loss = self.process_one_batch(batch_users, batch_pos, batch_neg)
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss
                if exp_config.verbose and b % 50 == 0:
                    None
            epoch_summary = {'loss': epoch_loss}
            if evaluator is not None and epoch >= exp_config.test_from and epoch % exp_config.test_step == 0:
                scores = evaluator.evaluate(self)
                epoch_summary.update(scores)
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(epoch_summary, epoch=epoch)
                if early_stop is not None:
                    is_update, should_stop = early_stop.step(scores, epoch)
                    if should_stop:
                        break
            elif loggers is not None:
                for logger in loggers:
                    logger.log_metrics(epoch_summary, epoch=epoch)
        best_score = early_stop.best_score if early_stop is not None else scores
        return {'scores': best_score}

    def process_one_batch(self, users, pos_items, neg_items):
        self.update_ngcf_embedding()
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -F.sigmoid(pos_scores - neg_scores).log().mean()
        return loss

    def predict_batch_users(self, user_ids):
        user_embeddings = F.embedding(user_ids, self.user_embeddings)
        item_embeddings = self.item_embeddings
        return user_embeddings @ item_embeddings.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        self.update_ngcf_embedding()
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()
        pred_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_matrix

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def _ngcf_embedding(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.node_dropout > 0:
            if self.training:
                g_droped = self.__dropout(graph, self.node_dropout)
            else:
                g_droped = graph
        else:
            g_droped = graph
        ego_emb = all_emb
        for k in range(self.num_layers):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], ego_emb))
                side_emb = torch.cat(temp_emb, dim=0)
            else:
                side_emb = torch.sparse.mm(g_droped, ego_emb)
            sum_emb = torch.matmul(side_emb, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]
            bi_emb = torch.mul(ego_emb, side_emb)
            bi_emb = torch.matmul(bi_emb, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]
            ego_emb = F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)
            ego_emb = F.dropout(ego_emb, self.mess_dropout, training=self.training)
            norm_emb = F.normalize(ego_emb, p=2, dim=1)
            embs += [norm_emb]
        embs = torch.stack(embs, dim=1)
        ngcf_out = torch.mean(embs, dim=1)
        users, items = torch.split(ngcf_out, [self.num_users, self.num_items])
        return users, items

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_users + self.num_items) // self.num_folds
        for i_fold in range(self.num_folds):
            start = i_fold * fold_len
            if i_fold == self.num_folds - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, rating_matrix):
        n_users, n_items = rating_matrix.shape
        None
        filename = f'{self.data_name}_s_pre_adj_mat.npz'
        try:
            pre_adj_mat = sp.load_npz(os.path.join(self.path, filename))
            None
            norm_adj = pre_adj_mat
        except:
            None
            s = time.time()
            adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = rating_matrix.tolil()
            adj_mat[:n_users, n_users:] = R
            adj_mat[n_users:, :n_users] = R.T
            adj_mat = adj_mat.todok()
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            None
            sp.save_npz(os.path.join(self.path, filename), norm_adj)
        if self.split == True:
            Graph = self._split_A_hat(norm_adj)
            None
        else:
            Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            Graph = Graph.coalesce()
            None
        return Graph


class P3a(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(P3a, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.topk = hparams['topk']
        self.alpha = hparams['alpha']

    def fit_p3a(self, train_matrix, block_dim=200):
        num_items = train_matrix.shape[1]
        Pui = normalize(train_matrix, norm='l1', axis=1)
        X_bool = train_matrix.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        Piu = normalize(X_bool, norm='l1', axis=1)
        del X_bool
        if self.alpha != 1:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)
        dataBlock = 10000000
        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)
        numCells = 0
        item_blocks = range(0, num_items, block_dim)
        tqdm_iterator = tqdm(item_blocks, desc='# items blocks covered', total=len(item_blocks))
        for cur_items_start_idx in tqdm_iterator:
            if cur_items_start_idx + block_dim > num_items:
                block_dim = num_items - cur_items_start_idx
            Piui = Piu[cur_items_start_idx:cur_items_start_idx + block_dim, :] * Pui
            Piui = Piui.toarray()
            for row_in_block in range(block_dim):
                row_data = Piui[row_in_block, :]
                row_data[cur_items_start_idx + row_in_block] = 0
                best = row_data.argsort()[::-1][:self.topk]
                notZerosMask = row_data[best] != 0.0
                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]
                for index in range(len(values_to_add)):
                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))
                    rows[numCells] = cur_items_start_idx + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]
                    numCells += 1
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.fit_p3a(train_matrix.tocsc())
        output = train_matrix @ self.W_sparse
        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()))
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)
        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = test_batch_matrix @ self.W_sparse
            preds[batch_idx] = batch_pred_matrix
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


class PureSVD(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(PureSVD, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_factors = hparams['num_factors']
        self.device = device

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data.toarray()
        U, sigma, Vt = randomized_svd(train_matrix, n_components=self.num_factors, random_state=123)
        s_Vt = sp.diags(sigma) * Vt
        self.user_embedding = U
        self.item_embedding = s_Vt.T
        output = self.user_embedding @ self.item_embedding.T
        loss = F.binary_cross_entropy(torch.tensor(train_matrix), torch.tensor(output))
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        return {'scores': scores, 'loss': loss}

    def predict_batch_users(self, user_ids):
        user_latent = self.user_embedding[user_ids]
        return user_latent @ self.item_embedding.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_eval_users:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
            batch_users = eval_users[batch_idx]
            pred_matrix[batch_users] = self.predict_batch_users(batch_users)
        pred_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_matrix


class RP3b(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(RP3b, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.topk = hparams['topk']
        self.alpha = hparams['alpha']
        self.beta = hparams['beta']

    def fit_rp3b(self, train_matrix, block_dim=200):
        num_items = train_matrix.shape[1]
        Pui = normalize(train_matrix, norm='l1', axis=1)
        X_bool = train_matrix.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        degree = np.zeros(train_matrix.shape[1])
        nonZeroMask = X_bool_sum != 0.0
        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)
        Piu = normalize(X_bool, norm='l1', axis=1)
        del X_bool
        if self.alpha != 1:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)
        dataBlock = 10000000
        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)
        numCells = 0
        item_blocks = range(0, num_items, block_dim)
        tqdm_iterator = tqdm(item_blocks, desc='# items blocks covered', total=len(item_blocks))
        for cur_items_start_idx in tqdm_iterator:
            if cur_items_start_idx + block_dim > num_items:
                block_dim = num_items - cur_items_start_idx
            Piui = Piu[cur_items_start_idx:cur_items_start_idx + block_dim, :] * Pui
            Piui = Piui.toarray()
            for row_in_block in range(block_dim):
                row_data = np.multiply(Piui[row_in_block, :], degree)
                row_data[cur_items_start_idx + row_in_block] = 0
                best = row_data.argsort()[::-1][:self.topk]
                notZerosMask = row_data[best] != 0.0
                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]
                for index in range(len(values_to_add)):
                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))
                    rows[numCells] = cur_items_start_idx + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]
                    numCells += 1
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data
        self.fit_rp3b(train_matrix.tocsc())
        output = train_matrix @ self.W_sparse
        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()))
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        preds = (eval_pos * self.W_sparse).toarray()
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


class SLIM(BaseModel):

    def __init__(self, dataset, hparams, device):
        super(SLIM, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.l1_reg = hparams['l1_reg']
        self.l2_reg = hparams['l2_reg']
        self.topk = hparams['topk']
        self.device = device
        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha
        self.slim = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False, copy_X=False, precompute=True, selection='random', max_iter=300, tol=0.001)

    def fit_slim(self, train_matrix, num_blocks=10000000):
        num_items = train_matrix.shape[1]
        rows = np.zeros(num_blocks, dtype=np.int32)
        cols = np.zeros(num_blocks, dtype=np.int32)
        values = np.zeros(num_blocks, dtype=np.float32)
        numCells = 0
        tqdm_iterator = tqdm(range(num_items), desc='# items covered', total=num_items)
        for item in tqdm_iterator:
            y = train_matrix[:, item].toarray()
            start_pos = train_matrix.indptr[item]
            end_pos = train_matrix.indptr[item + 1]
            current_item_data_backup = train_matrix.data[start_pos:end_pos].copy()
            train_matrix.data[start_pos:end_pos] = 0.0
            self.slim.fit(train_matrix, y)
            nonzero_model_coef_index = self.slim.sparse_coef_.indices
            nonzero_model_coef_value = self.slim.sparse_coef_.data
            local_topK = min(len(nonzero_model_coef_value) - 1, self.topk)
            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]
            for index in range(len(ranking)):
                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(num_blocks, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(num_blocks, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(num_blocks, dtype=np.float32)))
                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = item
                values[numCells] = nonzero_model_coef_value[ranking[index]]
                numCells += 1
            train_matrix.data[start_pos:end_pos] = current_item_data_backup
        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(num_items, num_items), dtype=np.float32)

    def fit(self, dataset, exp_config, evaluator=None, early_stop=None, loggers=None):
        train_matrix = dataset.train_data.tocsc()
        self.fit_slim(train_matrix)
        output = train_matrix.tocsr() @ self.W_sparse
        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()))
        if evaluator is not None:
            scores = evaluator.evaluate(self)
        else:
            scores = None
        if loggers is not None:
            if evaluator is not None:
                for logger in loggers:
                    logger.log_metrics(scores, epoch=1)
        return {'scores': scores, 'loss': loss}

    def predict(self, eval_users, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)
        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size:(b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = test_batch_matrix @ self.W_sparse
            preds[batch_idx] = batch_pred_matrix
        preds[eval_pos.nonzero()] = float('-inf')
        return preds


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
]

class Test_yoongi0428_RecSys_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

