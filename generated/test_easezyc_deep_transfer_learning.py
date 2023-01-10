import sys
_module = sys.modules[__name__]
del sys
process = _module
main = _module
metamodel = _module
model = _module
run = _module
utils = _module
dataset = _module
main = _module
hen = _module
layer = _module
lstm4fd = _module
m3r = _module
nfm = _module
wd = _module
utils = _module
weight = _module
data_loader = _module
mfsan = _module
mmd = _module
resnet = _module
data_loader = _module
mfsan = _module
mmd = _module
resnet = _module
DAN = _module
ResNet = _module
data_loader = _module
mmd = _module
DDC = _module
ResNet = _module
data_loader = _module
mmd = _module
Coral = _module
DeepCoral = _module
ResNet = _module
data_loader = _module
ResNet = _module
RevGrad = _module
data_loader = _module
DAN = _module
ResNet = _module
data_loader = _module
mmd = _module
DSAN = _module
ResNet = _module
data_loader = _module
lmmd = _module
main = _module
Coral = _module
DeepCoral = _module
ResNet = _module
data_loader = _module
MRAN = _module
ResNet = _module
data_loader = _module
mmd = _module

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


import random


from collections import OrderedDict


from torch.nn import functional as F


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import pandas as pd


from tensorflow import keras


from sklearn.metrics import roc_auc_score


import copy


import math


import torch.utils.data


from torch.utils.data import WeightedRandomSampler


import time


from torchvision import datasets


from torchvision import transforms


from torch.autograd import Variable


import torch.utils.model_zoo as model_zoo


import torch.optim as optim


from torch.utils import model_zoo


from torch.autograd import Function


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x, values):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        values = values.unsqueeze(3)
        return values * self.embedding(x)


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = 11 * 56 * embed_dim
        self.layer = torch.nn.Linear(self.embed_output_dim, 32)
        self.src_layer = torch.nn.Linear(self.embed_output_dim, 32)
        self.tgt_layer = torch.nn.Linear(self.embed_output_dim, 32)
        self.linear = FeaturesLinear(field_dims)
        self.mlp = MultiLayerPerceptron(32, mlp_dims, dropout)
        self.src_domain_K = torch.nn.Linear(32, 32)
        self.src_domain_Q = torch.nn.Linear(32, 32)
        self.src_domain_V = torch.nn.Linear(32, 32)
        self.tgt_domain_K = torch.nn.Linear(32, 32)
        self.tgt_domain_Q = torch.nn.Linear(32, 32)
        self.tgt_domain_V = torch.nn.Linear(32, 32)

    def forward(self, ids, values, seq_lengths, seq_mask, dlabel):
        """
        :param
        ids: the ids of fields (batch_size, seqlength, fields)
        values: the values of fields (batch_size, seqlength, fields)
        seq_length: the length of historical events (batch_size, 1)
        seq_mask: the attention mask for historical events (batch_size, seqlength)
        dlabel: the domain label of the batch samples (batch_size, 1)
        :return
        torch.sigmoid(result.squeeze(1)): the predition of the target payment
        term: the sequence embedding, output of user behavior extractor (batch_size, 32)
        """
        batch_size = ids.size()[0]
        if dlabel == 'src':
            shared_emb = self.embedding(ids, values).view(batch_size, -1)
            shared_term = self.layer(shared_emb)
            src_emb = self.src_embedding(ids, values).view(batch_size, -1)
            src_term = self.src_layer(src_emb)
            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 6)
            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 6)
            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
        if dlabel == 'tgt':
            shared_emb = self.embedding(ids, values).view(batch_size, -1)
            shared_term = self.layer(shared_emb)
            tgt_emb = self.tgt_embedding(ids, values).view(batch_size, -1)
            tgt_term = self.tgt_layer(tgt_emb)
            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 6)
            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 6)
            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term


class MetaModel(torch.nn.Module):

    def __init__(self, col_names, max_ids, embed_dim, mlp_dims, dropout, use_cuda, local_lr, global_lr, weight_decay, base_model_name, num_expert, num_output):
        super(MetaModel, self).__init__()
        if base_model_name == 'WD':
            self.model = WideAndDeepModel(col_names=col_names, max_ids=max_ids, embed_dim=embed_dim, mlp_dims=mlp_dims, dropout=dropout, use_cuda=use_cuda, num_expert=num_expert, num_output=num_output)
        self.local_lr = local_lr
        self.criterion = torch.nn.BCELoss()
        self.meta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=global_lr, weight_decay=weight_decay)

    def forward(self, x):
        return self.model(x)

    def local_update(self, support_set_x, support_set_y):
        batch_size = support_set_x.shape[0]
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        support_set_y_pred = self.model(support_set_x)
        label = torch.from_numpy(support_set_y.astype('float32'))
        loss = self.criterion(support_set_y_pred, label)
        self.model.zero_grad()
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
        fast_parameters = []
        for k, weight in enumerate(self.model.parameters()):
            if grad[k] is None:
                continue
            if weight.fast is None:
                weight.fast = weight - self.local_lr * grad[k]
            else:
                weight.fast = weight.fast - self.local_lr * grad[k]
            fast_parameters.append(weight.fast)
        return loss

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = len(support_set_xs)
        losses_q = []
        for i in range(batch_sz):
            loss_sup = self.local_update(support_set_xs[i], support_set_ys[i])
            query_set_y_pred = self.model(query_set_xs[i])
            label = torch.from_numpy(query_set_ys[i].astype('float32'))
            loss_q = self.criterion(query_set_y_pred, label)
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optimizer.zero_grad()
        losses_q.backward()
        self.meta_optimizer.step()
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        return losses_q


class Meta_Linear(torch.nn.Linear):

    def __init__(self, in_features, out_features):
        super(Meta_Linear, self).__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Meta_Linear, self).forward(x)
        return out


class Meta_Embedding(torch.nn.Embedding):

    def __init__(self, num_embedding, embedding_dim):
        super(Meta_Embedding, self).__init__(num_embedding, embedding_dim)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.embedding(x, self.weight.fast, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            out = F.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out


class DyEmb(nn.Module):

    def __init__(self, fnames, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DyEmb, self).__init__()
        self.fnames = fnames
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        self.embeddings = nn.ModuleList([Meta_Embedding(max_idx + 1, self.embedding_size) for max_idx in self.max_idxs.values()])

    def forward(self, dynamic_ids, dynamic_lengths):
        """
        input: relative id
        dynamic_ids: Batch_size * Field_size * Max_feature_size
        dynamic_lengths: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        for i, key in enumerate(self.fnames):
            dynamic_ids_tensor = torch.LongTensor(np.array(dynamic_ids[key].values.tolist()))
            dynamic_lengths_tensor = torch.LongTensor(dynamic_lengths[key + '_length'].values.astype(int))
            if self.use_cuda:
                dynamic_ids_tensor = dynamic_ids_tensor
            batch_size = dynamic_ids_tensor.size()[0]
            dynamic_embeddings_tensor = self.embeddings[i](dynamic_ids_tensor)
            dynamic_lengths_tensor = dynamic_lengths_tensor.unsqueeze(1)
            mask = (torch.arange(dynamic_embeddings_tensor.size(1))[None, :] < dynamic_lengths_tensor[:, None]).type(torch.FloatTensor)
            mask = mask.squeeze(1).unsqueeze(2)
            dynamic_embedding = dynamic_embeddings_tensor.masked_fill(mask == 0, 0)
            dynamic_lengths_tensor[dynamic_lengths_tensor == 0] = 1
            dynamic_embedding = (dynamic_embedding.sum(dim=1) / dynamic_lengths_tensor).unsqueeze(1)
            concat_embeddings.append(dynamic_embedding.view(batch_size, 1, self.embedding_size))
        concat_embeddings = torch.cat(concat_embeddings, 1)
        return concat_embeddings


class StEmb(nn.Module):

    def __init__(self, col_names, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(StEmb, self).__init__()
        self.col_names = col_names
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        self.embeddings = nn.ModuleList([Meta_Embedding(max_idx + 1, self.embedding_size) for max_idx in self.max_idxs.values()])

    def forward(self, static_ids):
        """
        input: relative id
        static_ids: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        batch_size = static_ids.shape[0]
        for i, key in enumerate(self.col_names):
            static_ids_tensor = torch.LongTensor(static_ids[key].values.astype(int))
            if self.use_cuda:
                static_ids_tensor = static_ids_tensor
            static_embeddings_tensor = self.embeddings[i](static_ids_tensor)
            concat_embeddings.append(static_embeddings_tensor.view(batch_size, 1, self.embedding_size))
        concat_embeddings = torch.cat(concat_embeddings, 1)
        return concat_embeddings


class Emb(nn.Module):

    def __init__(self, col_names, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(Emb, self).__init__()
        self.static_emb = StEmb(col_names['static'], max_idxs['static'], embedding_size, use_cuda)
        self.ad_emb = StEmb(col_names['ad'], max_idxs['ad'], embedding_size, use_cuda)
        self.dynamic_emb = DyEmb(col_names['dynamic'], max_idxs['dynamic'], embedding_size, use_cuda)
        self.col_names = col_names
        self.col_length_name = [(x + '_length') for x in col_names['dynamic']]

    def forward(self, x):
        static_emb = self.static_emb(x[self.col_names['static']])
        dynamic_emb = self.dynamic_emb(x[self.col_names['dynamic']], x[self.col_length_name])
        concat_embeddings = torch.cat([static_emb, dynamic_emb], 1)
        ad_emb = self.ad_emb(x[self.col_names['ad']])
        return concat_embeddings, ad_emb


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class HENModel(torch.nn.Module):
    """
    A pytorch implementation of Hierarchical Exlainable Network.
    """

    def __init__(self, field_dims, embed_dim, sequence_length, lstm_dims, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.mlp = MultiLayerPerceptron(embed_dim + embed_dim, mlp_dims, dropouts[1])
        self.attention = torch.nn.Embedding(sum(field_dims), 1)
        self.src_attention = torch.nn.Embedding(sum(field_dims), 1)
        self.tgt_attention = torch.nn.Embedding(sum(field_dims), 1)
        self.attr_softmax = torch.nn.Softmax(dim=2)
        self.fm = FactorizationMachine(reduce_sum=False)
        self.src_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(sequence_length), torch.nn.Dropout(dropouts[0]))
        self.tgt_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(sequence_length), torch.nn.Dropout(dropouts[0]))
        self.bn = torch.nn.Sequential(torch.nn.BatchNorm1d(sequence_length), torch.nn.Dropout(dropouts[0]))
        self.event_K = torch.nn.Linear(embed_dim, embed_dim)
        self.event_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.event_V = torch.nn.Linear(embed_dim, embed_dim)
        self.src_event_K = torch.nn.Linear(embed_dim, embed_dim)
        self.src_event_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.src_event_V = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_event_K = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_event_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_event_V = torch.nn.Linear(embed_dim, embed_dim)
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.src_domain_K = torch.nn.Linear(32, 32)
        self.src_domain_Q = torch.nn.Linear(32, 32)
        self.src_domain_V = torch.nn.Linear(32, 32)
        self.tgt_domain_K = torch.nn.Linear(32, 32)
        self.tgt_domain_Q = torch.nn.Linear(32, 32)
        self.tgt_domain_V = torch.nn.Linear(32, 32)

    def forward(self, ids, values, seq_lengths, seq_mask, dlabel):
        """
        :param
        ids: the ids of fields (batch_size, seqlength, fields)
        values: the values of fields (batch_size, seqlength, fields)
        seq_length: the length of historical events (batch_size, 1)
        seq_mask: the attention mask for historical events (batch_size, seqlength)
        dlabel: the domain label of the batch samples (batch_size, 1)
        :return
        torch.sigmoid(result.squeeze(1)): the predition of the target payment
        term: the sequence embedding, output of user behavior extractor (batch_size, 32)
        """
        if dlabel == 'src':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            src_emb = self.src_embedding(ids, values)
            src_attention = self.attr_softmax(self.src_attention(ids))
            src_event_fea = self.src_bn(torch.mean(src_attention * src_emb, 2) + self.fm(src_emb))
            src_payment_fea = src_event_fea[:, -1, :]
            src_history_fea = src_event_fea[:, :-1, :]
            src_event_K = self.src_event_K(src_history_fea)
            src_event_Q = self.src_event_Q(src_history_fea)
            src_event_V = self.src_event_V(src_history_fea)
            t = torch.sum(src_event_K * src_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 100000000.0
            src_his_fea = torch.sum(self.event_softmax(t) * src_event_V, 1)
            shared_attention = self.attr_softmax(self.attention(ids))
            shared_event_fea = self.bn(torch.mean(shared_attention * shared_emb, 2) + self.fm(shared_emb))
            shared_payment_fea = shared_event_fea[:, -1, :]
            shared_history_fea = shared_event_fea[:, :-1, :]
            shared_event_K = self.event_K(shared_history_fea)
            shared_event_Q = self.event_Q(shared_history_fea)
            shared_event_V = self.event_V(shared_history_fea)
            t = torch.sum(shared_event_K * shared_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 100000000.0
            shared_his_fea = torch.sum(self.event_softmax(t) * shared_event_V, 1)
            src_term = torch.cat((src_his_fea, src_payment_fea), 1)
            shared_term = torch.cat((shared_his_fea, shared_payment_fea), 1)
            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 6)
            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 6)
            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
        elif dlabel == 'tgt':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            tgt_emb = self.tgt_embedding(ids, values)
            tgt_attention = self.attr_softmax(self.tgt_attention(ids))
            tgt_event_fea = self.tgt_bn(torch.mean(tgt_attention * tgt_emb, 2) + self.fm(tgt_emb))
            tgt_payment_fea = tgt_event_fea[:, -1, :]
            tgt_history_fea = tgt_event_fea[:, :-1, :]
            tgt_event_K = self.tgt_event_K(tgt_history_fea)
            tgt_event_Q = self.tgt_event_Q(tgt_history_fea)
            tgt_event_V = self.tgt_event_V(tgt_history_fea)
            t = torch.sum(tgt_event_K * tgt_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 100000000.0
            tgt_his_fea = torch.sum(self.event_softmax(t) * tgt_event_V, 1)
            shared_attention = self.attr_softmax(self.attention(ids))
            shared_event_fea = self.bn(torch.mean(shared_attention * shared_emb, 2) + self.fm(shared_emb))
            shared_payment_fea = shared_event_fea[:, -1, :]
            shared_history_fea = shared_event_fea[:, :-1, :]
            shared_event_K = self.event_K(shared_history_fea)
            shared_event_Q = self.event_Q(shared_history_fea)
            shared_event_V = self.event_V(shared_history_fea)
            t = torch.sum(shared_event_K * shared_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 100000000.0
            shared_his_fea = torch.sum(self.event_softmax(t) * shared_event_V, 1)
            tgt_term = torch.cat((tgt_his_fea, tgt_payment_fea), 1)
            shared_term = torch.cat((shared_his_fea, shared_payment_fea), 1)
            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 6)
            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 6)
            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term


class LSTM4FDModel(torch.nn.Module):
    """
    A pytorch implementation LSTM4FD
    Reference:
        Wang S, Liu C, Gao X, et al. Session-based fraud detection in online e-commerce transactions using recurrent neural networks[C]//Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Cham, 2017: 241-252.
    """

    def __init__(self, field_dims, embed_dim, sequence_length, lstm_dims, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.mlp = MultiLayerPerceptron(embed_dim + embed_dim, mlp_dims, dropouts[1])
        self.embed_dim = embed_dim
        self.src_domain_K = torch.nn.Linear(32, 32)
        self.src_domain_Q = torch.nn.Linear(32, 32)
        self.src_domain_V = torch.nn.Linear(32, 32)
        self.tgt_domain_K = torch.nn.Linear(32, 32)
        self.tgt_domain_Q = torch.nn.Linear(32, 32)
        self.tgt_domain_V = torch.nn.Linear(32, 32)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.src_lstm = torch.nn.LSTM(embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.tgt_lstm = torch.nn.LSTM(embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, ids, values, seq_lengths, seq_mask, dlabel):
        """
        :param
        ids: the ids of fields (batch_size, seqlength, fields)
        values: the values of fields (batch_size, seqlength, fields)
        seq_length: the length of historical events (batch_size, 1)
        seq_mask: the attention mask for historical events (batch_size, seqlength)
        dlabel: the domain label of the batch samples (batch_size, 1)
        :return
        torch.sigmoid(result.squeeze(1)): the predition of the target payment
        term: the sequence embedding, output of user behavior extractor (batch_size, 32)
        """
        batch_size = ids.size()[0]
        if dlabel == 'src':
            shared_emb = self.embedding(ids, values)
            shared_t = torch.mean(shared_emb, 2)
            shared_history = shared_t[:, :-1, :]
            shared_term = shared_t[:, -1:, :].view(batch_size, -1)
            shared_pack = torch.nn.utils.rnn.pack_padded_sequence(shared_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (shared_lstm_hn, __) = self.lstm(shared_pack)
            shared_lstm_hn = torch.mean(shared_lstm_hn, dim=0)
            shared_term = torch.cat((shared_lstm_hn, shared_term), 1)
            src_emb = self.src_embedding(ids, values)
            src_t = torch.mean(src_emb, 2)
            src_history = src_t[:, :-1, :]
            src_term = src_t[:, -1:, :].view(batch_size, -1)
            src_pack = torch.nn.utils.rnn.pack_padded_sequence(src_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (src_lstm_hn, __) = self.src_lstm(src_pack)
            src_lstm_hn = torch.mean(src_lstm_hn, dim=0)
            src_term = torch.cat((src_lstm_hn, src_term), 1)
            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 7)
            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 7)
            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
        if dlabel == 'tgt':
            shared_emb = self.embedding(ids, values)
            shared_t = torch.mean(shared_emb, 2)
            shared_history = shared_t[:, :-1, :]
            shared_term = shared_t[:, -1:, :].view(batch_size, -1)
            shared_pack = torch.nn.utils.rnn.pack_padded_sequence(shared_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (shared_lstm_hn, __) = self.lstm(shared_pack)
            shared_lstm_hn = torch.mean(shared_lstm_hn, dim=0)
            shared_term = torch.cat((shared_lstm_hn, shared_term), 1)
            tgt_emb = self.tgt_embedding(ids, values)
            tgt_t = torch.mean(tgt_emb, 2)
            tgt_history = tgt_t[:, :-1, :]
            tgt_term = tgt_t[:, -1:, :].view(batch_size, -1)
            tgt_pack = torch.nn.utils.rnn.pack_padded_sequence(tgt_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (tgt_lstm_hn, __) = self.tgt_lstm(tgt_pack)
            tgt_lstm_hn = torch.mean(tgt_lstm_hn, dim=0)
            tgt_term = torch.cat((tgt_lstm_hn, tgt_term), 1)
            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 7)
            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 7)
            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term


class SeqM3RModel(torch.nn.Module):
    """
    A pytorch implementation of M3R.
    Reference:
        Tang J, Belletti F, Jain S, et al. Towards neural mixture recommender for long range dependent user sequences[C]//The World Wide Web Conference. ACM, 2019: 1782-1793.
    """

    def __init__(self, field_dims, embed_dim, sequence_length, lstm_dims, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fc = torch.nn.Linear(in_features=56 * embed_dim, out_features=embed_dim)
        self.src_fc = torch.nn.Linear(in_features=56 * embed_dim, out_features=embed_dim)
        self.tgt_fc = torch.nn.Linear(in_features=56 * embed_dim, out_features=embed_dim)
        self.bn = torch.nn.Sequential(torch.nn.BatchNorm1d(sequence_length), torch.nn.ReLU(), torch.nn.Dropout(dropouts[0]))
        self.src_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(sequence_length), torch.nn.ReLU(), torch.nn.Dropout(dropouts[0]))
        self.tgt_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(sequence_length), torch.nn.ReLU(), torch.nn.Dropout(dropouts[0]))
        self.linear = FeaturesLinear(field_dims)
        self.mlp = MultiLayerPerceptron(embed_dim + embed_dim, mlp_dims, dropouts[1])
        self.embed_dim = embed_dim
        self.src_domain_K = torch.nn.Linear(32, 32)
        self.src_domain_Q = torch.nn.Linear(32, 32)
        self.src_domain_V = torch.nn.Linear(32, 32)
        self.tgt_domain_K = torch.nn.Linear(32, 32)
        self.tgt_domain_Q = torch.nn.Linear(32, 32)
        self.tgt_domain_V = torch.nn.Linear(32, 32)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.src_lstm = torch.nn.LSTM(embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.tgt_lstm = torch.nn.LSTM(embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, ids, values, seq_lengths, seq_mask, dlabel):
        """
        :param
        ids: the ids of fields (batch_size, seqlength, fields)
        values: the values of fields (batch_size, seqlength, fields)
        seq_length: the length of historical events (batch_size, 1)
        seq_mask: the attention mask for historical events (batch_size, seqlength)
        dlabel: the domain label of the batch samples (batch_size, 1)
        :return
        torch.sigmoid(result.squeeze(1)): the predition of the target payment
        term: the sequence embedding, output of user behavior extractor (batch_size, 32)
        """
        batch_size = ids.size()[0]
        if dlabel == 'src':
            shared_emb = self.embedding(ids, values).view(-1, 56 * self.embed_dim)
            shared_t = self.bn(self.fc(shared_emb).view(batch_size, -1, self.embed_dim))
            shared_history = shared_t[:, :-1, :]
            shared_term = shared_t[:, -1:, :].view(batch_size, -1)
            shared_pack = torch.nn.utils.rnn.pack_padded_sequence(shared_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (shared_lstm_hn, __) = self.lstm(shared_pack)
            shared_lstm_hn = torch.mean(shared_lstm_hn, dim=0)
            shared_term = torch.cat((shared_lstm_hn, shared_term), 1)
            src_emb = self.src_embedding(ids, values).view(-1, 56 * self.embed_dim)
            src_t = self.src_bn(self.src_fc(src_emb).view(batch_size, -1, self.embed_dim))
            src_history = src_t[:, :-1, :]
            src_term = src_t[:, -1:, :].view(batch_size, -1)
            src_pack = torch.nn.utils.rnn.pack_padded_sequence(src_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (src_lstm_hn, __) = self.src_lstm(src_pack)
            src_lstm_hn = torch.mean(src_lstm_hn, dim=0)
            src_term = torch.cat((src_lstm_hn, src_term), 1)
            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 7)
            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 7)
            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
        if dlabel == 'tgt':
            shared_emb = self.embedding(ids, values).view(-1, 56 * self.embed_dim)
            shared_t = self.bn(self.fc(shared_emb).view(batch_size, -1, self.embed_dim))
            shared_history = shared_t[:, :-1, :]
            shared_term = shared_t[:, -1:, :].view(batch_size, -1)
            shared_pack = torch.nn.utils.rnn.pack_padded_sequence(shared_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (shared_lstm_hn, __) = self.lstm(shared_pack)
            shared_lstm_hn = torch.mean(shared_lstm_hn, dim=0)
            shared_term = torch.cat((shared_lstm_hn, shared_term), 1)
            tgt_emb = self.tgt_embedding(ids, values).view(-1, 56 * self.embed_dim)
            tgt_t = self.tgt_bn(self.tgt_fc(tgt_emb).view(batch_size, -1, self.embed_dim))
            tgt_history = tgt_t[:, :-1, :]
            tgt_term = tgt_t[:, -1:, :].view(batch_size, -1)
            tgt_pack = torch.nn.utils.rnn.pack_padded_sequence(tgt_history, seq_lengths, batch_first=True, enforce_sorted=False)
            _, (tgt_lstm_hn, __) = self.tgt_lstm(tgt_pack)
            tgt_lstm_hn = torch.mean(tgt_lstm_hn, dim=0)
            tgt_term = torch.cat((tgt_lstm_hn, tgt_term), 1)
            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 7)
            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 7)
            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term


class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=False)
        self.bn = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim), torch.nn.Dropout(dropouts[0]))
        self.tgt_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim), torch.nn.Dropout(dropouts[0]))
        self.src_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim), torch.nn.Dropout(dropouts[0]))
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])
        self.embed_dim = embed_dim
        self.src_domain_K = torch.nn.Linear(16, 16)
        self.src_domain_Q = torch.nn.Linear(16, 16)
        self.src_domain_V = torch.nn.Linear(16, 16)
        self.tgt_domain_K = torch.nn.Linear(16, 16)
        self.tgt_domain_Q = torch.nn.Linear(16, 16)
        self.tgt_domain_V = torch.nn.Linear(16, 16)

    def forward(self, ids, values, seq_lengths, seq_mask, dlabel):
        """
        :param
        ids: the ids of fields (batch_size, seqlength, fields)
        values: the values of fields (batch_size, seqlength, fields)
        seq_length: the length of historical events (batch_size, 1)
        seq_mask: the attention mask for historical events (batch_size, seqlength)
        dlabel: the domain label of the batch samples (batch_size, 1)
        :return
        torch.sigmoid(result.squeeze(1)): the predition of the target payment
        term: the sequence embedding, output of user behavior extractor (batch_size, 32)
        """
        if dlabel == 'src':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            shared_term = self.bn(self.fm(shared_emb[:, :, :].view(batch_size, -1, self.embed_dim)))
            src_emb = self.src_embedding(ids, values)
            src_term = self.src_bn(self.fm(src_emb[:, :, :].view(batch_size, -1, self.embed_dim)))
            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 4)
            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 4)
            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
        elif dlabel == 'tgt':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            shared_term = self.bn(self.fm(shared_emb[:, :, :].view(batch_size, -1, self.embed_dim)))
            tgt_emb = self.tgt_embedding(ids, values)
            tgt_term = self.tgt_bn(self.fm(tgt_emb[:, :, :].view(batch_size, -1, self.embed_dim)))
            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 4)
            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 4)
            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V
            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [(bandwidth * kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class MFSAN(nn.Module):

    def __init__(self, num_classes=31):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet50(True)
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)
        self.sonnet3 = ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):
        mmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)
            data_tgt_son1 = self.sonnet1(data_tgt)
            data_tgt_son1 = self.avgpool(data_tgt_son1)
            data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2)
            data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3)
            data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)
            if mark == 1:
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son1)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2
            if mark == 2:
                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son2)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1) - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1) - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2
            if mark == 3:
                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son3)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1) - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2
        else:
            data = self.sharedNet(data_src)
            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)
            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)
            fea_son3 = self.sonnet3(data)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.cls_fc_son3(fea_son3)
            return pred1, pred2, pred3


class DANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(DANNet, self).__init__()
        self.sharedNet = resnet50(True)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        source = self.cls_fc(source)
        return source, loss


class DDCNet(nn.Module):

    def __init__(self, num_classes=31):
        super(DDCNet, self).__init__()
        self.sharedNet = resnet50(False)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        source = self.sharedNet(source)
        loss = 0
        if self.training == True:
            target = self.sharedNet(target)
            loss = mmd.mmd_linear(source, target)
        source = self.cls_fc(source)
        return source, loss


def CORAL(source, target):
    d = source.data.shape[1]
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    loss = torch.mean(torch.mul(xc - xct, xc - xct))
    loss = loss / (4 * d * 4)
    return loss


class DeepCoral(nn.Module):

    def __init__(self, num_classes=31):
        super(DeepCoral, self).__init__()
        self.sharedNet = resnet50(True)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            loss += CORAL(source, target)
        source = self.cls_fc(source)
        return source, loss


class RevGrad(nn.Module):

    def __init__(self, num_classes=31):
        super(RevGrad, self).__init__()
        self.sharedNet = resnet50(False)
        self.cls_fc = nn.Linear(2048, num_classes)
        self.domain_fc = nn.Linear(2048, 2)

    def forward(self, data):
        data = self.sharedNet(data)
        clabel_pred = self.cls_fc(data)
        dlabel_pred = self.domain_fc(data)
        return clabel_pred, dlabel_pred


class DSAN(nn.Module):

    def __init__(self, num_classes=31, bottle_neck=True):
        super(DSAN, self).__init__()
        self.feature_layers = ResNet.resnet50(True)
        self.lmmd_loss = lmmd.LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_label = self.cls_fc(target)
        loss_lmmd = self.lmmd_loss.get_loss(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        return s_pred, loss_lmmd

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)


class LMMD_loss(nn.Module):

    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [(bandwidth * kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss)
        weight_tt = torch.from_numpy(weight_tt)
        weight_st = torch.from_numpy(weight_st)
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0])
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum
        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr
        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)
        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, num_classes):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.source_fc = nn.Linear(288, num_classes)

    def forward(self, source, target, s_label):
        s_branch1x1 = self.branch1x1(source)
        s_branch5x5 = self.branch5x5_1(source)
        s_branch5x5 = self.branch5x5_2(s_branch5x5)
        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = self.branch3x3dbl_3(s_branch3x3dbl)
        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)
        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch5x5 = self.avg_pool(s_branch5x5)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)
        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)
        s_branch5x5 = s_branch5x5.view(s_branch5x5.size(0), -1)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)
        t_branch1x1 = self.branch1x1(target)
        t_branch5x5 = self.branch5x5_1(target)
        t_branch5x5 = self.branch5x5_2(t_branch5x5)
        t_branch3x3dbl = self.branch3x3dbl_1(target)
        t_branch3x3dbl = self.branch3x3dbl_2(t_branch3x3dbl)
        t_branch3x3dbl = self.branch3x3dbl_3(t_branch3x3dbl)
        t_branch_pool = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        t_branch_pool = self.branch_pool(t_branch_pool)
        t_branch1x1 = self.avg_pool(t_branch1x1)
        t_branch5x5 = self.avg_pool(t_branch5x5)
        t_branch3x3dbl = self.avg_pool(t_branch3x3dbl)
        t_branch_pool = self.avg_pool(t_branch_pool)
        t_branch1x1 = t_branch1x1.view(t_branch1x1.size(0), -1)
        t_branch5x5 = t_branch5x5.view(t_branch5x5.size(0), -1)
        t_branch3x3dbl = t_branch3x3dbl.view(t_branch3x3dbl.size(0), -1)
        t_branch_pool = t_branch_pool.view(t_branch_pool.size(0), -1)
        source = torch.cat([s_branch1x1, s_branch5x5, s_branch3x3dbl, s_branch_pool], 1)
        target = torch.cat([t_branch1x1, t_branch5x5, t_branch3x3dbl, t_branch_pool], 1)
        source = self.source_fc(source)
        t_label = self.source_fc(target)
        t_label = t_label.data.max(1)[1]
        loss = torch.Tensor([0])
        loss = loss
        if self.training == True:
            loss += mmd.cmmd(s_branch1x1, t_branch1x1, s_label, t_label)
            loss += mmd.cmmd(s_branch5x5, t_branch5x5, s_label, t_label)
            loss += mmd.cmmd(s_branch3x3dbl, t_branch3x3dbl, s_label, t_label)
            loss += mmd.cmmd(s_branch_pool, t_branch_pool, s_label, t_label)
        return source, loss


class MRANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(MRANNet, self).__init__()
        self.sharedNet = resnet50(True)
        self.Inception = InceptionA(2048, 64, num_classes)

    def forward(self, source, target, s_label):
        source = self.sharedNet(source)
        target = self.sharedNet(target)
        source, loss = self.Inception(source, target, s_label)
        return source, loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ADDneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizationMachine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Meta_Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiLayerPerceptron,
     lambda: ([], {'input_dim': 4, 'embed_dims': [4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (WideAndDeepModel,
     lambda: ([], {'field_dims': [4, 4], 'embed_dim': 4, 'mlp_dims': [4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_easezyc_deep_transfer_learning(_paritybench_base):
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

