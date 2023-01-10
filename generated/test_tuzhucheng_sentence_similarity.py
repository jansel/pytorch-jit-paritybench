import sys
_module = sys.modules[__name__]
del sys
datasets = _module
sick = _module
wikiqa = _module
main = _module
metrics = _module
pearson_correlation = _module
retrieval_metrics = _module
spearman_correlation = _module
models = _module
bimpm = _module
mpcnn = _module
mpcnn_lite = _module
sentence_embedding_baseline = _module
runners = _module
train = _module
utils = _module
hyperband = _module
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


import torch


from torch.autograd import Variable


import torch.nn as nn


import math


import numpy as np


from torchtext.data import interleave_keys


import logging


import random


import torch.optim as O


import scipy.stats as stats


import uuid


import torch.nn.functional as F


from collections import defaultdict


from sklearn.decomposition import TruncatedSVD


class BiMPM(nn.Module):

    def __init__(self, word_embedding, n_word_dim, n_char_dim, n_perspectives, n_hidden_units, n_classes, dropout=0.1):
        super(BiMPM, self).__init__()
        self.word_embedding = word_embedding
        self.n_word_dim = n_word_dim
        self.n_char_dim = n_char_dim
        self.n_perspectives = n_perspectives
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes
        self.dropout = dropout
        self.d = n_word_dim
        self.l = n_perspectives
        self.context_representation_lstm = nn.LSTM(input_size=self.d, hidden_size=n_hidden_units, num_layers=1, batch_first=True, dropout=dropout, bidirectional=True)
        self.m_full_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_full_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_maxpool_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_maxpool_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_attn_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_attn_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_maxattn_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.m_maxattn_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))
        self.aggregation_lstm = nn.LSTM(input_size=8 * self.l, hidden_size=n_hidden_units, num_layers=1, batch_first=True, dropout=dropout, bidirectional=True)
        self.prediction_layer = nn.Sequential(nn.Linear(4 * self.n_hidden_units, 2 * self.n_hidden_units), nn.Tanh(), nn.Dropout(dropout), nn.Linear(2 * self.n_hidden_units, self.n_classes), nn.LogSoftmax(1))

    def matching_strategy_full(self, v1, v2, W):
        """
        :param v1: batch x seq_len x n_hidden
        :param v2: batch x n_hidden (FULL) or batch x seq_len x n_hidden (ATTENTIVE)
        :param W:  l x n_hidden
        :return: batch x seq_len x l
        """
        l = W.size(0)
        batch_size = v1.size(0)
        seq_len = v1.size(1)
        v1 = v1.unsqueeze(2).expand(-1, -1, l, -1)
        W_expanded = W.expand(batch_size, seq_len, -1, -1)
        Wv1 = W_expanded.mul(v1)
        if len(v2.size()) == 2:
            v2 = v2.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, l, -1)
        elif len(v2.size()) == 3:
            v2 = v2.unsqueeze(2).expand(-1, -1, l, -1)
        else:
            raise ValueError(f'Invalid v2 tensor size {v2.size()}')
        Wv2 = W_expanded.mul(v2)
        cos_sim = F.cosine_similarity(Wv1, Wv2, dim=3)
        return cos_sim

    def matching_strategy_pairwise(self, v1, v2, W):
        """
        Used as a subroutine for (2) Maxpooling-Matching
        :param v1: batch x seq_len_1 x n_hidden
        :param v2: batch x seq_len_2 x n_hidden
        :param W: l x n_hidden
        :return: batch x seq_len_1 x seq_len_2 x l
        """
        l = W.size(0)
        batch_size = v1.size(0)
        v1_expanded = v1.unsqueeze(1).expand(-1, l, -1, -1)
        W1_expanded = W.unsqueeze(1).expand(batch_size, -1, v1.size(1), -1)
        Wv1 = W1_expanded.mul(v1_expanded)
        v2_expanded = v2.unsqueeze(1).expand(-1, l, -1, -1)
        W2_expanded = W.unsqueeze(1).expand(batch_size, -1, v2.size(1), -1)
        Wv2 = W2_expanded.mul(v2_expanded)
        dot = torch.matmul(Wv1, Wv2.transpose(3, 2))
        v1_norm = v1_expanded.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2_expanded.norm(p=2, dim=3, keepdim=True)
        norm_product = torch.matmul(v1_norm, v2_norm.transpose(3, 2))
        cosine_matrix = dot / norm_product
        cosine_matrix = cosine_matrix.permute(0, 2, 3, 1)
        return cosine_matrix

    def matching_strategy_attention(self, v1, v2):
        """
        Used as a subroutine for (3) Attentive-Matching
        :param v1: batch x seq_len_1 x n_hidden
        :param v2: batch x seq_len_2 x n_hidden
        :return: batch x seq_len_1 x seq_len_2
        """
        dot = torch.bmm(v1, v2.transpose(2, 1))
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True)
        norm_product = torch.bmm(v1_norm, v2_norm.transpose(2, 1))
        return dot / norm_product

    def forward(self, batch):
        sent1 = self.word_embedding(batch.sentence_a)
        sent2 = self.word_embedding(batch.sentence_b)
        s1_context_out, (s1_context_h, s1_context_c) = self.context_representation_lstm(sent1)
        s2_context_out, (s2_context_h, s2_context_c) = self.context_representation_lstm(sent2)
        s1_context_forward, s1_context_backward = torch.split(s1_context_out, self.n_hidden_units, dim=2)
        s2_context_forward, s2_context_backward = torch.split(s2_context_out, self.n_hidden_units, dim=2)
        m_full_s1_f = self.matching_strategy_full(s1_context_forward, s2_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s1_b = self.matching_strategy_full(s1_context_backward, s2_context_backward[:, 0, :], self.m_full_backward_W)
        m_full_s2_f = self.matching_strategy_full(s2_context_forward, s1_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s2_b = self.matching_strategy_full(s2_context_backward, s1_context_backward[:, 0, :], self.m_full_backward_W)
        m_pair_f = self.matching_strategy_pairwise(s1_context_forward, s2_context_backward, self.m_maxpool_forward_W)
        m_pair_b = self.matching_strategy_pairwise(s1_context_backward, s2_context_backward, self.m_maxpool_backward_W)
        m_maxpool_s1_f, _ = m_pair_f.max(dim=2)
        m_maxpool_s1_b, _ = m_pair_b.max(dim=2)
        m_maxpool_s2_f, _ = m_pair_f.max(dim=1)
        m_maxpool_s2_b, _ = m_pair_b.max(dim=1)
        cosine_f = self.matching_strategy_attention(s1_context_forward, s2_context_forward)
        cosine_b = self.matching_strategy_attention(s1_context_backward, s2_context_backward)
        attn_s1_f = s1_context_forward.unsqueeze(2) * cosine_f.unsqueeze(3)
        attn_s1_b = s1_context_forward.unsqueeze(2) * cosine_b.unsqueeze(3)
        attn_s2_f = s2_context_forward.unsqueeze(1) * cosine_f.unsqueeze(3)
        attn_s2_b = s2_context_forward.unsqueeze(1) * cosine_b.unsqueeze(3)
        attn_mean_vec_s2_f = attn_s1_f.sum(dim=1) / cosine_f.sum(1, keepdim=True).transpose(2, 1)
        attn_mean_vec_s2_b = attn_s1_b.sum(dim=1) / cosine_b.sum(1, keepdim=True).transpose(2, 1)
        attn_mean_vec_s1_f = attn_s2_f.sum(dim=2) / cosine_f.sum(2, keepdim=True)
        attn_mean_vec_s1_b = attn_s2_b.sum(dim=2) / cosine_b.sum(2, keepdim=True)
        m_attn_s1_f = self.matching_strategy_full(s1_context_forward, attn_mean_vec_s1_f, self.m_attn_forward_W)
        m_attn_s1_b = self.matching_strategy_full(s1_context_backward, attn_mean_vec_s1_b, self.m_attn_forward_W)
        m_attn_s2_f = self.matching_strategy_full(s2_context_forward, attn_mean_vec_s2_f, self.m_attn_forward_W)
        m_attn_s2_b = self.matching_strategy_full(s2_context_backward, attn_mean_vec_s2_b, self.m_attn_forward_W)
        attn_max_vec_s2_f, _ = attn_s1_f.max(dim=1)
        attn_max_vec_s2_b, _ = attn_s1_b.max(dim=1)
        attn_max_vec_s1_f, _ = attn_s2_f.max(dim=2)
        attn_max_vec_s1_b, _ = attn_s2_b.max(dim=2)
        m_maxattn_s1_f = self.matching_strategy_full(s1_context_forward, attn_max_vec_s1_f, self.m_maxattn_forward_W)
        m_maxattn_s1_b = self.matching_strategy_full(s1_context_backward, attn_max_vec_s1_b, self.m_maxattn_forward_W)
        m_maxattn_s2_f = self.matching_strategy_full(s2_context_forward, attn_max_vec_s2_f, self.m_maxattn_forward_W)
        m_maxattn_s2_b = self.matching_strategy_full(s2_context_backward, attn_max_vec_s2_b, self.m_maxattn_forward_W)
        s1_combined_match_vec = torch.cat([m_full_s1_f, m_maxpool_s1_f, m_attn_s1_f, m_maxattn_s1_f, m_full_s1_b, m_maxpool_s1_b, m_attn_s1_b, m_maxattn_s1_b], dim=2)
        s2_combined_match_vec = torch.cat([m_full_s2_f, m_maxpool_s2_f, m_attn_s2_f, m_maxattn_s2_f, m_full_s2_b, m_maxpool_s2_b, m_attn_s2_b, m_maxattn_s2_b], dim=2)
        s1_agg_out, (s1_agg_h, s1_agg_c) = self.aggregation_lstm(s1_combined_match_vec)
        s2_agg_out, (s2_agg_h, s2_agg_c) = self.aggregation_lstm(s2_combined_match_vec)
        matching_vector = torch.cat([s1_agg_h.transpose(1, 0), s2_agg_h.transpose(1, 0)], dim=1).view(-1, 4 * self.n_hidden_units)
        final_pred = self.prediction_layer(matching_vector)
        return final_pred


class MPCNN(nn.Module):

    def __init__(self, embedding, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout):
        super(MPCNN, self).__init__()
        self.embedding = embedding
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        holistic_conv_layers = []
        per_dim_conv_layers = []
        self.in_channels = n_word_dim
        for ws in filter_widths:
            if np.isinf(ws):
                continue
            holistic_conv_layers.append(nn.Sequential(nn.Conv1d(self.in_channels, n_holistic_filters, ws), nn.Tanh()))
            per_dim_conv_layers.append(nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels * n_per_dim_filters, ws, groups=self.in_channels), nn.Tanh()))
        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)
        self.per_dim_conv_layers = nn.ModuleList(per_dim_conv_layers)
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + n_holistic_filters, 2 + self.in_channels, 2
        n_feat_h = 3 * len(self.filter_widths) * COMP_2_COMPONENTS
        n_feat_v = 3 * (len(self.filter_widths) - 1) ** 2 * COMP_1_COMPONENTS_HOLISTIC + 3 * 3 + 2 * (len(self.filter_widths) - 1) * n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        n_feat = n_feat_h + n_feat_v
        self.final_layers = nn.Sequential(nn.Linear(n_feat, hidden_layer_units), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_layer_units, num_classes), nn.LogSoftmax(1))

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1), 'min': F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1), 'mean': F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)}
                continue
            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent)
            block_a[ws] = {'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters), 'min': F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters), 'mean': F.avg_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)}
            per_dim_conv_out = self.per_dim_conv_layers[ws - 1](sent)
            block_b[ws] = {'max': F.max_pool1d(per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters), 'min': F.max_pool1d(-1 * per_dim_conv_out, per_dim_conv_out.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)}
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                batch_size = x1.size()[0]
                comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                batch_size = x1.size()[0]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if not np.isinf(ws1) and not np.isinf(ws2) or np.isinf(ws1) and np.isinf(ws2):
                        comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                        comparison_feats.append(F.pairwise_distance(x1, x2))
                        comparison_feats.append(torch.abs(x1 - x2))
        for pool in ('max', 'min'):
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, i]
                    x2 = oG_2B[:, :, i]
                    batch_size = x1.size()[0]
                    comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                    comparison_feats.append(F.pairwise_distance(x1, x2))
                    comparison_feats.append(torch.abs(x1 - x2))
        return torch.cat(comparison_feats, dim=1)

    def forward(self, batch):
        sent1 = self.embedding(batch.sentence_a).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_b).transpose(1, 2)
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        combined_feats = [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)
        preds = self.final_layers(feat_all)
        return preds


class MPCNNLite(nn.Module):

    def __init__(self, embedding, n_word_dim, n_holistic_filters, filter_widths, hidden_layer_units, num_classes, dropout):
        super(MPCNNLite, self).__init__()
        self.embedding = embedding
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.filter_widths = filter_widths
        holistic_conv_layers = []
        self.in_channels = n_word_dim
        for ws in filter_widths:
            if np.isinf(ws):
                continue
            holistic_conv_layers.append(nn.Sequential(nn.Conv1d(self.in_channels, n_holistic_filters, ws), nn.Tanh()))
        self.holistic_conv_layers = nn.ModuleList(holistic_conv_layers)
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + n_holistic_filters, 2
        n_feat_h = len(self.filter_widths) * COMP_2_COMPONENTS
        n_feat_v = (len(self.filter_widths) - 1) ** 2 * COMP_1_COMPONENTS_HOLISTIC + 3
        n_feat = n_feat_h + n_feat_v
        self.final_layers = nn.Sequential(nn.Linear(n_feat, hidden_layer_units), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_layer_units, num_classes), nn.LogSoftmax(1))

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)}
                continue
            holistic_conv_out = self.holistic_conv_layers[ws - 1](sent)
            block_a[ws] = {'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size(2)).contiguous().view(-1, self.n_holistic_filters)}
        return block_a

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max',):
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool]
                x2 = sent2_block_a[ws][pool]
                batch_size = x1.size()[0]
                comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                comparison_feats.append(F.pairwise_distance(x1, x2))
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in ('max',):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                batch_size = x1.size()[0]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if not np.isinf(ws1) and not np.isinf(ws2) or np.isinf(ws1) and np.isinf(ws2):
                        comparison_feats.append(F.cosine_similarity(x1, x2).contiguous().view(batch_size, 1))
                        comparison_feats.append(F.pairwise_distance(x1, x2))
                        comparison_feats.append(torch.abs(x1 - x2))
        return torch.cat(comparison_feats, dim=1)

    def forward(self, batch):
        sent1 = self.embedding(batch.sentence_a).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_b).transpose(1, 2)
        sent1_block_a = self._get_blocks_for_sentence(sent1)
        sent2_block_a = self._get_blocks_for_sentence(sent2)
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a)
        combined_feats = [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)
        preds = self.final_layers(feat_all)
        return preds


class SmoothInverseFrequencyBaseline(nn.Module):

    def __init__(self, n_out, alpha, embedding, remove_special_direction=True, frequency_dataset='enwiki', supervised=True):
        super(SmoothInverseFrequencyBaseline, self).__init__()
        self.n_out = n_out
        self.alpha = alpha
        self.embedding = embedding
        self.remove_special_direction = remove_special_direction
        self.frequency_dataset = frequency_dataset
        self.supervised = supervised
        self.unigram_prob = defaultdict(int)
        self.word_vec_dim = self.embedding.weight.size(1)
        self.classifier = nn.Sequential(nn.Linear(2 * self.word_vec_dim, 2400), nn.Tanh(), nn.Linear(2400, n_out), nn.LogSoftmax(1))

    def populate_word_frequency_estimation(self, data_loader):
        """
        Computing and storing the unigram probability.
        """
        if self.frequency_dataset == 'enwiki':
            with open('./data/enwiki_vocab_min200.txt') as f:
                for line in f:
                    word, freq = line.split(' ')
                    self.unigram_prob[word] = int(freq)
        else:
            for batch_idx, batch in enumerate(data_loader):
                for sent_a, sent_b in zip(batch.raw_sentence_a, batch.raw_sentence_b):
                    for w in sent_a:
                        self.unigram_prob[w] += 1
                    for w in sent_b:
                        self.unigram_prob[w] += 1
        total_words = sum(self.unigram_prob.values())
        for word, count in self.unigram_prob.items():
            self.unigram_prob[word] = count / total_words

    def _compute_sentence_embedding_as_weighted_sum(self, sentence, sentence_embedding):
        """
        Compute sentence embedding as weighted sum of word vectors of its individual words.
        :param sentence: Tokenized sentence
        :param sentence_embedding: A 2D tensor where dim 0 is the word vector dimension and dim 1 is the words
        :return: A vector that is the weighted word vectors of the words in the sentence
        """
        weights = [(self.alpha / (self.unigram_prob.get(w, 0) + self.alpha)) for w in sentence]
        weights.extend([0.0] * (sentence_embedding.size(1) - len(weights)))
        weights = sentence_embedding.data.float().new(weights).expand_as(sentence_embedding.data)
        return (Variable(weights) * sentence_embedding).sum(1)

    def _remove_projection_on_first_principle_component(self, batch_sentence_embedding):
        """
        Remove the projection onto the first principle component of the sentences from each sentence embedding.
        See https://plot.ly/ipython-notebooks/principal-component-analysis/ for a nice tutorial on PCA.
        Follows official implementation at https://github.com/PrincetonML/SIF
        :param batch_sentence_embedding: A group of sentence embeddings (a 2D tensor, each row is a separate
        sentence and each column is a feature of a sentence)
        :return: A new batch sentence embedding with the projection removed
        """
        svd = TruncatedSVD(n_components=1, n_iter=7)
        X = batch_sentence_embedding.data.cpu().numpy()
        svd.fit(X)
        pc = Variable(batch_sentence_embedding.data.float().new(svd.components_))
        new_embedding = batch_sentence_embedding - batch_sentence_embedding.matmul(pc.transpose(0, 1)).matmul(pc)
        return new_embedding

    def cosine_similarity(self, sentence_embedding_a, sentence_embedding_b):
        """
        Compute cosine similarity of sentence embedding a and sentence embedding b.
        Each row is the sentence embedding (vector) for a sentence.
        """
        dot_product = (sentence_embedding_a * sentence_embedding_b).sum(1)
        norm_a = sentence_embedding_a.norm(p=2, dim=1)
        norm_b = sentence_embedding_b.norm(p=2, dim=1)
        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim

    def compute_sentence_embedding(self, batch):
        sentence_embedding_a = Variable(batch.sentence_a.data.float().new(batch.sentence_a.size(0), self.word_vec_dim).zero_())
        sentence_embedding_b = Variable(batch.sentence_b.data.float().new(batch.sentence_b.size(0), self.word_vec_dim).zero_())
        for i, (raw_sent_a, raw_sent_b, sent_a_idx, sent_b_idx) in enumerate(zip(batch.raw_sentence_a, batch.raw_sentence_b, batch.sentence_a, batch.sentence_b)):
            sent_a = self.embedding(sent_a_idx).transpose(0, 1)
            sent_b = self.embedding(sent_b_idx).transpose(0, 1)
            sentence_embedding_a[i] = self._compute_sentence_embedding_as_weighted_sum(raw_sent_a, sent_a)
            sentence_embedding_b[i] = self._compute_sentence_embedding_as_weighted_sum(raw_sent_b, sent_b)
        if self.remove_special_direction:
            sentence_embedding_a = self._remove_projection_on_first_principle_component(sentence_embedding_a)
            sentence_embedding_b = self._remove_projection_on_first_principle_component(sentence_embedding_b)
        return sentence_embedding_a, sentence_embedding_b

    def forward(self, batch):
        if len(self.unigram_prob) == 0:
            raise ValueError('Word frequency lookup dictionary is not populated. Did you call populate_word_frequency_estimation?')
        sentence_embedding_a, sentence_embedding_b = self.compute_sentence_embedding(batch)
        if self.supervised:
            elem_wise_product = sentence_embedding_a * sentence_embedding_b
            abs_diff = torch.abs(sentence_embedding_a - sentence_embedding_b)
            concat_input = torch.cat([elem_wise_product, abs_diff], dim=1)
            scores = self.classifier(concat_input)
        else:
            scores = self.cosine_similarity(sentence_embedding_a, sentence_embedding_b)
        return scores

