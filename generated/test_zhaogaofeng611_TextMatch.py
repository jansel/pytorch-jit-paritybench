import sys
_module = sys.modules[__name__]
del sys
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
layers = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
layers = _module
model = _module
test = _module
train = _module
utils = _module
RE2 = _module
data = _module
model = _module
modules = _module
alignment = _module
connection = _module
embedding = _module
encoder = _module
fusion = _module
pooling = _module
prediction = _module
test = _module
train = _module
util = _module
loader = _module
logger = _module
metrics = _module
params = _module
registry = _module
vocab = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
data = _module
model = _module
test = _module
train = _module
utils = _module
args = _module
data_utils = _module
gen_corpus = _module
lcqmc_dataset = _module
load_data = _module
train_w2v = _module

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


import re


import numpy as np


import pandas as pd


import torch


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


import time


from sklearn.metrics import roc_auc_score


from torch import nn


from typing import Collection


import math


import torch.nn.functional as f


from functools import partial


def match_score(s1, s2, mask1, mask2):
    """
    s1, s2:  batch_size * seq_len  * dim
    """
    batch, seq_len, dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


class Wide_Conv(nn.Module):

    def __init__(self, seq_len, embeds_size, device='gpu'):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.W
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sent1, sent2, mask1, mask2):
        """
        sent1, sent2: batch_size * seq_len * dim
        """
        A = match_score(sent1, sent2, mask1, mask2)
        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2


def attention_avg_pooling(sent1, sent2, mask1, mask2):
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2


class ABCNN(nn.Module):

    def __init__(self, embeddings, num_layer=1, linear_size=300, max_length=50, device='gpu'):
        super(ABCNN, self).__init__()
        self.device = device
        self.embeds_dim = embeddings.shape[1]
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embed.float()
        self.embed.weight.requires_grad = True
        self.embed
        self.linear_size = linear_size
        self.num_layer = num_layer
        self.conv = nn.ModuleList([Wide_Conv(max_length, embeddings.shape[1], device) for _ in range(self.num_layer)])
        self.fc = nn.Sequential(nn.Linear(self.embeds_dim * (1 + self.num_layer) * 2, self.linear_size), nn.LayerNorm(self.linear_size), nn.ReLU(inplace=True), nn.Linear(self.linear_size, 2))

    def forward(self, q1, q2):
        mask1, mask2 = q1.eq(0), q2.eq(0)
        res = [[], []]
        q1_encode = self.embed(q1)
        q2_encode = self.embed(q2)
        res[0].append(F.avg_pool1d(q1_encode.transpose(1, 2), kernel_size=q1_encode.size(1)).squeeze(-1))
        res[1].append(F.avg_pool1d(q2_encode.transpose(1, 2), kernel_size=q2_encode.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(q1_encode, q2_encode, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            q1_encode, q2_encode = o1 + q1_encode, o2 + q2_encode
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        sim = self.fc(x)
        probabilities = nn.functional.softmax(sim, dim=-1)
        return sim, probabilities


class AlbertModel(nn.Module):

    def __init__(self):
        super(AlbertModel, self).__init__()
        self.albert = AlbertForSequenceClassification.from_pretrained('voidful/albert_chinese_base', num_labels=2)
        self.device = torch.device('cuda')
        for param in self.albert.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class AlbertModelTest(nn.Module):

    def __init__(self):
        super(AlbertModelTest, self).__init__()
        config = AlbertConfig.from_pretrained('models/config.json')
        self.albert = AlbertForSequenceClassification(config)
        self.device = torch.device('cuda')

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


def div_with_small_value(n, d, eps=1e-08):
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d


def attention(v1, v2):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm
    return div_with_small_value(a, d)


def mp_matching_func(v1, v2, w, l=20):
    """
    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (l, hidden_size)
    :return: (batch, l)
    """
    seq_len = v1.size(1)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    v1 = w * torch.stack([v1] * l, dim=3)
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * l, dim=3)
    else:
        v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * l, dim=3)
    m = F.cosine_similarity(v1, v2, dim=2)
    return m


def mp_matching_func_pairwise(v1, v2, w, l=20):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (l, hidden_size)
    :return: (batch, l, seq_len1, seq_len2)
    """
    w = w.unsqueeze(0).unsqueeze(2)
    v1, v2 = w * torch.stack([v1] * l, dim=1), w * torch.stack([v2] * l, dim=1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)
    return m


class BIMPM(nn.Module):

    def __init__(self, embeddings, hidden_size=100, num_perspective=20, class_size=2, device='gpu'):
        super(BIMPM, self).__init__()
        self.embeds_dim = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.l = num_perspective
        self.word_emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True
        self.word_emb
        self.class_size = class_size
        self.device = device
        self.context_LSTM = nn.LSTM(input_size=self.embeds_dim, hidden_size=self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(self.l, self.hidden_size)))
        self.aggregation_LSTM = nn.LSTM(input_size=self.l * 8, hidden_size=self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.pred_fc1 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.pred_fc2 = nn.Linear(self.hidden_size * 2, self.class_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.word_emb.weight.data[0], -0.1, 0.1)
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)
        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=0.1, training=self.training)

    def forward(self, q1, q2):
        p_encode = self.word_emb(q1)
        h_endoce = self.word_emb(q2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        con_p, _ = self.context_LSTM(p_encode)
        con_h, _ = self.context_LSTM(h_endoce)
        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)
        con_p_fw, con_p_bw = torch.split(con_p, self.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.hidden_size, dim=-1)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1, self.l)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2, self.l)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1, self.l)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2, self.l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3, self.l)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4, self.l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)
        mv_p = torch.cat([mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw, mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat([mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw, mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)
        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)
        x = torch.cat([agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2), agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)], dim=1)
        x = self.dropout(x)
        x = torch.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities


class BertModel(nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=2)
        self.device = torch.device('cuda')
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModelTest(nn.Module):

    def __init__(self):
        super(BertModelTest, self).__init__()
        config = BertConfig.from_pretrained('models/config.json')
        self.bert = BertForSequenceClassification(config)
        self.device = torch.device('cuda')

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class EmbedingDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask


class DecomposableAttention(nn.Module):

    def __init__(self, embeddings, f_in_dim=200, f_hid_dim=200, f_out_dim=200, dropout=0.2, embedd_dim=300, num_classes=2, device='gpu'):
        super(DecomposableAttention, self).__init__()
        self.device = device
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embed.float()
        self.embed.weight.requires_grad = True
        self.embed
        self.project_embedd = nn.Linear(embedd_dim, f_in_dim)
        self.F = nn.Sequential(nn.Dropout(0.2), nn.Linear(f_in_dim, f_hid_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(f_hid_dim, f_out_dim), nn.ReLU())
        self.G = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * f_in_dim, f_hid_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(f_hid_dim, f_out_dim), nn.ReLU())
        self.H = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * f_in_dim, f_hid_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(f_hid_dim, f_out_dim), nn.ReLU())
        self.last_layer = nn.Linear(f_out_dim, num_classes)

    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = generate_sent_masks(q1, q1_lengths)
        q2_mask = generate_sent_masks(q2, q2_lengths)
        q1_embed = self.embed(q1)
        q2_embed = self.embed(q2)
        q1_encoded = self.project_embedd(q1_embed)
        q2_encoded = self.project_embedd(q2_embed)
        attend_out1 = self.F(q1_encoded)
        attend_out2 = self.F(q2_encoded)
        similarity_matrix = attend_out1.bmm(attend_out2.transpose(2, 1).contiguous())
        prem_hyp_attn = masked_softmax(similarity_matrix, q2_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), q1_mask)
        q1_aligned = weighted_sum(q2_encoded, prem_hyp_attn, q1_mask)
        q2_aligned = weighted_sum(q1_encoded, hyp_prem_attn, q2_mask)
        compare_i = torch.cat((q1_encoded, q1_aligned), dim=2)
        compare_j = torch.cat((q2_encoded, q2_aligned), dim=2)
        v1_i = self.G(compare_i)
        v2_j = self.G(compare_j)
        v1_sum = torch.sum(v1_i, dim=1)
        v2_sum = torch.sum(v2_j, dim=1)
        output_tolast = self.H(torch.cat((v1_sum, v2_sum), dim=1))
        logits = self.last_layer(output_tolast)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities


class DistilBertModel(nn.Module):

    def __init__(self):
        super(DistilBertModel, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained('adamlin/bert-distil-chinese', num_labels=2)
        self.device = torch.device('cuda')
        for param in self.distilbert.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.distilbert(input_ids=batch_seqs, attention_mask=batch_seq_masks, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class DistilBertModelTest(nn.Module):

    def __init__(self):
        super(DistilBertModelTest, self).__init__()
        config = DistilBertConfig.from_pretrained('models/config.json')
        self.distilbert = DistilBertForSequenceClassification(config)
        self.device = torch.device('cuda')

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.distilbert(input_ids=batch_seqs, attention_mask=batch_seq_masks, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


class Seq2SeqEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        """rnn_type must be a class inheriting from torch.nn.RNNBase"""
        assert issubclass(rnn_type, nn.RNNBase)
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = rnn_type(input_size, hidden_size, num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if platform == 'linux' or platform == 'linux2':
            reordered_outputs = outputs.index_select(0, restoration_idx)
        else:
            reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses


def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


class ESIM(nn.Module):

    def __init__(self, hihdden_size, embeddings=None, dropout=0.5, num_classes=2, device='gpu'):
        super(ESIM, self).__init__()
        self.embedding_dim = embeddings.shape[1]
        self.hidden_size = hihdden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.word_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.word_embedding.float()
        self.word_embedding.weight.requires_grad = True
        self.word_embedding
        if self.dropout:
            self.rnn_dropout = RNNDropout(p=self.dropout)
        self.first_rnn = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size), nn.ReLU())
        self.attention = SoftmaxAttention()
        self.second_rnn = Seq2SeqEncoder(nn.LSTM, self.hidden_size, self.hidden_size, bidirectional=True)
        self.classification = nn.Sequential(nn.Linear(2 * 4 * self.hidden_size, self.hidden_size), nn.ReLU(), nn.Dropout(p=self.dropout), nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(p=self.dropout), nn.Linear(self.hidden_size // 2, self.num_classes))

    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = get_mask(q1, q1_lengths)
        q2_mask = get_mask(q2, q2_lengths)
        q1_embed = self.word_embedding(q1)
        q2_embed = self.word_embedding(q2)
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)
        q1_encoded = self.first_rnn(q1_embed, q1_lengths)
        q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)
        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(q2_mask, dim=1, keepdim=True)
        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -10000000.0).max(dim=1)
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -10000000.0).max(dim=1)
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)
        logits = self.classification(merged)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities


class GeLU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class Conv1d(Module):

    def __init__(self, in_channels, out_channels, kernel_sizes: Collection[int]):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2.0 / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)


class Encoder(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.encoders = nn.ModuleList([Conv1d(in_channels=input_size if i == 0 else args.hidden_size, out_channels=args.hidden_size, kernel_sizes=args.kernel_sizes) for i in range(args.enc_layers)])

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        mask = mask.transpose(1, 2)
        for i, encoder in enumerate(self.encoders):
            r_mask = ~mask
            r_mask = r_mask.bool()
            x.masked_fill_(r_mask, 0.0)
            if i > 0:
                x = f.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = f.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)


class ModuleDict(nn.ModuleDict):

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for key, module in self.items():
            if hasattr(module, 'get_summary'):
                name = base_name + key
                summary.update(module.get_summary(name))
        return summary


class ModuleList(nn.ModuleList):

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for i, module in enumerate(self):
            if hasattr(module, 'get_summary'):
                name = base_name + str(i)
                summary.update(module.get_summary(name))
        return summary


class Pooling(nn.Module):

    def forward(self, x, mask):
        r_mask = ~mask
        r_mask = r_mask.bool()
        return x.masked_fill_(r_mask, -float('inf')).max(dim=1)[0]


class RE2(Module):

    def __init__(self, args, embeddings, device='gpu'):
        super().__init__()
        self.dropout = args.dropout
        self.device = device
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embedding.float()
        self.embedding.weight.requires_grad = True
        self.embedding
        self.blocks = ModuleList([ModuleDict({'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size), 'alignment': alignment[args.alignment](args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2), 'fusion': fusion[args.fusion](args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2)}) for i in range(args.blocks)])
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, a, b):
        mask_a = torch.ne(a, 0).unsqueeze(2)
        mask_b = torch.ne(b, 0).unsqueeze(2)
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        logits = self.prediction(a, b)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities


class Module(nn.Module):

    def __init__(self):
        super().__init__()
        self.summary = {}

    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        if self.summary:
            summary.update({(base_name + name): val for name, val in self.summary.items()})
        for name, child in self.named_children():
            if hasattr(child, 'get_summary'):
                name = base_name + name
                summary.update(child.get_summary(name))
        return summary


class Linear(nn.Module):

    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2.0 if activations else 1.0) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


registry = {}


class Alignment(Module):

    def __init__(self, args, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)))

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        r_mask = ~mask
        r_mask = r_mask.bool()
        attn.masked_fill_(r_mask, -10000000.0)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b)
        return feature_a, feature_b


class MappedAlignment(Alignment):

    def __init__(self, args, input_size):
        super().__init__(args, input_size)
        self.projection = nn.Sequential(nn.Dropout(args.dropout), Linear(input_size, args.hidden_size, activations=True))

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)


class NullConnection(nn.Module):

    def forward(self, x, _, __):
        return x


class Residual(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.embedding_dim, args.hidden_size)

    def forward(self, x, res, i):
        if i == 1:
            res = self.linear(res)
        return (x + res) * math.sqrt(0.5)


class AugmentedResidual(nn.Module):

    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)


class Embedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fix_embeddings = args.fix_embeddings
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=0)
        self.dropout = args.dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict({'weight': torch.tensor(value)})

    def forward(self, x):
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x


class Fusion(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.fusion = Linear(input_size * 2, args.hidden_size, activations=True)

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align], dim=-1))


class FullFusion(nn.Module):

    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.fusion1 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion = Linear(args.hidden_size * 3, args.hidden_size, activations=True)

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = f.dropout(x, self.dropout, self.training)
        return self.fusion(x)


class Prediction(nn.Module):

    def __init__(self, args, inp_features=2):
        super().__init__()
        self.dense = nn.Sequential(nn.Dropout(args.dropout), Linear(args.hidden_size * inp_features, args.hidden_size, activations=True), nn.Dropout(args.dropout), Linear(args.hidden_size, args.num_classes))

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


class AdvancedPrediction(Prediction):

    def __init__(self, args):
        super().__init__(args, inp_features=4)

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


class SymmetricPrediction(AdvancedPrediction):

    def forward(self, a, b):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))


class SiaGRU(nn.Module):

    def __init__(self, embeddings, hidden_size=300, num_layer=2, device='gpu'):
        super(SiaGRU, self).__init__()
        self.device = device
        self.embeds_dim = embeddings.shape[1]
        self.word_emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True
        self.word_emb
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0
        self.pred_fc = nn.Linear(50, 2)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.gru(x)
        return output

    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)

    def forward(self, sent1, sent2):
        p_encode = self.word_emb(sent1)
        h_endoce = self.word_emb(sent2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        encoding1 = self.forward_once(p_encode)
        encoding2 = self.forward_once(h_endoce)
        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        x = self.pred_fc(sim.squeeze(dim=-1))
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities


class XlnetModel(nn.Module):

    def __init__(self):
        super(XlnetModel, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained('hfl/chinese-xlnet-base', num_labels=2)
        self.device = torch.device('cuda')
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class XlnetModelTest(nn.Module):

    def __init__(self):
        super(XlnetModelTest, self).__init__()
        config = XLNetConfig.from_pretrained('models/config.json')
        self.xlnet = XLNetForSequenceClassification(config)
        self.device = torch.device('cuda')

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids=batch_seqs, attention_mask=batch_seq_masks, token_type_ids=batch_seq_segments, labels=labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdvancedPrediction,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Alignment,
     lambda: ([], {'args': _mock_config(hidden_size=4), '__': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (AugmentedResidual,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), 0], {}),
     True),
    (EmbedingDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FullFusion,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4), 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Fusion,
     lambda: ([], {'args': _mock_config(hidden_size=4), 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GeLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MappedAlignment,
     lambda: ([], {'args': _mock_config(hidden_size=4, dropout=0.5), 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NullConnection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Prediction,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RNNDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'args': _mock_config(embedding_dim=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (SoftmaxAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (SymmetricPrediction,
     lambda: ([], {'args': _mock_config(dropout=0.5, hidden_size=4, num_classes=4)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Wide_Conv,
     lambda: ([], {'seq_len': 4, 'embeds_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_zhaogaofeng611_TextMatch(_paritybench_base):
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

