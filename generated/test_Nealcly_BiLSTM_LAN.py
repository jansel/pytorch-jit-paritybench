import sys
_module = sys.modules[__name__]
del sys
main = _module
main_parse = _module
model = _module
charbigru = _module
charbilstm = _module
charcnn = _module
crf = _module
lstm_attention = _module
seqmodel = _module
wordrep = _module
wordsequence = _module
utils = _module
alphabet = _module
data = _module
functions = _module
metric = _module
tagSchemeConverter = _module

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


import time


import random


import torch


import torch.autograd as autograd


import torch.nn as nn


import torch.optim as optim


import numpy as np


import copy


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.autograd import *


class CharBiGRU(nn.Module):

    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu, bidirect_flag=True):
        super(CharBiGRU, self).__init__()
        None
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_lstm = nn.GRU(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirect_flag)
        if self.gpu:
            self.char_drop = self.char_drop
            self.char_embeddings = self.char_embeddings
            self.char_lstm = self.char_lstm

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[(index), :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class CharBiLSTM(nn.Module):

    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu, bidirect_flag=True):
        super(CharBiLSTM, self).__init__()
        None
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        if bidirect_flag:
            self.hidden_dim = hidden_dim // 2
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirect_flag)
        if self.gpu:
            self.char_drop = self.char_drop
            self.char_embeddings = self.char_embeddings
            self.char_lstm = self.char_lstm

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[(index), :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_hidden = None
        pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
        char_rnn_out, char_hidden = self.char_lstm(pack_input, char_hidden)
        char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        return char_rnn_out.transpose(1, 0)

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


class CharCNN(nn.Module):

    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout, gpu):
        super(CharCNN, self).__init__()
        None
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrain_char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.char_cnn = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        if self.gpu:
            self.char_drop = self.char_drop
            self.char_embeddings = self.char_embeddings
            self.char_cnn = self.char_cnn

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[(index), :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, input, seq_lengths):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2, 1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)


START_TAG = -2


STOP_TAG = -1


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):

    def __init__(self, tagset_size, gpu):
        super(CRF, self).__init__()
        None
        self.gpu = gpu
        self.tagset_size = tagset_size
        init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        init_transitions[:, (START_TAG)] = -10000.0
        init_transitions[(STOP_TAG), :] = -10000.0
        init_transitions[:, (0)] = -10000.0
        init_transitions[(0), :] = -10000.0
        if self.gpu:
            init_transitions = init_transitions
        self.transitions = nn.Parameter(init_transitions)

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)
        partition = inivalues[:, (START_TAG), :].clone().view(batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            mask_idx = mask[(idx), :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, (STOP_TAG)]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)
        partition = inivalues[:, (START_TAG), :].clone().view(batch_size, tag_size)
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        pointer = last_bp[:, (STOP_TAG)]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, (0)] = (tag_size - 2) * tag_size + tags[:, (0)]
            else:
                new_tags[:, (idx)] = tags[:, (idx - 1)] * tag_size + tags[:, (idx)]
        end_transition = self.transitions[:, (STOP_TAG)].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def _viterbi_decode_nbest(self, feats, mask, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert tag_size == self.tagset_size + 2
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)
        partition = inivalues[:, (START_TAG), :].clone()
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            else:
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size) + partition.contiguous().view(batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, tag_size, nbest).transpose(1, 0).contiguous()
        last_position = length_mask.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.transitions.view(1, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        end_bp = end_bp.transpose(2, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size, nbest)).long()
        if self.gpu:
            pad_zero = pad_zero
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)
        pointer = end_bp[:, (STOP_TAG), :]
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1, 0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        """
        back_points: in simple demonstratration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        """
        back_points = back_points.transpose(1, 0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size, nbest))
        if self.gpu:
            decode_idx = decode_idx
        decode_idx[-1] = pointer.data / nbest
        for idx in range(len(back_points) - 2, -1, -1):
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size * nbest), 1, pointer.contiguous().view(batch_size, nbest))
            decode_idx[idx] = new_pointer.data / nbest
            pointer = new_pointer + pointer.contiguous().view(batch_size, nbest) * mask[idx].view(batch_size, 1).expand(batch_size, nbest).long()
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        scores = end_partition[:, :, (STOP_TAG)]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        return path_score, decode_idx


class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        """Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        """
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        if self.gpu:
            self.Q_proj = self.Q_proj
            self.K_proj = self.K_proj
            self.V_proj = self.V_proj
        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values, last_layer=False):
        Q = self.Q_proj(queries)
        K = self.K_proj(keys)
        V = self.V_proj(values)
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))
        outputs = outputs / K_.size()[-1] ** 0.5
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])
        outputs = outputs * query_masks
        outputs = self.output_dropout(outputs)
        if last_layer == True:
            return outputs
        outputs = torch.bmm(outputs, V_)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)
        outputs += queries
        return outputs


class LSTM_attention(nn.Module):
    """ Compose with two layers """

    def __init__(self, lstm_hidden, bilstm_flag, data):
        super(LSTM_attention, self).__init__()
        self.lstm = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bilstm_flag)
        self.label_attn = multihead_attention(data.HP_hidden_dim, num_heads=data.num_attention_head, dropout_rate=data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.gpu = data.HP_gpu
        if self.gpu:
            self.lstm = self.lstm
            self.label_attn = self.label_attn

    def forward(self, lstm_out, label_embs, word_seq_lengths, hidden):
        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        label_attention_output = self.label_attn(lstm_out, label_embs, label_embs)
        lstm_out = torch.cat([lstm_out, label_attention_output], -1)
        return lstm_out


class WordRep(nn.Module):

    def __init__(self, data):
        super(WordRep, self).__init__()
        None
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.char_all_feature = False
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_feature_extractor == 'CNN':
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == 'LSTM':
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == 'GRU':
                self.char_feature = CharBiGRU(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == 'ALL':
                self.char_all_feature = True
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
                self.char_feature_extra = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                None
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        self.label_dim = data.HP_hidden_dim
        self.label_embedding = nn.Embedding(data.label_alphabet_size, self.label_dim)
        self.label_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding_label(data.label_alphabet_size, self.label_dim, data.label_embedding_scale)))
        self.feature_num = data.feature_num
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()
        for idx in range(self.feature_num):
            self.feature_embeddings.append(nn.Embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]))
        for idx in range(self.feature_num):
            if data.pretrain_feature_embeddings[idx] is not None:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[idx]))
            else:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(self.random_embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx])))
        if self.gpu:
            self.drop = self.drop
            self.word_embedding = self.word_embedding
            self.label_embedding = self.label_embedding
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx]

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[(index), :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def random_embedding_label(self, vocab_size, embedding_dim, scale):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        for index in range(vocab_size):
            pretrain_emb[(index), :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, input_label_seq_tensor):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
                input_label_seq_tensor: (batch_size, number of label)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embedding(word_inputs)
        word_list = [word_embs]
        for idx in range(self.feature_num):
            word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        label_embs = self.label_embedding(input_label_seq_tensor)
        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_list.append(char_features)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size, sent_len, -1)
                word_list.append(char_features_extra)
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent, label_embs


class WordSequence(nn.Module):

    def __init__(self, data):
        super(WordSequence, self).__init__()
        None
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.num_of_lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == 'ALL':
                self.input_size += data.HP_char_hidden_dim
        for idx in range(data.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.word_feature_extractor = data.word_feature_extractor
        self.lstm_first = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=self.bilstm_flag)
        self.lstm_layer = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=1, batch_first=True, bidirectional=self.bilstm_flag)
        self.self_attention_first = multihead_attention(data.HP_hidden_dim, num_heads=data.num_attention_head, dropout_rate=data.HP_dropout, gpu=self.gpu)
        self.self_attention_last = multihead_attention(data.HP_hidden_dim, num_heads=1, dropout_rate=0, gpu=self.gpu)
        self.lstm_attention_stack = nn.ModuleList([LSTM_attention(lstm_hidden, self.bilstm_flag, data) for _ in range(int(self.num_of_lstm_layer) - 2)])
        if self.gpu:
            self.droplstm = self.droplstm
            self.lstm_first = self.lstm_first
            self.lstm_layer = self.lstm_layer
            self.self_attention_first = self.self_attention_first
            self.self_attention_last = self.self_attention_last
            self.lstm_attention_stack = self.lstm_attention_stack

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, input_label_seq_tensor):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
                label_size: nubmer of label
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_represent, label_embs = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, input_label_seq_tensor)
        """
        First LSTM layer (input word only)
        """
        lstm_out = word_represent
        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm_first(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        attention_label = self.self_attention_first(lstm_out, label_embs, label_embs)
        lstm_out = torch.cat([lstm_out, attention_label], -1)
        for layer in self.lstm_attention_stack:
            lstm_out = layer(lstm_out, label_embs, word_seq_lengths, hidden)
        """
        Last Layer 
        Attention weight calculate loss
        """
        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm_layer(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        lstm_out = self.self_attention_last(lstm_out, label_embs, label_embs, True)
        return lstm_out


class SeqModel(nn.Module):

    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        None
        None
        if data.use_char:
            None
        None
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        label_size = data.label_alphabet_size
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, input_label_seq_tensor):
        outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, input_label_seq_tensor)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, input_label_seq_tensor):
        outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, input_label_seq_tensor)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        outs = outs.view(batch_size * seq_len, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long() * tag_seq
        return tag_seq


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (multihead_attention,
     lambda: ([], {'num_units': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_Nealcly_BiLSTM_LAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

