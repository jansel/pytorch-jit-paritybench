import sys
_module = sys.modules[__name__]
del sys
data_utils = _module
infer_example = _module
infer_example_bert_models = _module
layers = _module
attention = _module
dynamic_rnn = _module
point_wise_feed_forward = _module
squeeze_embedding = _module
models = _module
aen = _module
aoa = _module
atae_lstm = _module
bert_spc = _module
cabasc = _module
ian = _module
lcf_bert = _module
lstm = _module
memnet = _module
mgan = _module
ram = _module
tc_lstm = _module
td_lstm = _module
tnet_lf = _module
train = _module
train_k_fold_cross_val = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn.functional as F


import numpy as np


import math


import torch.nn as nn


import copy


import numpy


import logging


from time import strftime


from time import localtime


import random


from torch.utils.data import DataLoader


from torch.utils.data import random_split


from torch.utils.data import ConcatDataset


class Attention(nn.Module):

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1,
        score_function='dot_product', dropout=0):
        """ Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        """
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.
            hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.
            hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class DynamicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
        batch_first=True, dropout=0, bidirectional=False,
        only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=
                hidden_size, num_layers=num_layers, bias=bias, batch_first=
                batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=
                hidden_size, num_layers=num_layers, bias=bias, batch_first=
                batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=
                hidden_size, num_layers=num_layers, bias=bias, batch_first=
                batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len,
            batch_first=self.batch_first)
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)
        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack,
                batch_first=self.batch_first)
            out = out[0]
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid=None, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class SqueezeEmbedding(nn.Module):
    """
    Squeeze sequence embedding length to the longest one in the batch
    """

    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len,
            batch_first=self.batch_first)
        """unpack: out"""
        out = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=
            self.batch_first)
        out = out[0]
        """unsort"""
        out = out[x_unsort_idx]
        return out


class CrossEntropyLoss_LSR(nn.Module):

    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += 1.0 - self.para_LSR
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_GloVe(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(AEN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim,
            n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim,
            n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.
            dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.
            dropout)
        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function=
            'mlp', dropout=opt.dropout)
        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)
        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)
        s1, _ = self.attn_s1(hc, ht)
        context_len = torch.tensor(context_len, dtype=torch.float)
        target_len = torch.tensor(target_len, dtype=torch.float)
        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(
            context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(
            target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(
            context_len.size(0), 1))
        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out


class AEN_BERT(nn.Module):

    def __init__(self, bert, opt):
        super(AEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim,
            n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim,
            n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.
            dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.
            dropout)
        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function=
            'mlp', dropout=opt.dropout)
        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def forward(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context)
        context = self.dropout(context)
        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target)
        target = self.dropout(target)
        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)
        s1, _ = self.attn_s1(hc, ht)
        context_len = torch.tensor(context_len, dtype=torch.float)
        target_len = torch.tensor(target_len, dtype=torch.float)
        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(
            context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(
            target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(
            context_len.size(0), 1))
        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out


class AOA(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        aspect_indices = inputs[1]
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_raw_indices)
        asp = self.embed(aspect_indices)
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len)
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)
        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2))
        alpha = F.softmax(interaction_mat, dim=1)
        beta = F.softmax(interaction_mat, dim=2)
        beta_avg = beta.mean(dim=1, keepdim=True)
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2))
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma
            ).squeeze(-1)
        out = self.dense(weighted_sum)
        return out


class NoQueryAttention(Attention):
    """q is a parameter"""

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1,
        score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim,
            out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1.0 / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)


class ATAE_LSTM(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim,
            num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim + opt.embed_dim,
            score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.tensor(torch.sum(aspect_indices != 0, dim=-1),
            dtype=torch.float)
        x = self.embed(text_raw_indices)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_indices)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(
            aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)
        out = self.dense(output)
        return out


class BERT_SPC(nn.Module):

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits


class Cabasc(nn.Module):

    def __init__(self, embedding_matrix, opt, _type='c'):
        super(Cabasc, self).__init__()
        self.opt = opt
        self.type = _type
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.linear1 = nn.Linear(3 * opt.embed_dim, opt.embed_dim)
        self.linear2 = nn.Linear(opt.embed_dim, 1, bias=False)
        self.mlp = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.rnn_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=
            1, batch_first=True, rnn_type='GRU')
        self.rnn_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=
            1, batch_first=True, rnn_type='GRU')
        self.mlp_l = nn.Linear(opt.hidden_dim, 1)
        self.mlp_r = nn.Linear(opt.hidden_dim, 1)

    def context_attention(self, x_l, x_r, memory, memory_len, aspect_len):
        left_len, right_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r !=
            0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        context_l, (_, _) = self.rnn_l(x_l, left_len)
        context_r, (_, _) = self.rnn_r(x_r, right_len)
        attn_l = torch.sigmoid(self.mlp_l(context_l)) + 0.5
        attn_r = torch.sigmoid(self.mlp_r(context_r)) + 0.5
        for i in range(memory.size(0)):
            aspect_start = (left_len[i] - aspect_len[i]).item()
            aspect_end = left_len[i]
            for idx in range(memory_len[i]):
                if idx < aspect_start:
                    memory[i][idx] *= attn_l[i][idx]
                elif idx < aspect_end:
                    memory[i][idx] *= (attn_l[i][idx] + attn_r[i][idx -
                        aspect_start]) / 2
                else:
                    memory[i][idx] *= attn_r[i][idx - aspect_start]
        return memory

    def locationed_memory(self, memory, memory_len):
        """
        # differ from description in paper here, but may be better
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i] - aspect_len[i]
                aspect_end = left_len[i] 
                if idx < aspect_start: l = aspect_start.item() - idx                   
                elif idx <= aspect_end: l = 0 
                else: l = idx - aspect_end.item()
                memory[i][idx] *= (1-float(l)/int(memory_len[i]))
        """
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= 1 - float(idx) / int(memory_len[i])
        return memory

    def forward(self, inputs):
        text_raw_indices, aspect_indices, x_l, x_r = inputs[0], inputs[1
            ], inputs[2], inputs[3]
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = aspect_len.float()
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        v_a = torch.div(aspect, nonzeros_aspect.unsqueeze(1)).unsqueeze(1)
        memory = self.embed(text_raw_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        nonzeros_memory = memory_len.float()
        v_s = torch.sum(memory, dim=1)
        v_s = torch.div(v_s, nonzeros_memory.unsqueeze(1)).unsqueeze(1)
        if self.type == 'c':
            memory = self.locationed_memory(memory, memory_len)
        elif self.type == 'cabasc':
            memory = self.context_attention(x_l, x_r, memory, memory_len,
                aspect_len)
            v_s = torch.sum(memory, dim=1)
            v_s = torch.div(v_s, nonzeros_memory.unsqueeze(1))
            v_s = v_s.unsqueeze(dim=1)
        """
        # no multi-hop, but may be better. 
        # however, here is something totally different from what paper depits
        for _ in range(self.opt.hops):  
            #x = self.x_linear(x)
            v_ts, _ = self.attention(memory, v_a)
        """
        memory_chunks = memory.chunk(memory.size(1), dim=1)
        c = []
        for memory_chunk in memory_chunks:
            c_i = self.linear1(torch.cat([memory_chunk, v_a, v_s], dim=1).
                view(memory_chunk.size(0), -1))
            c_i = self.linear2(torch.tanh(c_i))
            c.append(c_i)
        alpha = F.softmax(torch.cat(c, dim=1), dim=1)
        v_ts = torch.matmul(memory.transpose(1, 2), alpha.unsqueeze(-1)
            ).transpose(1, 2)
        v_ns = v_ts + v_s
        v_ns = v_ns.view(v_ns.size(0), -1)
        v_ms = torch.tanh(self.mlp(v_ns))
        out = self.dense(v_ms)
        return out


class IAN(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function=
            'bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function=
            'bi_linear')
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)
        aspect_len = torch.tensor(aspect_len, dtype=torch.float)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.
            size(0), 1))
        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(
            text_raw_len.size(0), 1))
        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)
        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out


class SelfAttention(nn.Module):

    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt
            .max_seq_len), dtype=np.float32), dtype=torch.float32)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_BERT(nn.Module):

    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()
        self.bert_spc = bert
        self.opt = opt
        self.bert_local = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self
            .opt.max_seq_len, self.opt.bert_dim), dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros(self.opt.
                    bert_dim, dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len
                ):
                masked_text_raw_indices[text_i][j] = np.zeros(self.opt.
                    bert_dim, dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self
            .opt.max_seq_len, self.opt.bert_dim), dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.
                float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index) + asp_len / 
                        2 - self.opt.SRD) / np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[
                    text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]
        bert_spc_out, _ = self.bert_spc(text_bert_indices, bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)
        bert_local_out, _ = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)
        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(
                text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)
        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(
                text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out,
                weighted_text_local_features)
        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        return dense_out


class LSTM(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1,
            batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out


class MemNet(nn.Module):

    def locationed_memory(self, memory, memory_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1 - float(idx + 1) / memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = torch.tensor(weight)
        memory = weight.unsqueeze(2) * memory
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MemNet, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float)
        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.
            size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.opt.hops):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out


class LocationEncoding(nn.Module):

    def __init__(self, opt):
        super(LocationEncoding, self).__init__()
        self.opt = opt

    def forward(self, x, pos_inx):
        batch_size, seq_len = x.size()[0], x.size()[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][0]):
                relative_pos = pos_inx[i][0] - j
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
            for j in range(pos_inx[i][0], pos_inx[i][1] + 1):
                weight[i].append(0)
            for j in range(pos_inx[i][1] + 1, seq_len):
                relative_pos = j - pos_inx[i][1]
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
        weight = torch.tensor(weight)
        return weight


class AlignmentMatrix(nn.Module):

    def __init__(self, opt):
        super(AlignmentMatrix, self).__init__()
        self.opt = opt
        self.w_u = nn.Parameter(torch.Tensor(6 * opt.hidden_dim, 1))

    def forward(self, batch_size, ctx, asp):
        ctx_len = ctx.size(1)
        asp_len = asp.size(1)
        alignment_mat = torch.zeros(batch_size, ctx_len, asp_len)
        ctx_chunks = ctx.chunk(ctx_len, dim=1)
        asp_chunks = asp.chunk(asp_len, dim=1)
        for i, ctx_chunk in enumerate(ctx_chunks):
            for j, asp_chunk in enumerate(asp_chunks):
                feat = torch.cat([ctx_chunk, asp_chunk, ctx_chunk *
                    asp_chunk], dim=2)
                alignment_mat[:, (i), (j)] = feat.matmul(self.w_u.expand(
                    batch_size, -1, -1)).squeeze(-1).squeeze(-1)
        return alignment_mat


class MGAN(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(MGAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)
        self.location = LocationEncoding(opt)
        self.w_a2c = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.
            hidden_dim))
        self.w_c2a = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.
            hidden_dim))
        self.alignment = AlignmentMatrix(opt)
        self.dense = nn.Linear(8 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        aspect_indices = inputs[1]
        text_left_indices = inputs[2]
        batch_size = text_raw_indices.size(0)
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        left_len = torch.sum(text_left_indices != 0, dim=-1)
        aspect_in_text = torch.cat([left_len.unsqueeze(-1), (left_len +
            asp_len - 1).unsqueeze(-1)], dim=-1)
        ctx = self.embed(text_raw_indices)
        asp = self.embed(aspect_indices)
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len)
        ctx_out = self.location(ctx_out, aspect_in_text)
        ctx_pool = torch.sum(ctx_out, dim=1)
        ctx_pool = torch.div(ctx_pool, ctx_len.float().unsqueeze(-1)
            ).unsqueeze(-1)
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)
        asp_pool = torch.sum(asp_out, dim=1)
        asp_pool = torch.div(asp_pool, asp_len.float().unsqueeze(-1)
            ).unsqueeze(-1)
        alignment_mat = self.alignment(batch_size, ctx_out, asp_out)
        f_asp2ctx = torch.matmul(ctx_out.transpose(1, 2), F.softmax(
            alignment_mat.max(2, keepdim=True)[0], dim=1)).squeeze(-1)
        f_ctx2asp = torch.matmul(F.softmax(alignment_mat.max(1, keepdim=
            True)[0], dim=2), asp_out).transpose(1, 2).squeeze(-1)
        c_asp2ctx_alpha = F.softmax(ctx_out.matmul(self.w_a2c.expand(
            batch_size, -1, -1)).matmul(asp_pool), dim=1)
        c_asp2ctx = torch.matmul(ctx_out.transpose(1, 2), c_asp2ctx_alpha
            ).squeeze(-1)
        c_ctx2asp_alpha = F.softmax(asp_out.matmul(self.w_c2a.expand(
            batch_size, -1, -1)).matmul(ctx_pool), dim=1)
        c_ctx2asp = torch.matmul(asp_out.transpose(1, 2), c_ctx2asp_alpha
            ).squeeze(-1)
        feat = torch.cat([c_asp2ctx, f_asp2ctx, f_ctx2asp, c_ctx2asp], dim=1)
        out = self.dense(feat)
        return out


class RAM(nn.Module):

    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        left_len = left_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        u = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(left_len[i]):
                weight[i].append(1 - (left_len[i] - idx) / memory_len[i])
                u[i].append(idx - left_len[i])
            for idx in range(left_len[i], left_len[i] + aspect_len[i]):
                weight[i].append(1)
                u[i].append(0)
            for idx in range(left_len[i] + aspect_len[i], memory_len[i]):
                weight[i].append(1 - (idx - left_len[i] - aspect_len[i] + 1
                    ) / memory_len[i])
                u[i].append(idx - left_len[i] - aspect_len[i] + 1)
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
                u[i].append(0)
        u = torch.tensor(u, dtype=memory.dtype).unsqueeze(2)
        weight = torch.tensor(weight).unsqueeze(2)
        v = memory * weight
        memory = torch.cat([v, u], dim=2)
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True)
        self.att_linear = nn.Linear(opt.hidden_dim * 2 + 1 + opt.embed_dim *
            2, 1)
        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2 + 1, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, text_left_indices = inputs[0
            ], inputs[1], inputs[2]
        left_len = torch.sum(text_left_indices != 0, dim=-1)
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = aspect_len.float()
        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len,
            aspect_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        et = torch.zeros_like(aspect)
        batch_size = memory.size(0)
        seq_len = memory.size(1)
        for _ in range(self.opt.hops):
            g = self.att_linear(torch.cat([memory, torch.zeros(batch_size,
                seq_len, self.opt.embed_dim) + et.unsqueeze(1), torch.zeros
                (batch_size, seq_len, self.opt.embed_dim) + aspect.
                unsqueeze(1)], dim=-1))
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), memory).squeeze(1)
            et = self.gru_cell(i, et)
        out = self.dense(et)
        return out


class TC_LSTM(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(TC_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim,
            num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim,
            num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        x_l, x_r, target = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0,
            dim=-1)
        target_len = torch.sum(target != 0, dim=-1, dtype=torch.float)[:, (
            None), (None)]
        x_l, x_r, target = self.embed(x_l), self.embed(x_r), self.embed(target)
        v_target = torch.div(target.sum(dim=1, keepdim=True), target_len)
        x_l = torch.cat((x_l, torch.cat([v_target] * x_l.shape[1], 1)), 2)
        x_r = torch.cat((x_r, torch.cat([v_target] * x_r.shape[1], 1)), 2)
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out


class TD_LSTM(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(TD_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers
            =1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers
            =1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0,
            dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out


class Absolute_Position_Embedding(nn.Module):

    def __init__(self, opt, size=None, mode='sum'):
        self.opt = opt
        self.size = size
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def forward(self, x, pos_inx):
        if self.size is None or self.mode == 'sum':
            self.size = int(x.size(-1))
        batch_size, seq_len = x.size()[0], x.size()[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)
        weight = torch.tensor(weight)
        return weight


class TNet_LF(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(TNet_LF, self).__init__()
        None
        self.embed = nn.Embedding.from_pretrained(torch.tensor(
            embedding_matrix, dtype=torch.float))
        self.position = Absolute_Position_Embedding(opt)
        self.opt = opt
        D = opt.embed_dim
        C = opt.polarities_dim
        L = opt.max_seq_len
        HD = opt.hidden_dim
        self.lstm1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=
            1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=
            1, batch_first=True, bidirectional=True)
        self.convs3 = nn.Conv1d(2 * HD, 50, 3, padding=1)
        self.fc1 = nn.Linear(4 * HD, 2 * HD)
        self.fc = nn.Linear(50, C)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, aspect_in_text = inputs[0], inputs[1
            ], inputs[2]
        feature_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        feature = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = v.transpose(1, 2)
        e = e.transpose(1, 2)
        for i in range(2):
            a = torch.bmm(e.transpose(1, 2), v)
            a = F.softmax(a, 1)
            aspect_mid = torch.bmm(e, a)
            aspect_mid = torch.cat((aspect_mid, v), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            v = aspect_mid + v
            v = self.position(v.transpose(1, 2), aspect_in_text).transpose(1, 2
                )
        z = F.relu(self.convs3(v))
        z = F.max_pool1d(z, z.size(2)).squeeze(2)
        out = self.fc(z)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_songyouwei_ABSA_PyTorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AOA(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, hidden_dim=4, polarities_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_001(self):
        self._check(ATAE_LSTM(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, hidden_dim=4, polarities_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_002(self):
        self._check(Attention(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 1, 4]), torch.rand([4, 4, 1, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Cabasc(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, polarities_dim=4, hidden_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_004(self):
        self._check(CrossEntropyLoss_LSR(*[], **{'device': 0}), [torch.rand([4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_005(self):
        self._check(DynamicLSTM(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_006(self):
        self._check(IAN(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, hidden_dim=4, polarities_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_007(self):
        self._check(LSTM(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, hidden_dim=4, polarities_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_008(self):
        self._check(MemNet(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, polarities_dim=4, hops=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_009(self):
        self._check(NoQueryAttention(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 1, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(PositionwiseFeedForward(*[], **{'d_hid': 4}), [torch.rand([4, 4, 4])], {})

    def test_011(self):
        self._check(SqueezeEmbedding(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_012(self):
        self._check(TC_LSTM(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, hidden_dim=4, polarities_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_013(self):
        self._check(TD_LSTM(*[], **{'embedding_matrix': torch.rand([4, 4]), 'opt': _mock_config(embed_dim=4, hidden_dim=4, polarities_dim=4)}), [torch.zeros([4, 4, 4], dtype=torch.int64)], {})

