import sys
_module = sys.modules[__name__]
del sys
annotate_graphs = _module
evaluation = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
eval = _module
meteor = _module
legacy_meteor = _module
rouge = _module
layers = _module
attention = _module
common = _module
graphs = _module
model = _module
model_handler = _module
models = _module
graph2seq = _module
seq2seq = _module
utils = _module
bert_utils = _module
constants = _module
data_utils = _module
eval_utils = _module
generic_utils = _module
io_utils = _module
logger = _module
padding_utils = _module
timer = _module
vocab_utils = _module
main = _module
run_eval = _module

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


import torch.nn as nn


from typing import List


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import torch.nn.functional as F


import random


import numpy as np


from collections import Counter


import torch.optim as optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


import time


from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn


import string


from typing import Union


from collections import defaultdict


from collections import namedtuple


import re


from scipy.sparse import *


INF = 1e+20


class Context2AnswerAttention(nn.Module):

    def __init__(self, dim, hidden_size):
        super(Context2AnswerAttention, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, answers, out_answers, ans_mask=None):
        """
        Parameters
        :context, (batch_size, L, dim)
        :answers, (batch_size, N, dim)
        :out_answers, (batch_size, N, dim)
        :ans_mask, (batch_size, N)

        Returns
        :ques_emb, (batch_size, L, dim)
        """
        context_fc = torch.relu(self.linear_sim(context))
        questions_fc = torch.relu(self.linear_sim(answers))
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))
        if ans_mask is not None:
            attention = attention.masked_fill_(1 - ans_mask.byte().unsqueeze(1), -INF)
        prob = torch.softmax(attention, dim=-1)
        ques_emb = torch.matmul(prob, out_answers)
        return ques_emb


class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W1 = torch.Tensor(input_size, hidden_size)
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(self.W1))
        self.W2 = torch.Tensor(hidden_size, 1)
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))

    def forward(self, x, attention_mask=None):
        attention = torch.mm(torch.tanh(torch.mm(x.view(-1, x.size(-1)), self.W1)), self.W2).view(x.size(0), -1)
        if attention_mask is not None:
            attention = attention.masked_fill_(1 - attention_mask.byte(), -INF)
        probs = torch.softmax(attention, dim=-1).unsqueeze(1)
        weighted_x = torch.bmm(probs, x).squeeze(1)
        return weighted_x


class Attention(nn.Module):

    def __init__(self, hidden_size, h_state_embed_size=None, in_memory_embed_size=None, attn_type='simple'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        if not h_state_embed_size:
            h_state_embed_size = hidden_size
        if not in_memory_embed_size:
            in_memory_embed_size = hidden_size
        if attn_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform_(self.W))
            if attn_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform_(self.W3))
        elif attn_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

    def forward(self, query_embed, in_memory_embed, attn_mask=None, addition_vec=None):
        if self.attn_type == 'simple':
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'mul':
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'add':
            attention = torch.mm(in_memory_embed.view(-1, in_memory_embed.size(-1)), self.W2).view(in_memory_embed.size(0), -1, self.W2.size(-1)) + torch.mm(query_embed, self.W).unsqueeze(1)
            if addition_vec is not None:
                attention = attention + addition_vec
            attention = torch.tanh(attention)
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))
        if attn_mask is not None:
            attention = attn_mask * attention - (1 - attn_mask) * INF
        return attention


class GatedFusion(nn.Module):

    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        """GatedFusion module"""
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state * input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state


class GRUStep(nn.Module):

    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        """GRU module"""
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or not training:
        return x
    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1.0 - drop_prob).div_(1.0 - drop_prob)
    mask = mask.expand_as(x)
    return x * mask


def to_cuda(x, device=None):
    if device:
        x = x
    return x


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=False, num_layers=1, rnn_type='lstm', rnn_dropout=None, device=None):
        super(EncoderRNN, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            None
        else:
            None
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.device = device
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """x: [batch_size * max_length * emb_dim]
           x_len: [batch_size]
        """
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)
        h0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions * self.num_layers, x_len.size(0), self.hidden_size), self.device)
            packed_h, (packed_h_t, packed_c_t) = self.model(x, (h0, c0))
        else:
            packed_h, packed_h_t = self.model(x, h0)
        if self.num_directions == 2:
            packed_h_t = torch.cat((packed_h_t[-1], packed_h_t[-2]), 1)
            if self.rnn_type == 'lstm':
                packed_c_t = torch.cat((packed_c_t[-1], packed_c_t[-2]), 1)
        else:
            packed_h_t = packed_h_t[-1]
            if self.rnn_type == 'lstm':
                packed_c_t = packed_c_t[-1]
        _, inverse_indx = torch.sort(indx, 0)
        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        restore_hh = hh[inverse_indx]
        restore_hh = dropout(restore_hh, self.rnn_dropout, shared_axes=[-2], training=self.training)
        restore_hh = restore_hh.transpose(0, 1)
        restore_packed_h_t = packed_h_t[inverse_indx]
        restore_packed_h_t = dropout(restore_packed_h_t, self.rnn_dropout, training=self.training)
        restore_packed_h_t = restore_packed_h_t.unsqueeze(0)
        if self.rnn_type == 'lstm':
            restore_packed_c_t = packed_c_t[inverse_indx]
            restore_packed_c_t = dropout(restore_packed_c_t, self.rnn_dropout, training=self.training)
            restore_packed_c_t = restore_packed_c_t.unsqueeze(0)
            rnn_state_t = restore_packed_h_t, restore_packed_c_t
        else:
            rnn_state_t = restore_packed_h_t
        return restore_hh, rnn_state_t


VERY_SMALL_NUMBER = 1e-31


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, *, rnn_type='lstm', enc_attn=True, dec_attn=True, enc_attn_cover=True, pointer=True, tied_embedding=None, out_embed_size=None, in_drop: float=0, rnn_drop: float=0, out_drop: float=0, enc_hidden_size=None, device=None):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.in_drop = in_drop
        self.out_drop = out_drop
        self.rnn_drop = rnn_drop
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn
        self.enc_attn_cover = enc_attn_cover
        self.pointer = pointer
        self.out_embed_size = out_embed_size
        if tied_embedding is not None and self.out_embed_size and embed_size != self.out_embed_size:
            None
            self.out_embed_size = embed_size
        model = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.model = model(embed_size, self.hidden_size)
        if enc_attn:
            self.fc_dec_input = nn.Linear(enc_hidden_size + embed_size, embed_size)
            if not enc_hidden_size:
                enc_hidden_size = self.hidden_size
            self.enc_attn_fn = Attention(self.hidden_size, 2 * self.hidden_size, self.hidden_size, attn_type='add')
            self.combined_size += enc_hidden_size
            if enc_attn_cover:
                self.cover_weight = torch.Tensor(1, 1, self.hidden_size)
                self.cover_weight = nn.Parameter(nn.init.xavier_uniform_(self.cover_weight))
        if dec_attn:
            self.dec_attn_fn = Attention(self.hidden_size, 2 * self.hidden_size, self.hidden_size, attn_type='add')
            self.combined_size += self.hidden_size
        if pointer:
            self.ptr = nn.Linear(self.combined_size + embed_size + self.hidden_size, 1)
        if tied_embedding is not None and embed_size != self.combined_size:
            self.out_embed_size = embed_size
        if self.out_embed_size:
            self.pre_out = nn.Linear(self.combined_size, self.out_embed_size, bias=False)
            size_before_output = self.out_embed_size
        else:
            size_before_output = self.combined_size
        self.out = nn.Linear(size_before_output, vocab_size, bias=False)
        if tied_embedding is not None:
            self.out.weight = tied_embedding.weight

    def forward(self, embedded, rnn_state, encoder_hiddens=None, decoder_hiddens=None, coverage_vector=None, *, input_mask=None, encoder_word_idx=None, ext_vocab_size: int=None, log_prob: bool=True, prev_enc_context=None):
        """
    :param embedded: (batch size, embed size)
    :param rnn_state: LSTM: ((1, batch size, decoder hidden size), (1, batch size, decoder hidden size)), GRU:(1, batch size, decoder hidden size)
    :param encoder_hiddens: (src seq len, batch size, hidden size), for attention mechanism
    :param decoder_hiddens: (past dec steps, batch size, hidden size), for attention mechanism
    :param encoder_word_idx: (batch size, src seq len), for pointer network
    :param ext_vocab_size: the dynamic word_vocab size, determined by the max num of OOV words contained
                           in any src seq in this batch, for pointer network
    :param log_prob: return log probability instead of probability
    :return: tuple of four things:
             1. word prob or log word prob, (batch size, dynamic word_vocab size);
             2. rnn_state, RNN hidden (and/or ceil) state after this step, (1, batch size, decoder hidden size);
             3. attention weights over encoder states, (batch size, src seq len);
             4. prob of copying by pointing as opposed to generating, (batch size, 1)

    Perform single-step decoding.
    """
        batch_size = embedded.size(0)
        combined = to_cuda(torch.zeros(batch_size, self.combined_size), self.device)
        embedded = dropout(embedded, self.in_drop, training=self.training)
        if self.enc_attn:
            if prev_enc_context is None:
                prev_enc_context = to_cuda(torch.zeros(batch_size, encoder_hiddens.size(-1)), self.device)
            dec_input_emb = self.fc_dec_input(torch.cat([embedded, prev_enc_context], -1))
        else:
            dec_input_emb = embedded
        output, rnn_state = self.model(dec_input_emb.unsqueeze(0), rnn_state)
        output = dropout(output, self.rnn_drop, training=self.training)
        if self.rnn_type == 'lstm':
            rnn_state = tuple([dropout(x, self.rnn_drop, training=self.training) for x in rnn_state])
            hidden = torch.cat(rnn_state, -1).squeeze(0)
        else:
            rnn_state = dropout(rnn_state, self.rnn_drop, training=self.training)
            hidden = rnn_state.squeeze(0)
        combined[:, :self.hidden_size] = output.squeeze(0)
        offset = self.hidden_size
        enc_attn, prob_ptr = None, None
        if self.enc_attn or self.pointer:
            num_enc_steps = encoder_hiddens.size(0)
            enc_total_size = encoder_hiddens.size(2)
            if self.enc_attn_cover and coverage_vector is not None:
                addition_vec = coverage_vector.unsqueeze(-1) * self.cover_weight
            else:
                addition_vec = None
            enc_energy = self.enc_attn_fn(hidden, encoder_hiddens.transpose(0, 1).contiguous(), attn_mask=input_mask, addition_vec=addition_vec).transpose(0, 1).unsqueeze(-1)
            enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)
            if self.enc_attn:
                enc_context = torch.bmm(encoder_hiddens.permute(1, 2, 0), enc_attn).squeeze(2)
                combined[:, offset:offset + enc_total_size] = enc_context
                offset += enc_total_size
            else:
                enc_context = None
            enc_attn = enc_attn.squeeze(2)
        if self.dec_attn:
            if decoder_hiddens is not None and len(decoder_hiddens) > 0:
                dec_energy = self.dec_attn_fn(hidden, decoder_hiddens.transpose(0, 1).contiguous()).transpose(0, 1).unsqueeze(-1)
                dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
                dec_context = torch.bmm(decoder_hiddens.permute(1, 2, 0), dec_attn)
                combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
            offset += self.hidden_size
        if self.out_embed_size:
            out_embed = torch.tanh(self.pre_out(combined))
        else:
            out_embed = combined
        out_embed = dropout(out_embed, self.out_drop, training=self.training)
        logits = self.out(out_embed)
        if self.pointer:
            output = to_cuda(torch.zeros(batch_size, ext_vocab_size), self.device)
            pgen_cat = [embedded, hidden]
            if self.enc_attn:
                pgen_cat.append(enc_context)
            if self.dec_attn:
                pgen_cat.append(dec_context)
            prob_ptr = torch.sigmoid(self.ptr(torch.cat(pgen_cat, -1)))
            prob_gen = 1 - prob_ptr
            gen_output = F.softmax(logits, dim=1)
            output[:, :self.vocab_size] = prob_gen * gen_output
            ptr_output = enc_attn
            output.scatter_add_(1, encoder_word_idx, prob_ptr * ptr_output)
            if log_prob:
                output = torch.log(output + VERY_SMALL_NUMBER)
        elif log_prob:
            output = F.log_softmax(logits, dim=1)
        else:
            output = F.softmax(logits, dim=1)
        return output, rnn_state, enc_attn, prob_ptr, enc_context


class GraphLearner(nn.Module):

    def __init__(self, input_size, hidden_size, topk=10, num_pers=16, device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.linear_sim = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, context, ctx_mask):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :adjacency_matrix, (batch_size, ctx_size, ctx_size)
        """
        context_fc = torch.relu(self.linear_sim(context))
        attention = torch.matmul(context_fc, context_fc.transpose(-1, -2))
        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), -INF)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), -INF)
        weighted_adjacency_matrix = self.build_knn_neighbourhood(attention, self.topk)
        return weighted_adjacency_matrix

    def build_knn_neighbourhood(self, attention, topk):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((-INF * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix


class GraphMessagePassing(nn.Module):

    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['hidden_size']
        if config['message_function'] == 'edge_mm':
            self.edge_weight_tensor = torch.Tensor(config['num_edge_types'], hidden_size * hidden_size)
            self.edge_weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor))
            self.mp_func = self.msg_pass_edge_mm
        elif config['message_function'] == 'edge_network':
            self.edge_network = torch.Tensor(config['edge_embed_dim'], hidden_size, hidden_size)
            self.edge_network = nn.Parameter(nn.init.xavier_uniform_(self.edge_network))
            self.mp_func = self.msg_pass_edge_network
        elif config['message_function'] == 'edge_pair':
            self.linear_edge = nn.Linear(config['edge_embed_dim'], hidden_size, bias=False)
            self.mp_func = self.msg_pass
        elif config['message_function'] == 'no_edge':
            self.mp_func = self.msg_pass
        else:
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)
        if edge_vec is not None and self.config['message_function'] == 'edge_pair':
            node2edge_emb = node2edge_emb + self.linear_edge(edge_vec)
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, node2edge_emb) + node_state) / norm_
        return agg_state

    def msg_pass_maxpool(self, node_state, edge_vec, node2edge, edge2node, fc_maxpool):
        node2edge_emb = torch.bmm(node2edge, node_state)
        node2edge_emb = fc_maxpool(node2edge_emb)
        node2edge_emb = node2edge_emb.unsqueeze(1) * edge2node.unsqueeze(-1) - (1 - edge2node).unsqueeze(-1) * INF
        node2edge_emb = node2edge_emb.view(-1, node2edge_emb.size(-2), node2edge_emb.size(-1)).transpose(-1, -2)
        agg_state = F.max_pool1d(node2edge_emb, kernel_size=node2edge_emb.size(-1)).squeeze(-1).view(node_state.size())
        agg_state = agg_state * (torch.sum(edge2node, dim=-1, keepdim=True) != 0).float()
        return agg_state

    def msg_pass_edge_mm(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = F.embedding(edge_vec[:, i], self.edge_weight_tensor).view(-1, node_state.size(-1), node_state.size(-1))
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1)
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, new_node2edge_emb) + node_state) / norm_
        return agg_state

    def msg_pass_edge_network(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = torch.mm(edge_vec[:, i], self.edge_network.view(self.edge_network.size(0), -1)).view((-1,) + self.edge_network.shape[-2:])
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1)
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, new_node2edge_emb) + node_state) / norm_
        return agg_state


class GraphNN(nn.Module):

    def __init__(self, config):
        super(GraphNN, self).__init__()
        None
        self.device = config['device']
        hidden_size = config['hidden_size']
        self.graph_direction = config.get('graph_direction', 'all')
        assert self.graph_direction in ('all', 'forward', 'backward')
        self.graph_type = config['graph_type']
        self.graph_hops = config['graph_hops']
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.graph_type in ('static', 'hybrid_sep'):
            self.static_graph_mp = GraphMessagePassing(config)
            self.static_gru_step = GRUStep(hidden_size, hidden_size)
            if self.graph_direction == 'all':
                self.static_gated_fusion = GatedFusion(hidden_size)
        if self.graph_type in ('dynamic', 'hybrid_sep'):
            self.graph_learner = GraphLearner(config['gl_input_size'], hidden_size, topk=config['graph_learner_topk'], num_pers=config['graph_learner_num_pers'], device=self.device)
            self.dynamic_gru_step = GRUStep(hidden_size, hidden_size)
            if self.graph_direction == 'all':
                self.dynamic_gated_fusion = GatedFusion(hidden_size)
        if self.graph_type == 'static':
            self.graph_update = self.static_graph_update
        elif self.graph_type == 'dynamic':
            self.graph_update = self.dynamic_graph_update
        elif self.graph_type == 'hybrid':
            self.graph_update = self.hybrid_graph_update
            self.graph_learner = GraphLearner(config['gl_input_size'], hidden_size, topk=config['graph_learner_topk'], num_pers=config['graph_learner_num_pers'], device=self.device)
            self.static_graph_mp = GraphMessagePassing(config)
            self.hybrid_gru_step = GRUStep(hidden_size, hidden_size // 4 * 4)
            self.linear_kernels = nn.ModuleList([nn.Linear(hidden_size, hidden_size // 4, bias=False) for _ in range(4)])
        elif self.graph_type == 'static_gcn':
            self.graph_update = self.static_gcn
            self.gcn_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(self.graph_hops)])
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(self.graph_type))
        None
        None

    def forward(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        node_state, graph_embedding = self.graph_update(node_state, edge_vec, adj, node_mask=node_mask, raw_node_vec=raw_node_vec)
        return node_state, graph_embedding

    def static_graph_update(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        """Static graph update"""
        node2edge, edge2node = adj
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)
        for _ in range(self.graph_hops):
            bw_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, node2edge, edge2node)
            fw_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            if self.graph_direction == 'all':
                agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
                node_state = self.static_gru_step(node_state, agg_state)
            elif self.graph_direction == 'forward':
                node_state = self.static_gru_step(node_state, fw_agg_state)
            else:
                node_state = self.static_gru_step(node_state, bw_agg_state)
        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def dynamic_graph_update(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        """Dynamic graph update"""
        assert raw_node_vec is not None
        node2edge, edge2node = adj
        dynamic_adjacency_matrix = self.graph_learner(raw_node_vec, node_mask)
        bw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix.transpose(-1, -2), dim=-1)
        fw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix, dim=-1)
        for _ in range(self.graph_hops):
            bw_agg_state = self.aggregate(node_state, bw_dynamic_adjacency_matrix)
            fw_agg_state = self.aggregate(node_state, fw_dynamic_adjacency_matrix)
            if self.graph_direction == 'all':
                agg_state = self.dynamic_gated_fusion(bw_agg_state, fw_agg_state)
                node_state = self.dynamic_gru_step(node_state, agg_state)
            elif self.graph_direction == 'forward':
                node_state = self.dynamic_gru_step(node_state, fw_agg_state)
            else:
                node_state = self.dynamic_gru_step(node_state, bw_agg_state)
        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def hybrid_graph_update(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        assert raw_node_vec is not None
        node2edge, edge2node = adj
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)
        dynamic_adjacency_matrix = self.graph_learner(raw_node_vec, node_mask)
        bw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix.transpose(-1, -2), dim=-1)
        fw_dynamic_adjacency_matrix = torch.softmax(dynamic_adjacency_matrix, dim=-1)
        for _ in range(self.graph_hops):
            bw_dyn_agg_state = self.aggregate(node_state, bw_dynamic_adjacency_matrix)
            fw_dyn_agg_state = self.aggregate(node_state, fw_dynamic_adjacency_matrix)
            bw_sta_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, node2edge, edge2node)
            fw_sta_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            agg_state = torch.cat([self.linear_kernels[i](x) for i, x in enumerate([bw_dyn_agg_state, fw_dyn_agg_state, bw_sta_agg_state, fw_sta_agg_state])], -1)
            node_state = self.hybrid_gru_step(node_state, agg_state)
        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def graph_maxpool(self, node_state, node_mask=None):
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1)).squeeze(-1)
        return graph_embedding

    def aggregate(self, node_state, weighted_adjacency_matrix):
        return torch.bmm(weighted_adjacency_matrix, node_state)

    def static_gcn(self, node_state, edge_vec, adj, node_mask=None, raw_node_vec=None):
        """Static GCN update"""
        node2edge, edge2node = adj
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)
        adj = torch.bmm(edge2node, node2edge)
        adj = adj + adj.transpose(1, 2)
        adj = adj + to_cuda(torch.eye(adj.shape[1], adj.shape[2]), self.device)
        adj = torch.clamp(adj, max=1)
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.stack([torch.diagflat(d_inv_sqrt[i]) for i in range(d_inv_sqrt.shape[0])], dim=0)
        adj = torch.bmm(d_mat_inv_sqrt, torch.bmm(adj, d_mat_inv_sqrt))
        for _ in range(self.graph_hops):
            node_state = F.relu(self.gcn_linear[_](torch.bmm(adj, node_state)))
        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding


class Graph2SeqOutput(object):

    def __init__(self, encoder_outputs, encoder_state, decoded_tokens, loss=0, loss_value=0, enc_attn_weights=None, ptr_probs=None):
        self.encoder_outputs = encoder_outputs
        self.encoder_state = encoder_state
        self.decoded_tokens = decoded_tokens
        self.loss = loss
        self.loss_value = loss_value
        self.enc_attn_weights = enc_attn_weights
        self.ptr_probs = ptr_probs


def create_mask(x, N, device=None):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return to_cuda(torch.Tensor(mask), device)


class Graph2Seq(nn.Module):

    def __init__(self, config, word_embedding, word_vocab):
        """
    :param word_vocab: mainly for info about special tokens and word_vocab size
    :param config: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the graph2seq model; its encoder and decoder will be created automatically.
    """
        super(Graph2Seq, self).__init__()
        self.name = 'Graph2Seq'
        self.device = config['device']
        self.word_dropout = config['word_dropout']
        self.edge_dropout = config['edge_dropout']
        self.bert_dropout = config['bert_dropout']
        self.word_vocab = word_vocab
        self.vocab_size = len(word_vocab)
        self.f_case = config['f_case']
        self.f_pos = config['f_pos']
        self.f_ner = config['f_ner']
        self.f_freq = config['f_freq']
        self.f_dep = config['f_dep']
        self.f_ans = config['f_ans']
        self.dan_type = config.get('dan_type', 'all')
        self.max_dec_steps = config['max_dec_steps']
        self.rnn_type = config['rnn_type']
        self.enc_attn = config['enc_attn']
        self.enc_attn_cover = config['enc_attn_cover']
        self.dec_attn = config['dec_attn']
        self.pointer = config['pointer']
        self.pointer_loss_ratio = config['pointer_loss_ratio']
        self.cover_loss = config['cover_loss']
        self.cover_func = config['cover_func']
        self.message_function = config['message_function']
        self.use_bert = config['use_bert']
        self.use_bert_weight = config['use_bert_weight']
        self.use_bert_gamma = config['use_bert_gamma']
        self.finetune_bert = config.get('finetune_bert', None)
        bert_dim = config['bert_dim'] if self.use_bert else 0
        enc_hidden_size = config['rnn_size']
        if config['dec_hidden_size']:
            dec_hidden_size = config['dec_hidden_size']
            if self.rnn_type == 'lstm':
                self.enc_dec_adapter = nn.ModuleList([nn.Linear(enc_hidden_size, dec_hidden_size) for _ in range(2)])
            else:
                self.enc_dec_adapter = nn.Linear(enc_hidden_size, dec_hidden_size)
        else:
            dec_hidden_size = enc_hidden_size
            self.enc_dec_adapter = None
        enc_input_dim = config['word_embed_dim']
        self.word_embed = word_embedding
        if config['fix_word_embed']:
            None
            for param in self.word_embed.parameters():
                param.requires_grad = False
        self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
        if self.f_case:
            self.case_embed = nn.Embedding(3, config['case_embed_dim'], padding_idx=0)
            enc_input_dim += config['case_embed_dim']
        if self.f_pos:
            self.pos_embed = nn.Embedding(config['num_features_f_pos'], config['pos_embed_dim'], padding_idx=0)
            enc_input_dim += config['pos_embed_dim']
        if self.f_ner:
            self.ner_embed = nn.Embedding(config['num_features_f_ner'], config['ner_embed_dim'], padding_idx=0)
            enc_input_dim += config['ner_embed_dim']
        if self.f_freq:
            self.freq_embed = nn.Embedding(4, config['freq_embed_dim'], padding_idx=0)
            enc_input_dim += config['freq_embed_dim']
        if self.f_dep:
            self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
            enc_input_dim += config['edge_embed_dim']
        if self.f_ans and self.dan_type in ('all', 'word'):
            enc_input_dim += config['word_embed_dim']
        if self.use_bert:
            enc_input_dim += config['bert_dim']
        if self.use_bert and self.use_bert_weight:
            num_bert_layers = config['bert_layer_indexes'][1] - config['bert_layer_indexes'][0]
            self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, num_bert_layers)))
            if self.use_bert_gamma:
                self.gamma_bert_layers = nn.Parameter(nn.init.constant_(torch.Tensor(1, 1), 1.0))
        config['gl_input_size'] = enc_input_dim
        self.ctx_rnn_encoder = EncoderRNN(enc_input_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'], device=self.device)
        if self.f_ans:
            if self.dan_type in ('all', 'word'):
                self.ctx2ans_attn_l1 = Context2AnswerAttention(config['word_embed_dim'], config['hidden_size'])
            if self.dan_type in ('all', 'hidden'):
                self.ans_rnn_encoder = EncoderRNN(config['word_embed_dim'] + bert_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'], device=self.device)
                self.ctx2ans_attn_l2 = Context2AnswerAttention(config['word_embed_dim'] + config['hidden_size'] + bert_dim, config['hidden_size'])
                self.ctx_rnn_encoder_l2 = EncoderRNN(2 * enc_hidden_size, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'], device=self.device)
            None
        self.graph_encoder = GraphNN(config)
        self.decoder = DecoderRNN(self.vocab_size, config['word_embed_dim'], dec_hidden_size, rnn_type=self.rnn_type, enc_attn=config['enc_attn'], dec_attn=config['dec_attn'], pointer=config['pointer'], out_embed_size=config['out_embed_size'], tied_embedding=self.word_embed if config['tie_embed'] else None, in_drop=config['dec_in_dropout'], rnn_drop=config['dec_rnn_dropout'], out_drop=config['dec_out_dropout'], enc_hidden_size=enc_hidden_size, device=self.device)

    def filter_oov(self, tensor, ext_vocab_size):
        """Replace any OOV index in `tensor` with UNK"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = tensor.clone()
            result[tensor >= self.vocab_size] = self.word_vocab.UNK
            return result
        return tensor

    def get_coverage_vector(self, enc_attn_weights):
        """Combine the past attention weights into one vector"""
        if self.cover_func == 'max':
            coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
        elif self.cover_func == 'sum':
            coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
        else:
            raise ValueError('Unrecognized cover_func: ' + self.cover_func)
        return coverage_vector

    def forward(self, ex, target_tensor=None, criterion=None, criterion_reduction=True, criterion_nll_only=False, rl_loss=False, *, forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False, saved_out: Graph2SeqOutput=None, visualize: bool=None, include_cover_loss: bool=False) ->Graph2SeqOutput:
        """
    :param input_tensor: tensor of word indices, (batch size, src seq len)
    :param target_tensor: tensor of word indices, (batch size, tgt seq len)
    :param input_lengths: see explanation in `EncoderRNN`
    :param criterion: the loss function; if set, loss will be returned
    :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
    :param partial_forcing: see explanation in `Params` (training only)
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                   of greedily selecting the token of the highest probability at each step
    :param saved_out: the output of this function in a previous run; if set, the encoding step will
                      be skipped and we reuse the encoder states saved in this object
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

    Run the graph2seq model for training or testing.
    """
        input_tensor = ex['context']
        input_lengths = ex['context_lens']
        batch_size, input_length = input_tensor.shape
        input_mask = create_mask(input_lengths, input_length, self.device)
        log_prob = not (sample or self.decoder.pointer)
        if visualize is None:
            visualize = criterion is None
        if visualize and not (self.enc_attn or self.pointer):
            visualize = False
        if target_tensor is None:
            target_length = self.max_dec_steps
            target_mask = None
        else:
            target_tensor = target_tensor.transpose(1, 0)
            target_length = target_tensor.size(0)
            target_mask = create_mask(ex['target_lens'], target_length, self.device)
        if forcing_ratio == 1:
            use_teacher_forcing = True
        elif forcing_ratio > 0:
            if partial_forcing:
                use_teacher_forcing = None
            else:
                use_teacher_forcing = random.random() < forcing_ratio
        else:
            use_teacher_forcing = False
        if saved_out:
            encoder_outputs = saved_out.encoder_outputs
            encoder_state = saved_out.encoder_state
            assert input_length == encoder_outputs.size(0)
            assert batch_size == encoder_outputs.size(1)
        else:
            encoder_embedded = self.word_embed(self.filter_oov(input_tensor, ext_vocab_size))
            encoder_embedded = dropout(encoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
            enc_input_cat = [encoder_embedded]
            if self.f_case:
                case_features = self.case_embed(ex['context_case'])
                enc_input_cat.append(case_features)
            if self.f_pos:
                pos_features = self.pos_embed(ex['context_pos'])
                enc_input_cat.append(pos_features)
            if self.f_ner:
                ner_features = self.ner_embed(ex['context_ner'])
                enc_input_cat.append(ner_features)
            if self.f_freq:
                freq_features = self.freq_embed(ex['context_freq'])
                enc_input_cat.append(freq_features)
            if self.f_dep:
                dep_features = self.edge_embed(ex['context_dep'])
                enc_input_cat.append(dep_features)
            if self.f_ans:
                answer_tensor = ex['answers']
                answer_lengths = ex['answer_lens']
                ans_mask = create_mask(answer_lengths, answer_tensor.size(1), self.device)
                ans_embedded = self.word_embed(self.filter_oov(answer_tensor, ext_vocab_size))
                ans_embedded = dropout(ans_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
                enc_answer_cat = [ans_embedded]
                if self.dan_type in ('all', 'word'):
                    ctx_aware_ans_emb = self.ctx2ans_attn_l1(encoder_embedded, ans_embedded, ans_embedded, ans_mask)
                    enc_input_cat.append(ctx_aware_ans_emb)
            if self.use_bert:
                context_bert = ex['context_bert']
                if not self.finetune_bert:
                    assert context_bert.requires_grad == False
                if self.use_bert_weight:
                    weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
                    if self.use_bert_gamma:
                        weights_bert_layers = weights_bert_layers * self.gamma_bert_layers
                    context_bert = torch.mm(weights_bert_layers, context_bert.view(context_bert.size(0), -1)).view(context_bert.shape[1:])
                    context_bert = dropout(context_bert, self.bert_dropout, shared_axes=[-2], training=self.training)
                    enc_input_cat.append(context_bert)
                    if self.f_ans and self.dan_type in ('all', 'hidden'):
                        answer_bert = ex['answer_bert']
                        if not self.finetune_bert:
                            assert answer_bert.requires_grad == False
                        answer_bert = torch.mm(weights_bert_layers, answer_bert.view(answer_bert.size(0), -1)).view(answer_bert.shape[1:])
                        answer_bert = dropout(answer_bert, self.bert_dropout, shared_axes=[-2], training=self.training)
                        enc_answer_cat.append(answer_bert)
            raw_input_vec = torch.cat(enc_input_cat, -1)
            encoder_outputs = self.ctx_rnn_encoder(raw_input_vec, input_lengths)[0].transpose(0, 1)
            if self.f_ans and self.dan_type in ('all', 'hidden'):
                enc_answer_cat = torch.cat(enc_answer_cat, -1)
                ans_encoder_outputs = self.ans_rnn_encoder(enc_answer_cat, answer_lengths)[0].transpose(0, 1)
                enc_cat_l2 = torch.cat([encoder_embedded, encoder_outputs], -1)
                ans_cat_l2 = torch.cat([ans_embedded, ans_encoder_outputs], -1)
                if self.use_bert:
                    enc_cat_l2 = torch.cat([enc_cat_l2, context_bert], -1)
                    ans_cat_l2 = torch.cat([ans_cat_l2, answer_bert], -1)
                ctx_aware_ans_emb = self.ctx2ans_attn_l2(enc_cat_l2, ans_cat_l2, ans_encoder_outputs, ans_mask)
                encoder_outputs = self.ctx_rnn_encoder_l2(torch.cat([encoder_outputs, ctx_aware_ans_emb], -1), input_lengths)[0].transpose(0, 1)
            input_graphs = ex['context_graphs']
            if self.message_function == 'edge_mm':
                edge_vec = input_graphs['edge_features']
            else:
                edge_vec = self.edge_embed(input_graphs['edge_features'])
            node_embedding, graph_embedding = self.graph_encoder(encoder_outputs, edge_vec, (input_graphs['node2edge'], input_graphs['edge2node']), node_mask=input_mask, raw_node_vec=raw_input_vec)
            encoder_outputs = node_embedding
            encoder_state = (graph_embedding, graph_embedding) if self.rnn_type == 'lstm' else graph_embedding
        r = Graph2SeqOutput(encoder_outputs, encoder_state, torch.zeros(target_length, batch_size, dtype=torch.long))
        if visualize:
            r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
            if self.pointer:
                r.ptr_probs = torch.zeros(target_length, batch_size)
        if self.enc_dec_adapter is None:
            decoder_state = encoder_state
        elif self.rnn_type == 'lstm':
            decoder_state = tuple([self.enc_dec_adapter[i](x) for i, x in enumerate(encoder_state)])
        else:
            decoder_state = self.enc_dec_adapter(encoder_state)
        decoder_hiddens = []
        enc_attn_weights = []
        enc_context = None
        dec_prob_ptr_tensor = []
        decoder_input = to_cuda(torch.tensor([self.word_vocab.SOS] * batch_size), self.device)
        for di in range(target_length):
            decoder_embedded = self.word_embed(self.filter_oov(decoder_input, ext_vocab_size))
            decoder_embedded = dropout(decoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
            if enc_attn_weights:
                coverage_vector = self.get_coverage_vector(enc_attn_weights)
            else:
                coverage_vector = None
            decoder_output, decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = self.decoder(decoder_embedded, decoder_state, encoder_outputs, torch.cat(decoder_hiddens) if decoder_hiddens else None, coverage_vector, input_mask=input_mask, encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size, log_prob=log_prob, prev_enc_context=enc_context)
            dec_prob_ptr_tensor.append(dec_prob_ptr)
            if self.dec_attn:
                decoder_hiddens.append(decoder_state[0] if self.rnn_type == 'lstm' else decoder_state)
            if not sample:
                _, top_idx = decoder_output.data.topk(1)
            else:
                prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
                top_idx = torch.multinomial(prob_distribution, 1)
            top_idx = top_idx.squeeze(1).detach()
            r.decoded_tokens[di] = top_idx
            if use_teacher_forcing or use_teacher_forcing is None and random.random() < forcing_ratio:
                decoder_input = target_tensor[di]
            else:
                decoder_input = top_idx
            if criterion:
                if target_tensor is None:
                    gold_standard = top_idx
                else:
                    gold_standard = target_tensor[di] if not rl_loss else decoder_input
                if not log_prob:
                    decoder_output = torch.log(decoder_output + VERY_SMALL_NUMBER)
                if criterion_reduction:
                    nll_loss = criterion(decoder_output, gold_standard)
                    r.loss += nll_loss
                    r.loss_value += nll_loss.item()
                else:
                    nll_loss = F.nll_loss(decoder_output, gold_standard, ignore_index=self.word_vocab.PAD, reduction='none')
                    r.loss += nll_loss
                    r.loss_value += nll_loss
            if self.enc_attn_cover or criterion and self.cover_loss > 0:
                if not criterion_nll_only and coverage_vector is not None and criterion and self.cover_loss > 0:
                    if criterion_reduction:
                        coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * self.cover_loss
                        r.loss += coverage_loss
                        if include_cover_loss:
                            r.loss_value += coverage_loss.item()
                    else:
                        coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn), dim=-1) * self.cover_loss
                        r.loss += coverage_loss
                        if include_cover_loss:
                            r.loss_value += coverage_loss
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
            if visualize:
                r.enc_attn_weights[di] = dec_enc_attn.data
                if self.pointer:
                    r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data
        if not criterion_nll_only and criterion and self.pointer_loss_ratio > 0 and target_tensor is not None:
            dec_prob_ptr_tensor = torch.cat(dec_prob_ptr_tensor, -1)
            pointer_loss = F.binary_cross_entropy(dec_prob_ptr_tensor, ex['target_copied'], reduction='none')
            if criterion_reduction:
                pointer_loss = torch.sum(pointer_loss * target_mask) / batch_size * self.pointer_loss_ratio
                r.loss += pointer_loss
                r.loss_value += pointer_loss.item()
            else:
                pointer_loss = torch.sum(pointer_loss * target_mask, dim=-1) * self.pointer_loss_ratio
                r.loss += pointer_loss
                r.loss_value += pointer_loss
        return r


class Seq2SeqOutput(object):

    def __init__(self, encoder_outputs, encoder_state, decoded_tokens, loss=0, loss_value=0, enc_attn_weights=None, ptr_probs=None):
        self.encoder_outputs = encoder_outputs
        self.encoder_state = encoder_state
        self.decoded_tokens = decoded_tokens
        self.loss = loss
        self.loss_value = loss_value
        self.enc_attn_weights = enc_attn_weights
        self.ptr_probs = ptr_probs


class Seq2Seq(nn.Module):
    """BERT feature is not implemented yet."""

    def __init__(self, config, word_embedding, word_vocab):
        """
    :param word_vocab: mainly for info about special tokens and word_vocab size
    :param config: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the seq2seq model; its encoder and decoder will be created automatically.
    """
        super(Seq2Seq, self).__init__()
        self.name = 'Seq2Seq'
        self.device = config['device']
        self.word_dropout = config['word_dropout']
        self.word_vocab = word_vocab
        self.vocab_size = len(word_vocab)
        self.f_case = config['f_case']
        self.f_pos = config['f_pos']
        self.f_ner = config['f_ner']
        self.f_freq = config['f_freq']
        self.f_dep = config['f_dep']
        self.f_ans = config['f_ans']
        self.max_dec_steps = config['max_dec_steps']
        self.rnn_type = config['rnn_type']
        self.enc_attn = config['enc_attn']
        self.enc_attn_cover = config['enc_attn_cover']
        self.dec_attn = config['dec_attn']
        self.pointer = config['pointer']
        self.pointer_loss_ratio = config['pointer_loss_ratio']
        self.cover_loss = config['cover_loss']
        self.cover_func = config['cover_func']
        self.use_bert = config['use_bert']
        enc_hidden_size = config['rnn_size']
        if config['dec_hidden_size']:
            dec_hidden_size = config['dec_hidden_size']
            if self.rnn_type == 'lstm':
                self.enc_dec_adapter = nn.ModuleList([nn.Linear(enc_hidden_size, dec_hidden_size) for _ in range(2)])
            else:
                self.enc_dec_adapter = nn.Linear(enc_hidden_size, dec_hidden_size)
        else:
            dec_hidden_size = enc_hidden_size
            self.enc_dec_adapter = None
        enc_input_dim = config['word_embed_dim']
        self.word_embed = word_embedding
        if config['fix_word_embed']:
            None
            for param in self.word_embed.parameters():
                param.requires_grad = False
        if self.f_case:
            self.case_embed = nn.Embedding(3, config['case_embed_dim'], padding_idx=0)
            enc_input_dim += config['case_embed_dim']
        if self.f_pos:
            self.pos_embed = nn.Embedding(config['num_features_f_pos'], config['pos_embed_dim'], padding_idx=0)
            enc_input_dim += config['pos_embed_dim']
        if self.f_ner:
            self.ner_embed = nn.Embedding(config['num_features_f_ner'], config['ner_embed_dim'], padding_idx=0)
            enc_input_dim += config['ner_embed_dim']
        if self.f_freq:
            self.freq_embed = nn.Embedding(4, config['freq_embed_dim'], padding_idx=0)
            enc_input_dim += config['freq_embed_dim']
        if self.f_dep:
            self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
            enc_input_dim += config['edge_embed_dim']
        if self.f_ans:
            enc_input_dim += config['word_embed_dim']
        self.ctx_rnn_encoder = EncoderRNN(enc_input_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'], device=self.device)
        self.ctx_rnn_encoder_l2 = EncoderRNN(2 * enc_hidden_size, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'], device=self.device)
        self.ans_rnn_encoder = EncoderRNN(config['word_embed_dim'], enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type, rnn_dropout=config['enc_rnn_dropout'], device=self.device)
        self.decoder = DecoderRNN(self.vocab_size, config['word_embed_dim'], dec_hidden_size, rnn_type=self.rnn_type, enc_attn=config['enc_attn'], dec_attn=config['dec_attn'], pointer=config['pointer'], out_embed_size=config['out_embed_size'], tied_embedding=self.word_embed if config['tie_embed'] else None, in_drop=config['dec_in_dropout'], rnn_drop=config['dec_rnn_dropout'], out_drop=config['dec_out_dropout'], enc_hidden_size=enc_hidden_size, device=self.device)
        self.ctx2ans_attn_l1 = Context2AnswerAttention(config['word_embed_dim'], config['hidden_size'])
        self.ctx2ans_attn_l2 = Context2AnswerAttention(config['word_embed_dim'] + config['hidden_size'], config['hidden_size'])

    def filter_oov(self, tensor, ext_vocab_size):
        """Replace any OOV index in `tensor` with UNK"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = tensor.clone()
            result[tensor >= self.vocab_size] = self.word_vocab.UNK
            return result
        return tensor

    def get_coverage_vector(self, enc_attn_weights):
        """Combine the past attention weights into one vector"""
        if self.cover_func == 'max':
            coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
        elif self.cover_func == 'sum':
            coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
        else:
            raise ValueError('Unrecognized cover_func: ' + self.cover_func)
        return coverage_vector

    def forward(self, ex, target_tensor=None, criterion=None, criterion_reduction=True, criterion_nll_only=False, rl_loss=False, *, forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False, saved_out: Seq2SeqOutput=None, visualize: bool=None, include_cover_loss: bool=False) ->Seq2SeqOutput:
        """
    :param input_tensor: tensor of word indices, (batch size, src seq len)
    :param target_tensor: tensor of word indices, (batch size, tgt seq len)
    :param input_lengths: see explanation in `EncoderRNN`
    :param criterion: the loss function; if set, loss will be returned
    :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
    :param partial_forcing: see explanation in `Params` (training only)
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                   of greedily selecting the token of the highest probability at each step
    :param saved_out: the output of this function in a previous run; if set, the encoding step will
                      be skipped and we reuse the encoder states saved in this object
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

    Run the graph2seq model for training or testing.
    """
        input_tensor = ex['context']
        input_lengths = ex['context_lens']
        input_graphs = ex['context_graphs']
        batch_size, input_length = input_tensor.shape
        input_mask = create_mask(input_lengths, input_length, self.device)
        log_prob = not (sample or self.decoder.pointer)
        if visualize is None:
            visualize = criterion is None
        if visualize and not (self.enc_attn or self.pointer):
            visualize = False
        if target_tensor is None:
            target_length = self.max_dec_steps
            target_mask = None
        else:
            target_tensor = target_tensor.transpose(1, 0)
            target_length = target_tensor.size(0)
            target_mask = create_mask(ex['target_lens'], target_length, self.device)
        if forcing_ratio == 1:
            use_teacher_forcing = True
        elif forcing_ratio > 0:
            if partial_forcing:
                use_teacher_forcing = None
            else:
                use_teacher_forcing = random.random() < forcing_ratio
        else:
            use_teacher_forcing = False
        if saved_out:
            encoder_outputs = saved_out.encoder_outputs
            encoder_state = saved_out.encoder_state
            assert input_length == encoder_outputs.size(0)
            assert batch_size == encoder_outputs.size(1)
        else:
            encoder_embedded = self.word_embed(self.filter_oov(input_tensor, ext_vocab_size))
            encoder_embedded = dropout(encoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
            enc_input_cat = [encoder_embedded]
            if self.f_case:
                case_features = self.case_embed(ex['context_case'])
                enc_input_cat.append(case_features)
            if self.f_pos:
                pos_features = self.pos_embed(ex['context_pos'])
                enc_input_cat.append(pos_features)
            if self.f_ner:
                ner_features = self.ner_embed(ex['context_ner'])
                enc_input_cat.append(ner_features)
            if self.f_freq:
                freq_features = self.freq_embed(ex['context_freq'])
                enc_input_cat.append(freq_features)
            if self.f_dep:
                dep_features = self.edge_embed(ex['context_dep'])
                enc_input_cat.append(dep_features)
            if self.f_ans:
                answer_tensor = ex['answers']
                answer_lengths = ex['answer_lens']
                ans_mask = create_mask(answer_lengths, answer_tensor.size(1), self.device)
                ans_embedded = self.word_embed(self.filter_oov(answer_tensor, ext_vocab_size))
                ans_embedded = dropout(ans_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
                ctx_aware_ans_emb = self.ctx2ans_attn_l1(encoder_embedded, ans_embedded, ans_embedded, ans_mask)
                enc_input_cat.append(ctx_aware_ans_emb)
            encoder_outputs, encoder_state = self.ctx_rnn_encoder(torch.cat(enc_input_cat, -1), input_lengths)
            if self.f_ans:
                encoder_outputs = encoder_outputs.transpose(0, 1)
                ans_encoder_outputs = self.ans_rnn_encoder(ans_embedded, answer_lengths)[0].transpose(0, 1)
                ctx_aware_ans_emb = self.ctx2ans_attn_l2(torch.cat([encoder_embedded, encoder_outputs], -1), torch.cat([ans_embedded, ans_encoder_outputs], -1), ans_encoder_outputs, ans_mask)
                encoder_outputs, encoder_state = self.ctx_rnn_encoder_l2(torch.cat([encoder_outputs, ctx_aware_ans_emb], -1), input_lengths)
        r = Seq2SeqOutput(encoder_outputs, encoder_state, torch.zeros(target_length, batch_size, dtype=torch.long))
        if visualize:
            r.enc_attn_weights = torch.zeros(target_length, batch_size, input_length)
            if self.pointer:
                r.ptr_probs = torch.zeros(target_length, batch_size)
        decoder_input = to_cuda(torch.tensor([self.word_vocab.SOS] * batch_size), self.device)
        if self.enc_dec_adapter is None:
            decoder_state = encoder_state
        elif self.rnn_type == 'lstm':
            decoder_state = tuple([self.enc_dec_adapter[i](x) for i, x in enumerate(encoder_state)])
        else:
            decoder_state = self.enc_dec_adapter(encoder_state)
        decoder_hiddens = []
        enc_attn_weights = []
        enc_context = None
        dec_prob_ptr_tensor = []
        for di in range(target_length):
            decoder_embedded = self.word_embed(self.filter_oov(decoder_input, ext_vocab_size))
            decoder_embedded = dropout(decoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
            if enc_attn_weights:
                coverage_vector = self.get_coverage_vector(enc_attn_weights)
            else:
                coverage_vector = None
            decoder_output, decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = self.decoder(decoder_embedded, decoder_state, encoder_outputs, torch.cat(decoder_hiddens) if decoder_hiddens else None, coverage_vector, input_mask=input_mask, encoder_word_idx=input_tensor, ext_vocab_size=ext_vocab_size, log_prob=log_prob, prev_enc_context=enc_context)
            dec_prob_ptr_tensor.append(dec_prob_ptr)
            if self.dec_attn:
                decoder_hiddens.append(decoder_state[0] if self.rnn_type == 'lstm' else decoder_state)
            if not sample:
                _, top_idx = decoder_output.data.topk(1)
            else:
                prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
                top_idx = torch.multinomial(prob_distribution, 1)
            top_idx = top_idx.squeeze(1).detach()
            r.decoded_tokens[di] = top_idx
            if use_teacher_forcing or use_teacher_forcing is None and random.random() < forcing_ratio:
                decoder_input = target_tensor[di]
            else:
                decoder_input = top_idx
            if criterion:
                if target_tensor is None:
                    gold_standard = top_idx
                else:
                    gold_standard = target_tensor[di]
                if not log_prob:
                    decoder_output = torch.log(decoder_output + VERY_SMALL_NUMBER)
                nll_loss = criterion(decoder_output, gold_standard)
                r.loss += nll_loss
                r.loss_value += nll_loss.item()
            if self.enc_attn_cover or criterion and self.cover_loss > 0:
                if coverage_vector is not None and criterion and self.cover_loss > 0:
                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * self.cover_loss
                    r.loss += coverage_loss
                    if include_cover_loss:
                        r.loss_value += coverage_loss.item()
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
            if visualize:
                r.enc_attn_weights[di] = dec_enc_attn.data
                if self.pointer:
                    r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data
        if criterion and self.pointer_loss_ratio > 0 and target_tensor is not None:
            dec_prob_ptr_tensor = torch.cat(dec_prob_ptr_tensor, -1)
            pointer_loss = F.binary_cross_entropy(dec_prob_ptr_tensor, ex['target_copied'], reduction='none')
            pointer_loss = torch.sum(pointer_loss * target_mask) / batch_size * self.pointer_loss_ratio
            r.loss += pointer_loss
            r.loss_value += pointer_loss.item()
        return r


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Context2AnswerAttention,
     lambda: ([], {'dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EncoderRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (GRUStep,
     lambda: ([], {'hidden_size': 4, 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedFusion,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_hugochan_RL_based_Graph2Seq_for_NQG(_paritybench_base):
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

