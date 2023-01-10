import sys
_module = sys.modules[__name__]
del sys
dataset = _module
doc_text = _module
preprocess_data = _module
squad_dataset = _module
analysis_ans = _module
analysis_dataset = _module
analysis_log = _module
analysis_oov = _module
ans_compare = _module
get_oov = _module
model_transform = _module
transform_hdf5 = _module
vis_char_emb = _module
models = _module
base_model = _module
layers = _module
loss = _module
match_lstm = _module
match_lstm_plus = _module
mnemonic = _module
r_net = _module
run = _module
test = _module
test_input = _module
train = _module
utils = _module
eval = _module
functions = _module
load_config = _module

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


import logging


import torch


import numpy as np


import math


import torch.utils.data


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import SequentialSampler


import pandas as pd


from collections import OrderedDict


import string


import torch.nn as nn


import torch.nn.functional as F


import matplotlib.pyplot as plt


import torch.optim as optim


from functools import reduce


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-06)
    return softmax


class AttentionPooling(torch.nn.Module):
    """
    Attention-Pooling for pointer net init hidden state generate.
    Equal to Self-Attention + MLP
    Modified from r-net.
    Args:
        input_size: The number of expected features in the input uq
        output_size: The number of expected features in the output rq_o

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (batch, output_size): tensor containing the output features
    """

    def __init__(self, input_size, output_size):
        super(AttentionPooling, self).__init__()
        self.linear_u = torch.nn.Linear(input_size, output_size)
        self.linear_t = torch.nn.Linear(output_size, 1)
        self.linear_o = torch.nn.Linear(input_size, output_size)

    def forward(self, uq, mask):
        q_tanh = F.tanh(self.linear_u(uq))
        q_s = self.linear_t(q_tanh).squeeze(2).transpose(0, 1)
        alpha = masked_softmax(q_s, mask, dim=1)
        rq = torch.bmm(alpha.unsqueeze(1), uq.transpose(0, 1)).squeeze(1)
        rq_o = F.tanh(self.linear_o(rq))
        return rq_o


class PointerAttention(torch.nn.Module):
    """
    attention mechanism in pointer network
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        Hk_last(batch, hidden_size): the last hidden output of previous time

    Outputs:
        beta(batch, context_len): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(PointerAttention, self).__init__()
        self.linear_wr = torch.nn.Linear(input_size, hidden_size)
        self.linear_wa = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wf = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hr, Hr_mask, Hk_pre):
        wr_hr = self.linear_wr(Hr)
        wa_ha = self.linear_wa(Hk_pre).unsqueeze(0)
        f = F.tanh(wr_hr + wa_ha)
        beta_tmp = self.linear_wf(f).squeeze(2).transpose(0, 1)
        beta = masked_softmax(beta_tmp, m=Hr_mask, dim=1)
        return beta


class UniBoundaryPointer(torch.nn.Module):
    """
    Unidirectional Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0(batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
        **hidden** (batch, hidden_size), [(batch, hidden_size)]: rnn last state
    """
    answer_len = 2

    def __init__(self, mode, input_size, hidden_size, enable_layer_norm):
        super(UniBoundaryPointer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enable_layer_norm = enable_layer_norm
        self.attention = PointerAttention(input_size, hidden_size)
        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hr, Hr_mask, h_0=None):
        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = Hr.new_zeros(batch_size, self.hidden_size)
        hidden = (h_0, h_0) if self.mode == 'LSTM' and isinstance(h_0, torch.Tensor) else h_0
        beta_out = []
        for t in range(self.answer_len):
            attention_input = hidden[0] if self.mode == 'LSTM' else hidden
            beta = self.attention.forward(Hr, Hr_mask, attention_input)
            beta_out.append(beta)
            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)).squeeze(1)
            if self.enable_layer_norm:
                context_beta = self.layer_norm(context_beta)
            hidden = self.hidden_cell.forward(context_beta, hidden)
        result = torch.stack(beta_out, dim=0)
        return result, hidden


class BoundaryPointer(torch.nn.Module):
    """
    Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - bidirectional: Bidirectional or Unidirectional
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm):
        super(BoundaryPointer, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.left_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        if bidirectional:
            self.right_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)
        left_beta, _ = self.left_ptr_rnn.forward(Hr, Hr_mask, h_0)
        rtn_beta = left_beta
        if self.bidirectional:
            right_beta_inv, _ = self.right_ptr_rnn.forward(Hr, Hr_mask, h_0)
            right_beta = right_beta_inv[[1, 0], :]
            rtn_beta = (left_beta + right_beta) / 2
        new_mask = torch.neg((Hr_mask - 1) * 1e-06)
        rtn_beta = rtn_beta + new_mask.unsqueeze(0)
        return rtn_beta


class CharCNN(torch.nn.Module):
    """
    Char-level CNN
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, filters_size, filters_num, dropout_p):
        super(CharCNN, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.cnns = torch.nn.ModuleList([torch.nn.Conv2d(1, fn, (fw, emb_size)) for fw, fn in zip(filters_size, filters_num)])

    def forward(self, x, char_mask, word_mask):
        x = self.dropout(x)
        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size).unsqueeze(1)
        x = [F.relu(cnn(x)).squeeze(-1) for cnn in self.cnns]
        x = [torch.max(cx, 2)[0] for cx in x]
        x = torch.cat(x, dim=1)
        x = x.view(batch_size, seq_len, -1)
        x = x * word_mask.unsqueeze(-1)
        return x.transpose(0, 1)


class Highway(torch.nn.Module):

    def __init__(self, in_size, n_layers, dropout_p):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.normal_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))
            x = gate * normal_layer_ret + (1 - gate) * x
        return x


class CharCNNEncoder(torch.nn.Module):
    """
    char-level cnn encoder with highway networks
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, hidden_size, filters_size, filters_num, dropout_p, enable_highway=True):
        super(CharCNNEncoder, self).__init__()
        self.enable_highway = enable_highway
        self.hidden_size = hidden_size
        self.cnn = CharCNN(emb_size=emb_size, filters_size=filters_size, filters_num=filters_num, dropout_p=dropout_p)
        if enable_highway:
            self.highway = Highway(in_size=hidden_size, n_layers=2, dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        o = self.cnn(x, char_mask, word_mask)
        assert o.shape[2] == self.hidden_size
        if self.enable_highway:
            o = self.highway(o)
        return o


def compute_mask(v, padding_idx=0):
    """
    compute mask on given tensor v
    :param v:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(v, padding_idx).float()
    return mask


class CharEmbedding(torch.nn.Module):
    """
    Char Embedding Layer, random weight
    Args:
        - dataset_h5_path: char embedding file path
    Inputs:
        **input** (batch, seq_len, word_len): word sequence with char index
    Outputs
        **output** (batch, seq_len, word_len, embedding_size): tensor contain char embeddings
        **mask** (batch, seq_len, word_len)
    """

    def __init__(self, dataset_h5_path, embedding_size, trainable=False):
        super(CharEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embedding = self.load_dataset_h5()
        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embedding, embedding_dim=embedding_size, padding_idx=0)
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_dataset_h5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            word_dict_size = f.attrs['char_dict_size']
        return int(word_dict_size)

    def forward(self, x):
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)
        mask = compute_mask(x, 0)
        x_emb = self.embedding_layer.forward(x)
        x_emb = x_emb.view(batch_size, seq_len, word_len, -1)
        mask = mask.view(batch_size, seq_len, word_len)
        return x_emb, mask


class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyRNNBase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm
        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask):
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        v_sort = v.index_select(1, idx_sort)
        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)
        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)
        o_unsort = o.index_select(1, idx_unsort)
        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        o_last = o_unsort.gather(0, len_idx)
        o_last = o_last.squeeze(0)
        return o_unsort, o_last


class MyStackedRNN(torch.nn.Module):
    """
    RNN with packed sequence and dropout, multi-layers used
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: number of rnn layers
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyStackedRNN, self).__init__()
        self.num_layers = num_layers
        self.rnn_list = torch.nn.ModuleList([MyRNNBase(mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm) for _ in range(num_layers)])

    def forward(self, v, mask):
        v_last = None
        for i in range(self.num_layers):
            v, v_last = self.rnn_list[i].forward(v, mask)
        return v, v_last


class CharEncoder(torch.nn.Module):
    """
    char-level encoder with MyRNNBase used
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p):
        super(CharEncoder, self).__init__()
        self.encoder = MyStackedRNN(mode=mode, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size)
        x = x.transpose(0, 1)
        char_mask = char_mask.view(-1, word_len)
        _, x_encode = self.encoder.forward(x, char_mask)
        x_encode = x_encode.view(batch_size, seq_len, -1)
        x_encode = x_encode * word_mask.unsqueeze(-1)
        return x_encode.transpose(0, 1)


class GloveEmbedding(torch.nn.Module):
    """
    Glove Embedding Layer, also compute the mask of padding index
    Args:
        - dataset_h5_path: glove embedding file path
    Inputs:
        **input** (batch, seq_len): sequence with word index
    Outputs
        **output** (seq_len, batch, embedding_size): tensor that change word index to word embeddings
        **mask** (batch, seq_len): tensor that show which index is padding
    """

    def __init__(self, dataset_h5_path):
        super(GloveEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embeddings, len_embedding, weights = self.load_glove_hdf5()
        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding, _weight=weights)
        self.embedding_layer.weight.requires_grad = False

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']
        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def forward(self, x):
        mask = compute_mask(x)
        tmp_emb = self.embedding_layer.forward(x)
        out_emb = tmp_emb.transpose(0, 1)
        return out_emb, mask


class MatchRNNAttention(torch.nn.Module):
    """
    attention mechanism in match-rnn
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hpi(batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output

    Outputs:
        alpha(batch, question_len): attention vector
    """

    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MatchRNNAttention, self).__init__()
        self.linear_wq = torch.nn.Linear(hq_input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(hp_input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)
        wp_hp = self.linear_wp(Hpi).unsqueeze(0)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(0)
        G = F.tanh(wq_hq + wp_hp + wr_hr)
        wg_g = self.linear_wg(G).squeeze(2).transpose(0, 1)
        alpha = masked_softmax(wg_g, m=Hq_mask, dim=1)
        return alpha


class UniMatchRNN(torch.nn.Module):
    """
    interaction context and question with attention mechanism, one direction, using LSTM cell
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded

    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
        alpha(batch, question_len, context_len): used for visual show
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm):
        super(UniMatchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.enable_layer_norm = enable_layer_norm
        rnn_in_size = hp_input_size + hq_input_size
        self.attention = MatchRNNAttention(hp_input_size, hq_input_size, hidden_size)
        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(rnn_in_size, rnn_in_size)
        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(rnn_in_size)
        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=rnn_in_size, hidden_size=hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size=rnn_in_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]
        h_0 = Hq.new_zeros(batch_size, self.hidden_size)
        hidden = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]
        vis_para = {}
        vis_alpha = []
        vis_gated = []
        for t in range(context_len):
            cur_hp = Hp[t, ...]
            attention_input = hidden[t][0] if self.mode == 'LSTM' else hidden[t]
            alpha = self.attention.forward(cur_hp, Hq, attention_input, Hq_mask)
            vis_alpha.append(alpha)
            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)).squeeze(1)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)
            if self.gated_attention:
                gate = F.sigmoid(self.gated_linear.forward(cur_z))
                vis_gated.append(gate.squeeze(-1))
                cur_z = gate * cur_z
            if self.enable_layer_norm:
                cur_z = self.layer_norm(cur_z)
            cur_hidden = self.hidden_cell.forward(cur_z, hidden[t])
            hidden.append(cur_hidden)
        vis_para['gated'] = torch.stack(vis_gated, dim=-1)
        vis_para['alpha'] = torch.stack(vis_alpha, dim=2)
        hidden_state = list(map(lambda x: x[0], hidden)) if self.mode == 'LSTM' else hidden
        result = torch.stack(hidden_state[1:], dim=0)
        return result, vis_para


def masked_flip(vin, mask, flip_dim=0):
    """
    flip a tensor
    :param vin: (..., batch, ...), batch should on dim=1, input batch with padding values
    :param mask: (batch, seq_len), show whether padding index
    :param flip_dim: dim to flip on
    :return:
    """
    length = mask.data.eq(1).long().sum(1)
    batch_size = vin.shape[1]
    flip_list = []
    for i in range(batch_size):
        cur_tensor = vin[:, i, :]
        cur_length = length[i]
        idx = list(range(cur_length - 1, -1, -1)) + list(range(cur_length, vin.shape[flip_dim]))
        idx = vin.new_tensor(idx, dtype=torch.long)
        cur_inv_tensor = cur_tensor.unsqueeze(1).index_select(flip_dim, idx).squeeze(1)
        flip_list.append(cur_inv_tensor)
    inv_tensor = torch.stack(flip_list, dim=1)
    return inv_tensor


class MatchRNN(torch.nn.Module):
    """
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - gated_attention: If ``True``, gated attention used, see more on R-NET

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_mask(batch, context_len): each context valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values

    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, bidirectional, gated_attention, dropout_p, enable_layer_norm):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2
        self.left_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)
        Hq = self.dropout(Hq)
        left_hidden, left_para = self.left_match_rnn.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden
        rtn_para = {'left': left_para}
        if self.bidirectional:
            Hp_inv = masked_flip(Hp, Hp_mask, flip_dim=0)
            right_hidden_inv, right_para_inv = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)
            right_alpha_inv = right_para_inv['alpha']
            right_alpha_inv = right_alpha_inv.transpose(0, 1)
            right_alpha = masked_flip(right_alpha_inv, Hp_mask, flip_dim=2)
            right_alpha = right_alpha.transpose(0, 1)
            right_gated_inv = right_para_inv['gated']
            right_gated_inv = right_gated_inv.transpose(0, 1)
            right_gated = masked_flip(right_gated_inv, Hp_mask, flip_dim=2)
            right_gated = right_gated.transpose(0, 1)
            right_hidden = masked_flip(right_hidden_inv, Hp_mask, flip_dim=0)
            rtn_para['right'] = {'alpha': right_alpha, 'gated': right_gated}
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)
        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]
        return real_rtn_hidden, last_hidden, rtn_para


class MultiHopBdPointer(torch.nn.Module):
    """
    Boundary Pointer Net with Multi-Hops that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - num_hops: Number of max hops
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, num_hops, dropout_p, enable_layer_norm):
        super(MultiHopBdPointer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hops = num_hops
        self.ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)
        beta_last = None
        for i in range(self.num_hops):
            beta, h_0 = self.ptr_rnn.forward(Hr, Hr_mask, h_0)
            if beta_last is not None and (beta_last == beta).sum().item() == beta.shape[0]:
                break
            beta_last = beta
        new_mask = torch.neg((Hr_mask - 1) * 1e-06)
        rtn_beta = beta + new_mask.unsqueeze(0)
        return rtn_beta


class SelfGated(torch.nn.Module):
    """
    Self-Gated layer. math: \\sigmoid(W*x) * x
    """

    def __init__(self, input_size):
        super(SelfGated, self).__init__()
        self.linear_g = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        x_l = self.linear_g(x)
        x_gt = F.sigmoid(x_l)
        x = x * x_gt
        return x


def answer_search(answer_prop, mask, max_tokens=15):
    """
    global search best answer for model predict
    :param answer_prop: (batch, answer_len, context_len)
    :return:
    """
    batch_size = answer_prop.shape[0]
    context_len = answer_prop.shape[2]
    lengths = mask.data.eq(1).long().sum(1).squeeze()
    min_length, _ = torch.min(lengths, 0)
    min_length = min_length.item()
    max_move = max_tokens + context_len - min_length
    max_move = min(context_len, max_move)
    ans_s_p = answer_prop[:, 0, :]
    ans_e_p = answer_prop[:, 1, :]
    b_zero = answer_prop.new_zeros(batch_size, 1)
    ans_s_e_p_lst = []
    for i in range(max_move):
        tmp_ans_s_e_p = ans_s_p * ans_e_p
        ans_s_e_p_lst.append(tmp_ans_s_e_p)
        ans_s_p = ans_s_p[:, :context_len - 1]
        ans_s_p = torch.cat((b_zero, ans_s_p), dim=1)
    ans_s_e_p = torch.stack(ans_s_e_p_lst, dim=2)
    max_prop1, max_prop_idx1 = torch.max(ans_s_e_p, 1)
    max_prop2, max_prop_idx2 = torch.max(max_prop1, 1)
    ans_e = max_prop_idx1.gather(1, max_prop_idx2.unsqueeze(1)).squeeze(1)
    ans_s = ans_e - max_prop_idx2
    ans_range = torch.stack((ans_s, ans_e), dim=1)
    return ans_range


def multi_scale_ptr(ptr_net, ptr_init_h, hr, hr_mask, scales):
    """
    for multi-scale pointer net output
    :param ptr_net:
    :param ptr_init_h: pointer net init hidden state
    :param hr: (seq_len, batch, hidden_size), question-aware passage representation
    :param hr_mask: (batch, seq_len)
    :param scales: list of different scales, for example: [1, 2, 4]. it should be even numbers
    :return:
    """
    seq_len = hr.shape[0]
    batch_size = hr.shape[1]
    ans_range_prop = hr.new_zeros((2, batch_size, seq_len))
    cut_idx = list(range(seq_len))
    for si, s in enumerate(scales):
        scale_seq_len = int(seq_len / s)
        down_idx = [(i * s + s - 1) for i in range(scale_seq_len)]
        if seq_len % s != 0:
            down_idx.append(seq_len - 1)
        down_hr = hr[down_idx]
        down_hr_mask = hr_mask[:, down_idx]
        down_ans_range_prop = ptr_net[si].forward(down_hr, down_hr_mask, ptr_init_h)
        down_seq_len = down_ans_range_prop.shape[2]
        up_idx = [[i for _ in range(s)] for i in range(down_seq_len)]
        up_idx = list(reduce(lambda x, y: x + y, up_idx))
        up_ans_range_prop = down_ans_range_prop[:, :, up_idx]
        up_ans_range_prop = up_ans_range_prop[:, :, cut_idx]
        ans_range_prop += up_ans_range_prop
    ans_range_prop /= len(scales)
    ans_range_prop = ans_range_prop.transpose(0, 1)
    return ans_range_prop


class BaseModel(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path, model_config):
        super(BaseModel, self).__init__()
        hidden_size = model_config['global']['hidden_size']
        hidden_mode = model_config['global']['hidden_mode']
        dropout_p = model_config['global']['dropout_p']
        emb_dropout_p = model_config['global']['emb_dropout_p']
        enable_layer_norm = model_config['global']['layer_norm']
        word_embedding_size = model_config['encoder']['word_embedding_size']
        char_embedding_size = model_config['encoder']['char_embedding_size']
        encoder_word_layers = model_config['encoder']['word_layers']
        encoder_char_layers = model_config['encoder']['char_layers']
        char_trainable = model_config['encoder']['char_trainable']
        char_type = model_config['encoder']['char_encode_type']
        char_cnn_filter_size = model_config['encoder']['char_cnn_filter_size']
        char_cnn_filter_num = model_config['encoder']['char_cnn_filter_num']
        self.enable_char = model_config['encoder']['enable_char']
        add_features = model_config['encoder']['add_features']
        self.enable_features = True if add_features > 0 else False
        self.mix_encode = model_config['encoder']['mix_encode']
        encoder_bidirection = model_config['encoder']['bidirection']
        encoder_direction_num = 2 if encoder_bidirection else 1
        match_lstm_bidirection = model_config['interaction']['match_lstm_bidirection']
        self_match_lstm_bidirection = model_config['interaction']['self_match_bidirection']
        self.enable_self_match = model_config['interaction']['enable_self_match']
        self.enable_birnn_after_self = model_config['interaction']['birnn_after_self']
        gated_attention = model_config['interaction']['gated_attention']
        self.enable_self_gated = model_config['interaction']['self_gated']
        self.enable_question_match = model_config['interaction']['question_match']
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        self_match_rnn_direction_num = 2 if self_match_lstm_bidirection else 1
        num_hops = model_config['output']['num_hops']
        self.scales = model_config['output']['scales']
        ptr_bidirection = model_config['output']['ptr_bidirection']
        self.init_ptr_hidden_mode = model_config['output']['init_ptr_hidden']
        self.enable_search = model_config['output']['answer_search']
        assert num_hops > 0, 'Pointer Net number of hops should bigger than zero'
        if num_hops > 1:
            assert not ptr_bidirection, 'Pointer Net bidirectional should with number of one hop'
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        encode_in_size = word_embedding_size + add_features
        if self.enable_char:
            self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path, embedding_size=char_embedding_size, trainable=char_trainable)
            if char_type == 'LSTM':
                self.char_encoder = CharEncoder(mode=hidden_mode, input_size=char_embedding_size, hidden_size=hidden_size, num_layers=encoder_char_layers, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
            elif char_type == 'CNN':
                self.char_encoder = CharCNNEncoder(emb_size=char_embedding_size, hidden_size=hidden_size, filters_size=char_cnn_filter_size, filters_num=char_cnn_filter_num, dropout_p=emb_dropout_p)
            else:
                raise ValueError('Unrecognized char_encode_type of value %s' % char_type)
            if self.mix_encode:
                encode_in_size += hidden_size * encoder_direction_num
        self.encoder = MyStackedRNN(mode=hidden_mode, input_size=encode_in_size, hidden_size=hidden_size, num_layers=encoder_word_layers, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
        encode_out_size = hidden_size * encoder_direction_num
        if self.enable_char and not self.mix_encode:
            encode_out_size *= 2
        match_rnn_in_size = encode_out_size
        if self.enable_question_match:
            self.question_match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=encode_out_size, hq_input_size=encode_out_size, hidden_size=hidden_size, bidirectional=match_lstm_bidirection, gated_attention=gated_attention, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
            match_rnn_in_size = hidden_size * match_rnn_direction_num
        self.match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=encode_out_size, hq_input_size=match_rnn_in_size, hidden_size=hidden_size, bidirectional=match_lstm_bidirection, gated_attention=gated_attention, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num
        if self.enable_self_match:
            self.self_match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=match_rnn_out_size, hq_input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=self_match_lstm_bidirection, gated_attention=gated_attention, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
            match_rnn_out_size = hidden_size * self_match_rnn_direction_num
        if self.enable_birnn_after_self:
            self.birnn_after_self = MyRNNBase(mode=hidden_mode, input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
            match_rnn_out_size = hidden_size * 2
        if self.enable_self_gated:
            self.self_gated = SelfGated(input_size=match_rnn_out_size)
        if num_hops == 1:
            self.pointer_net = torch.nn.ModuleList([BoundaryPointer(mode=hidden_mode, input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=ptr_bidirection, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm) for _ in range(len(self.scales))])
        else:
            self.pointer_net = torch.nn.ModuleList([MultiHopBdPointer(mode=hidden_mode, input_size=match_rnn_out_size, hidden_size=hidden_size, num_hops=num_hops, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm) for _ in range(len(self.scales))])
        if self.init_ptr_hidden_mode == 'pooling':
            self.init_ptr_hidden = AttentionPooling(encode_out_size, hidden_size)
        elif self.init_ptr_hidden_mode == 'linear':
            self.init_ptr_hidden = nn.Linear(match_rnn_out_size, hidden_size)
        elif self.init_ptr_hidden_mode == 'None':
            pass
        else:
            raise ValueError('Wrong init_ptr_hidden mode select %s, change to pooling or linear' % self.init_ptr_hidden_mode)

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        if self.enable_char:
            assert context_char is not None and question_char is not None
        if self.enable_features:
            assert context_f is not None and question_f is not None
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)
        if self.enable_features:
            assert context_f is not None and question_f is not None
            context_f = context_f.transpose(0, 1)
            question_f = question_f.transpose(0, 1)
            context_vec = torch.cat([context_vec, context_f], dim=-1)
            question_vec = torch.cat([question_vec, question_f], dim=-1)
        if self.enable_char:
            context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
            question_emb_char, question_char_mask = self.char_embedding.forward(question_char)
            context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
            question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)
            if self.mix_encode:
                context_vec = torch.cat((context_vec, context_vec_char), dim=-1)
                question_vec = torch.cat((question_vec, question_vec_char), dim=-1)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)
        if self.enable_char and not self.mix_encode:
            context_encode = torch.cat((context_encode, context_vec_char), dim=-1)
            question_encode = torch.cat((question_encode, question_vec_char), dim=-1)
        match_rnn_in_question = question_encode
        if self.enable_question_match:
            ct_aware_qt, _, _ = self.question_match_rnn.forward(question_encode, question_mask, context_encode, context_mask)
            match_rnn_in_question = ct_aware_qt
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask, match_rnn_in_question, question_mask)
        vis_param = {'match': match_para}
        if self.enable_self_match:
            qt_aware_ct, qt_aware_last_hidden, self_para = self.self_match_rnn.forward(qt_aware_ct, context_mask, qt_aware_ct, context_mask)
            vis_param['self'] = self_para
        if self.enable_birnn_after_self:
            qt_aware_ct, _ = self.birnn_after_self.forward(qt_aware_ct, context_mask)
        if self.enable_self_gated:
            qt_aware_ct = self.self_gated(qt_aware_ct)
        ptr_net_hidden = None
        if self.init_ptr_hidden_mode == 'pooling':
            ptr_net_hidden = self.init_ptr_hidden.forward(question_encode, question_mask)
        elif self.init_ptr_hidden_mode == 'linear':
            ptr_net_hidden = self.init_ptr_hidden.forward(qt_aware_last_hidden)
            ptr_net_hidden = F.tanh(ptr_net_hidden)
        ans_range_prop = multi_scale_ptr(self.pointer_net, ptr_net_hidden, qt_aware_ct, context_mask, self.scales)
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)
        return ans_range_prop, ans_range, vis_param


class SeqPointer(torch.nn.Module):
    """
    Sequence Pointer Net that output every possible answer position in context
    Args:

    Inputs:
        Hr: question-aware context representation
    Outputs:
        **output** every answer index possibility position in context, no fixed length
    """

    def __init__(self):
        super(SeqPointer, self).__init__()

    def forward(self, *input):
        return NotImplementedError


class SelfAttentionGated(torch.nn.Module):
    """
    Self-Attention Gated layer, it`s not weighted sum in the last, but just weighted
    math: \\softmax(W*	anh(W*x)) * x

    Args:
        input_size: The number of expected features in the input x

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (seq_len, batch, input_size): gated output tensor
    """

    def __init__(self, input_size):
        super(SelfAttentionGated, self).__init__()
        self.linear_g = torch.nn.Linear(input_size, input_size)
        self.linear_t = torch.nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        g_tanh = F.tanh(self.linear_g(x))
        gt = self.linear_t.forward(g_tanh).squeeze(2).transpose(0, 1)
        gt_prop = masked_softmax(gt, x_mask, dim=1)
        gt_prop = gt_prop.transpose(0, 1).unsqueeze(2)
        x_gt = x * gt_prop
        return x_gt


class SeqToSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h1: (seq1_len, batch, hidden_size)
        - h1_mask: (batch, seq1_len)
        - h2: (seq2_len, batch, hidden_size)
        - h2_mask: (batch, seq2_len)
    Outputs:
        - output: (seq1_len, batch, hidden_size)
        - alpha: (batch, seq1_len, seq2_len)
    """

    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, h1, h2, h2_mask):
        h1 = h1.transpose(0, 1)
        h2 = h2.transpose(0, 1)
        alpha = h1.bmm(h2.transpose(1, 2))
        alpha = masked_softmax(alpha, h2_mask.unsqueeze(1), dim=2)
        alpha_seq2 = alpha.bmm(h2)
        alpha_seq2 = alpha_seq2.transpose(0, 1)
        return alpha_seq2, alpha


class SelfSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h: (seq_len, batch, hidden_size)
        - h_mask: (batch, seq_len)
    Outputs:
        - output: (seq_len, batch, hidden_size)
        - alpha: (batch, seq_len, seq_len)
    """

    def __init__(self):
        super(SelfSeqAtten, self).__init__()

    def forward(self, h, h_mask):
        h = h.transpose(0, 1)
        batch, seq_len, _ = h.shape
        alpha = h.bmm(h.transpose(1, 2))
        mask = torch.eye(seq_len, dtype=torch.uint8, device=h.device)
        mask = mask.unsqueeze(0)
        alpha.masked_fill_(mask, 0.0)
        alpha = masked_softmax(alpha, h_mask.unsqueeze(1), dim=2)
        alpha_seq = alpha.bmm(h)
        alpha_seq = alpha_seq.transpose(0, 1)
        return alpha_seq, alpha


class SFU(torch.nn.Module):
    """
    only two input, one input vector and one fusion vector

    Args:
        - input_size:
        - fusions_size:
    Inputs:
        - input: (seq_len, batch, input_size)
        - fusions: (seq_len, batch, fusions_size)
    Outputs:
        - output: (seq_len, batch, input_size)
    """

    def __init__(self, input_size, fusions_size):
        super(SFU, self).__init__()
        self.linear_r = torch.nn.Linear(input_size + fusions_size, input_size)
        self.linear_g = torch.nn.Linear(input_size + fusions_size, input_size)

    def forward(self, input, fusions):
        m = torch.cat((input, fusions), dim=-1)
        r = F.tanh(self.linear_r(m))
        g = F.sigmoid(self.linear_g(m))
        o = g * r + (1 - g) * input
        return o


class ForwardNet(torch.nn.Module):
    """
    one hidden layer and one softmax layer.
    Args:
        - input_size:
        - hidden_size:
        - output_size:
        - dropout_p:
    Inputs:
        - x: (seq_len, batch, input_size)
        - x_mask: (batch, seq_len)
    Outputs:
        - beta: (batch, seq_len)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(ForwardNet, self).__init__()
        self.linear_h = torch.nn.Linear(input_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x, x_mask):
        h = F.relu(self.linear_h(x))
        h = self.dropout(h)
        o = self.linear_o(h)
        o = o.squeeze(2).transpose(0, 1)
        beta = masked_softmax(o, x_mask, dim=1)
        return beta


class MemPtrNet(torch.nn.Module):
    """
    memory pointer net
    Args:
        - input_size: zs and hc size
        - hidden_size:
        - dropout_p:
    Inputs:
        - zs: (batch, input_size)
        - hc: (seq_len, batch, input_size)
        - hc_mask: (batch, seq_len)
    Outputs:
        - ans_out: (ans_len, batch, seq_len)
        - zs_new: (batch, input_size)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(MemPtrNet, self).__init__()
        self.start_net = ForwardNet(input_size=input_size * 3, hidden_size=hidden_size, dropout_p=dropout_p)
        self.start_sfu = SFU(input_size, input_size)
        self.end_net = ForwardNet(input_size=input_size * 3, hidden_size=hidden_size, dropout_p=dropout_p)
        self.end_sfu = SFU(input_size, input_size)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, hc, hc_mask, zs):
        hc = self.dropout(hc)
        zs_ep = zs.unsqueeze(0).expand(hc.size())
        x = torch.cat((hc, zs_ep, hc * zs_ep), dim=-1)
        start_p = self.start_net(x, hc_mask)
        us = start_p.unsqueeze(1).bmm(hc.transpose(0, 1)).squeeze(1)
        ze = self.start_sfu(zs, us)
        ze_ep = ze.unsqueeze(0).expand(hc.size())
        x = torch.cat((hc, ze_ep, hc * ze_ep), dim=-1)
        end_p = self.end_net(x, hc_mask)
        ue = end_p.unsqueeze(1).bmm(hc.transpose(0, 1)).squeeze(1)
        zs_new = self.end_sfu(ze, ue)
        ans_out = torch.stack([start_p, end_p], dim=0)
        new_mask = 1 - hc_mask.unsqueeze(0).type(torch.uint8)
        ans_out.masked_fill_(new_mask, 1e-06)
        return ans_out, zs_new


class MyNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)
        - output: loss
    """

    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        torch.nn.modules.loss._assert_no_grad(y_true)
        y_pred_log = torch.log(y_pred)
        start_loss = F.nll_loss(y_pred_log[:, 0, :], y_true[:, 0])
        end_loss = F.nll_loss(y_pred_log[:, 1, :], y_true[:, 1])
        return start_loss + end_loss


class RLLoss(torch.nn.modules.loss._Loss):
    """
    a reinforcement learning loss. f1 score

    Shape:
        - y_pred: (batch, answer_len)
        - y_true: (batch, answer_len)
        - output: loss
    """

    def __init__(self):
        super(RLLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-06):
        return NotImplementedError
        torch.nn.modules.loss._assert_no_grad(y_true)
        assert y_pred.shape[1] == 2
        same_left = torch.stack([y_true[:, 0], y_pred[:, 0]], dim=1)
        same_left, _ = torch.max(same_left, dim=1)
        same_right = torch.stack([y_true[:, 1], y_pred[:, 1]], dim=1)
        same_right, _ = torch.min(same_right, dim=1)
        same_len = same_right - same_left + 1
        same_len = torch.stack([same_len, torch.zeros_like(same_len)], dim=1)
        same_len, _ = torch.max(same_len, dim=1)
        same_len = same_len.type(torch.float)
        pred_len = (y_pred[:, 1] - y_pred[:, 0] + 1).type(torch.float)
        true_len = (y_true[:, 1] - y_true[:, 0] + 1).type(torch.float)
        pre = same_len / (pred_len + eps)
        rec = same_len / (true_len + eps)
        f1 = 2 * pre * rec / (pre + rec + eps)
        return -torch.mean(f1)


class MatchLSTM(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path):
        super(MatchLSTM, self).__init__()
        hidden_size = 150
        dropout_p = 0.4
        emb_dropout_p = 0.1
        enable_layer_norm = False
        hidden_mode = 'LSTM'
        word_embedding_size = 300
        encoder_bidirection = True
        encoder_direction_num = 2 if encoder_bidirection else 1
        match_lstm_bidirection = True
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        ptr_bidirection = True
        self.enable_search = True
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.encoder = MyRNNBase(mode=hidden_mode, input_size=word_embedding_size, hidden_size=hidden_size, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
        encode_out_size = hidden_size * encoder_direction_num
        self.match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=encode_out_size, hq_input_size=encode_out_size, hidden_size=hidden_size, bidirectional=match_lstm_bidirection, gated_attention=False, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num
        self.pointer_net = BoundaryPointer(mode=hidden_mode, input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=ptr_bidirection, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        """
        context_char and question_char not used
        """
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask, question_encode, question_mask)
        vis_param = {'match': match_para}
        ans_range_prop = self.pointer_net.forward(qt_aware_ct, context_mask)
        ans_range_prop = ans_range_prop.transpose(0, 1)
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)
        return ans_range_prop, ans_range, vis_param


class MatchLSTMPlus(torch.nn.Module):
    """
    match-lstm+ model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path):
        super(MatchLSTMPlus, self).__init__()
        hidden_size = 150
        hidden_mode = 'GRU'
        dropout_p = 0.4
        emb_dropout_p = 0.1
        enable_layer_norm = False
        word_embedding_size = 300
        char_embedding_size = 64
        encoder_char_layers = 1
        add_feature_size = 73
        encoder_bidirection = True
        encoder_direction_num = 2 if encoder_bidirection else 1
        match_lstm_bidirection = True
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        ptr_bidirection = False
        self.enable_search = True
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path, embedding_size=char_embedding_size, trainable=True)
        self.char_encoder = CharEncoder(mode=hidden_mode, input_size=char_embedding_size, hidden_size=hidden_size, num_layers=encoder_char_layers, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
        encoder_in_size = add_feature_size + word_embedding_size
        self.encoder = MyRNNBase(mode=hidden_mode, input_size=encoder_in_size, hidden_size=hidden_size, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
        encode_out_size = hidden_size * encoder_direction_num * 2
        self.match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=encode_out_size, hq_input_size=encode_out_size, hidden_size=hidden_size, bidirectional=match_lstm_bidirection, gated_attention=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num
        self.birnn_after_self = MyRNNBase(mode=hidden_mode, input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        birnn_out_size = hidden_size * 2
        self.pointer_net = BoundaryPointer(mode=hidden_mode, input_size=birnn_out_size, hidden_size=hidden_size, bidirectional=ptr_bidirection, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        self.init_ptr_hidden = torch.nn.Linear(match_rnn_out_size, hidden_size)

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        assert context_char is not None and question_char is not None
        context_f = context_f.transpose(0, 1)
        question_f = question_f.transpose(0, 1)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)
        context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
        question_emb_char, question_char_mask = self.char_embedding.forward(question_char)
        context_vec = torch.cat([context_vec, context_f], dim=-1)
        question_vec = torch.cat([question_vec, question_f], dim=-1)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)
        context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
        question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)
        context_encode = torch.cat((context_encode, context_vec_char), dim=-1)
        question_encode = torch.cat((question_encode, question_vec_char), dim=-1)
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask, question_encode, question_mask)
        vis_param = {'match': match_para}
        qt_aware_ct_ag, _ = self.birnn_after_self.forward(qt_aware_ct, context_mask)
        ptr_net_hidden = F.tanh(self.init_ptr_hidden.forward(qt_aware_last_hidden))
        ans_range_prop = self.pointer_net.forward(qt_aware_ct_ag, context_mask, ptr_net_hidden)
        ans_range_prop = ans_range_prop.transpose(0, 1)
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)
        return ans_range_prop, ans_range, vis_param


class MReader(torch.nn.Module):
    """
    mnemonic reader model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path):
        super(MReader, self).__init__()
        hidden_size = 100
        char_encoder_hidden = 50
        hidden_mode = 'LSTM'
        dropout_p = 0.2
        emb_dropout_p = 0.2
        enable_layer_norm = False
        word_embedding_size = 300
        char_embedding_size = 50
        add_feature_size = 73
        self.num_align_hops = 2
        self.num_ptr_hops = 2
        self.enable_search = True
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path, embedding_size=char_embedding_size, trainable=True)
        self.char_encoder = CharEncoder(mode=hidden_mode, input_size=char_embedding_size, hidden_size=char_encoder_hidden, num_layers=1, bidirectional=True, dropout_p=emb_dropout_p)
        encoder_in_size = word_embedding_size + char_encoder_hidden * 2 + add_feature_size
        self.encoder = MyRNNBase(mode=hidden_mode, input_size=encoder_in_size, hidden_size=hidden_size, bidirectional=True, dropout_p=emb_dropout_p)
        self.aligner = torch.nn.ModuleList([SeqToSeqAtten() for _ in range(self.num_align_hops)])
        self.aligner_sfu = torch.nn.ModuleList([SFU(input_size=hidden_size * 2, fusions_size=hidden_size * 2 * 3) for _ in range(self.num_align_hops)])
        self.self_aligner = torch.nn.ModuleList([SelfSeqAtten() for _ in range(self.num_align_hops)])
        self.self_aligner_sfu = torch.nn.ModuleList([SFU(input_size=hidden_size * 2, fusions_size=hidden_size * 2 * 3) for _ in range(self.num_align_hops)])
        self.aggregation = torch.nn.ModuleList([MyRNNBase(mode=hidden_mode, input_size=hidden_size * 2, hidden_size=hidden_size, bidirectional=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm) for _ in range(self.num_align_hops)])
        self.ptr_net = torch.nn.ModuleList([MemPtrNet(input_size=hidden_size * 2, hidden_size=hidden_size, dropout_p=dropout_p) for _ in range(self.num_ptr_hops)])

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        assert context_char is not None and question_char is not None and context_f is not None and question_f is not None
        vis_param = {}
        context_f = context_f.transpose(0, 1)
        question_f = question_f.transpose(0, 1)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)
        context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
        question_emb_char, question_char_mask = self.char_embedding.forward(question_char)
        context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
        question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)
        context_vec = torch.cat((context_vec, context_vec_char, context_f), dim=-1)
        question_vec = torch.cat((question_vec, question_vec_char, question_f), dim=-1)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, zs = self.encoder.forward(question_vec, question_mask)
        align_ct = context_encode
        for i in range(self.num_align_hops):
            qt_align_ct, alpha = self.aligner[i](align_ct, question_encode, question_mask)
            bar_ct = self.aligner_sfu[i](align_ct, torch.cat([qt_align_ct, align_ct * qt_align_ct, align_ct - qt_align_ct], dim=-1))
            vis_param['match'] = alpha
            ct_align_ct, self_alpha = self.self_aligner[i](bar_ct, context_mask)
            hat_ct = self.self_aligner_sfu[i](bar_ct, torch.cat([ct_align_ct, bar_ct * ct_align_ct, bar_ct - ct_align_ct], dim=-1))
            vis_param['self-match'] = self_alpha
            align_ct, _ = self.aggregation[i](hat_ct, context_mask)
        for i in range(self.num_ptr_hops):
            ans_range_prop, zs = self.ptr_net[i](align_ct, context_mask, zs)
        ans_range_prop = ans_range_prop.transpose(0, 1)
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)
        return ans_range_prop, ans_range, vis_param


class RNet(torch.nn.Module):
    """
    R-NET model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, dataset_h5_path):
        super(RNet, self).__init__()
        hidden_size = 45
        hidden_mode = 'GRU'
        dropout_p = 0.2
        emb_dropout_p = 0.1
        enable_layer_norm = False
        word_embedding_size = 300
        char_embedding_size = 64
        encoder_word_layers = 3
        encoder_char_layers = 1
        encoder_bidirection = True
        encoder_direction_num = 2 if encoder_bidirection else 1
        match_lstm_bidirection = True
        self_match_lstm_bidirection = True
        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        self_match_rnn_direction_num = 2 if self_match_lstm_bidirection else 1
        ptr_bidirection = True
        self.enable_search = True
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path, embedding_size=char_embedding_size, trainable=True)
        self.char_encoder = CharEncoder(mode=hidden_mode, input_size=char_embedding_size, hidden_size=hidden_size, num_layers=encoder_char_layers, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
        encode_in_size = word_embedding_size + hidden_size * encoder_direction_num
        self.encoder = MyStackedRNN(mode=hidden_mode, input_size=encode_in_size, hidden_size=hidden_size, num_layers=encoder_word_layers, bidirectional=encoder_bidirection, dropout_p=emb_dropout_p)
        encode_out_size = hidden_size * encoder_direction_num
        self.match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=encode_out_size, hq_input_size=encode_out_size, hidden_size=hidden_size, bidirectional=match_lstm_bidirection, gated_attention=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num
        self.self_match_rnn = MatchRNN(mode=hidden_mode, hp_input_size=match_rnn_out_size, hq_input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=self_match_lstm_bidirection, gated_attention=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * self_match_rnn_direction_num
        self.birnn_after_self = MyRNNBase(mode=hidden_mode, input_size=match_rnn_out_size, hidden_size=hidden_size, bidirectional=True, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        birnn_out_size = hidden_size * 2
        self.pointer_net = BoundaryPointer(mode=hidden_mode, input_size=birnn_out_size, hidden_size=hidden_size, bidirectional=ptr_bidirection, dropout_p=dropout_p, enable_layer_norm=enable_layer_norm)
        self.init_ptr_hidden = AttentionPooling(encode_out_size, hidden_size)

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        assert context_char is not None and question_char is not None
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)
        context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
        question_emb_char, question_char_mask = self.char_embedding.forward(question_char)
        context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
        question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)
        context_vec = torch.cat((context_vec, context_vec_char), dim=-1)
        question_vec = torch.cat((question_vec, question_vec_char), dim=-1)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask, question_encode, question_mask)
        vis_param = {'match': match_para}
        ct_aware_ct, qt_aware_last_hidden, self_para = self.self_match_rnn.forward(qt_aware_ct, context_mask, qt_aware_ct, context_mask)
        vis_param['self'] = self_para
        ct_aware_ct_ag, _ = self.birnn_after_self.forward(ct_aware_ct, context_mask)
        ptr_net_hidden = self.init_ptr_hidden.forward(question_encode, question_mask)
        ans_range_prop = self.pointer_net.forward(ct_aware_ct_ag, context_mask, ptr_net_hidden)
        ans_range_prop = ans_range_prop.transpose(0, 1)
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)
        return ans_range_prop, ans_range, vis_param


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionPooling,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (ForwardNet,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'dropout_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Highway,
     lambda: ([], {'in_size': 4, 'n_layers': 1, 'dropout_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MatchRNNAttention,
     lambda: ([], {'hp_input_size': 4, 'hq_input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PointerAttention,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SFU,
     lambda: ([], {'input_size': 4, 'fusions_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttentionGated,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfGated,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfSeqAtten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (SeqPointer,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (SeqToSeqAtten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_laddie132_Match_LSTM(_paritybench_base):
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

