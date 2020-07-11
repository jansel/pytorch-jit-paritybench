import sys
_module = sys.modules[__name__]
del sys
layers = _module
model = _module
rnn_reader = _module
utils = _module
interact = _module
prepro = _module
train = _module

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


import torch.nn.functional as F


from torch.autograd import Variable


import random


import torch.optim as optim


import numpy as np


import logging


import time


import re


import math


import string


from collections import Counter


class StackedBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM, concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input, p=self.dropout_rate, training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0), x_mask.size(1) - output.size(1), output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            alpha = F.log_softmax(xWy, dim=1)
        else:
            alpha = F.softmax(xWy, dim=1)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None):
        super(RnnDocReader, self).__init__()
        self.opt = opt
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                self.embedding.weight.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                offset = self.opt['tune_partial'] + 2

                def embedding_hook(grad, offset=offset):
                    grad[offset:] = 0
                    return grad
                self.embedding.weight.register_hook(embedding_hook)
        else:
            self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']
        self.doc_rnn = layers.StackedBRNN(input_size=doc_input_size, hidden_size=opt['hidden_size'], num_layers=opt['doc_layers'], dropout_rate=opt['dropout_rnn'], dropout_output=opt['dropout_rnn_output'], concat_layers=opt['concat_rnn_layers'], rnn_type=self.RNN_TYPES[opt['rnn_type']], padding=opt['rnn_padding'])
        self.question_rnn = layers.StackedBRNN(input_size=opt['embedding_dim'], hidden_size=opt['hidden_size'], num_layers=opt['question_layers'], dropout_rate=opt['dropout_rnn'], dropout_output=opt['dropout_rnn_output'], concat_layers=opt['concat_rnn_layers'], rnn_type=self.RNN_TYPES[opt['rnn_type']], padding=opt['rnn_padding'])
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)
        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'], training=self.training)
        drnn_input_list = [x1_emb, x1_f]
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            drnn_input_list.append(x1_pos)
        if self.opt['ner']:
            drnn_input_list.append(x1_ner)
        drnn_input = torch.cat(drnn_input_list, 2)
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores

