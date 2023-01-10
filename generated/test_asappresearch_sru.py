import sys
_module = sys.modules[__name__]
del sys
layers = _module
model = _module
rnn_reader = _module
utils = _module
prepro = _module
train = _module
dataloader = _module
modules = _module
train_classifier = _module
train_enwik8 = _module
train_lm = _module
compare_cpu_speed_sru_gru = _module
compare_gpu_speed_sru_gru = _module
test_backward_with_transpose = _module
test_impl = _module
test_mm = _module
test_multigpu = _module
test_sru = _module
setup = _module
sru = _module
cuda_functional = _module
modules = _module
ops = _module
version = _module
build_artifact = _module
test_regression = _module
test_sru = _module
test_torchscript = _module
test_amp = _module
test_ts_cpp = _module

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


import torch.optim as optim


import numpy as np


import logging


import re


import random


import string


from collections import Counter


import pandas as pd


import time


import math


from torch import nn


from typing import Optional


from typing import Tuple


import warnings


from torch import Tensor


from torch.autograd import Function


from torch.utils.cpp_extension import load


import copy


from typing import List


from typing import Union


from torch.nn.utils.rnn import PackedSequence


from typing import Any


from torch import optim


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
            self.rnns.append(sru.SRUCell(input_size, hidden_size, dropout=dropout_rate, rnn_dropout=dropout_rate, use_tanh=0, bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output.contiguous()

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        lengths = x_mask.data.eq(0).long().sum(1)
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
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
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
            alpha = F.log_softmax(xWy)
        else:
            alpha = F.softmax(xWy)
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
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


def normalize_emb_(data):
    None
    norms = data.norm(2, 1) + 1e-08
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))
    None


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None, normalize_emb=False):
        super(RnnDocReader, self).__init__()
        self.opt = opt
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0), embedding.size(1), padding_idx=padding_idx)
            if normalize_emb:
                normalize_emb_(embedding)
            self.embedding.weight.data = embedding
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:
            self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            if normalize_emb:
                normalize_emb_(self.pos_embedding.weight.data)
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            if normalize_emb:
                normalize_emb_(self.ner_embedding.weight.data)
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']
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
            x1_pos_emb = self.pos_embedding(x1_pos)
            if self.opt['dropout_emb'] > 0:
                x1_pos_emb = nn.functional.dropout(x1_pos_emb, p=self.opt['dropout_emb'], training=self.training)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            if self.opt['dropout_emb'] > 0:
                x1_ner_emb = nn.functional.dropout(x1_ner_emb, p=self.opt['dropout_emb'], training=self.training)
            drnn_input_list.append(x1_ner_emb)
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


class CNN_Text(nn.Module):

    def __init__(self, n_in, widths=[3, 4, 5], filters=100):
        super(CNN_Text, self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x


class EmbeddingLayer(nn.Module):

    def __init__(self, n_d, words, fix_emb=False):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        for w in words:
            if w not in word2id:
                word2id[w] = len(word2id)
        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.embedding = nn.Embedding(self.n_V, n_d)

    def forward(self, x):
        return self.embedding(x)

    def map_to_ids(self, text):
        return np.asarray([self.word2id[x] for x in text], dtype='int64')


class Model(nn.Module):

    def __init__(self, rnn):
        super(Model, self).__init__()
        self.rnn = rnn

    def forward(self, x):
        out, state = self.rnn(x)
        return out[-1:]


@torch.jit.unused
def elementwise_recurrence_gpu(U: Tensor, x: Tensor, weight_c: Tensor, bias: Tensor, c_init: Tensor, activation_type: int, hidden_size: int, bidirectional: bool, has_skip_term: bool, scale_x: Optional[Tensor]=None, dropout_mask_c: Optional[Tensor]=None, mask_pad: Optional[Tensor]=None, amp_recurrence_fp16: bool=False) ->List[Tensor]:
    """Elementwise forward operation of SRU on GPU.

    """
    if amp_recurrence_fp16 and U.dtype == torch.float16:
        cast = torch.Tensor.half
    else:
        cast = torch.Tensor.float
    U = cast(U)
    x = cast(x)
    weight_c = cast(weight_c)
    bias = cast(bias)
    c_init = cast(c_init)
    scale_x = cast(scale_x) if scale_x is not None else scale_x
    dropout_mask_c = cast(dropout_mask_c) if dropout_mask_c is not None else dropout_mask_c
    return ElementwiseRecurrence.apply(U, x, weight_c, bias, c_init, activation_type, hidden_size, bidirectional, has_skip_term, scale_x, dropout_mask_c, mask_pad)


@torch.jit.unused
def elementwise_recurrence_naive(U: Tensor, x: Tensor, weight_c: Tensor, bias: Tensor, c_init: Tensor, activation_type: int, hidden_size: int, bidirectional: bool, has_skip_term: bool, scale_x: Optional[Tensor]=None, dropout_mask_c: Optional[Tensor]=None, mask_pad: Optional[Tensor]=None) ->List[Tensor]:
    """Elementwise forward operation of SRU in pure Python.

    """
    if torch.is_grad_enabled():
        warnings.warn('Running SRU on CPU with grad_enabled=True. Are you sure?')
    else:
        return elementwise_recurrence_inference(U, x, weight_c, bias, c_init, activation_type, hidden_size, bidirectional, has_skip_term, scale_x, dropout_mask_c, mask_pad)
    bidir = 2 if bidirectional else 1
    length = x.size(0) if x.dim() == 3 else 1
    batch = x.size(-2)
    k = U.size(-1) // hidden_size // bidir
    d = hidden_size
    is_custom = weight_c.dim() > 1
    U = U.contiguous().view(length, batch, bidir, d, k)
    if is_custom:
        weight_c = weight_c.view(length, batch, bidir, d, 2)
        forget_wc = weight_c[..., 0]
        reset_wc = weight_c[..., 1]
    else:
        forget_wc, reset_wc = weight_c.view(2, bidir, d)
    forget_bias, reset_bias = bias.view(2, bidir, d)
    if not has_skip_term:
        x_prime = None
    elif k == 3:
        x_prime = x.view(length, batch, bidir, d)
        x_prime = x_prime * scale_x if scale_x is not None else x_prime
    else:
        x_prime = U[..., 3]
    if c_init is None:
        c_init = x.new_zeros(size=(batch, bidir, d))
    else:
        c_init = c_init.view(batch, bidir, d)
    mask_pad = mask_pad.view(length, batch, 1).float() if mask_pad is not None else None
    mask_c = dropout_mask_c.view(batch, bidir, d) if dropout_mask_c is not None else None
    h = x.new_zeros(length, batch, bidir, d)
    c_final = []
    for di in range(bidir):
        time_seq = range(length) if di == 0 else range(length - 1, -1, -1)
        mask_c_ = 1 if mask_c is None else mask_c[:, di, :]
        c_prev = c_init[:, di, :]
        fb, rb = forget_bias[di], reset_bias[di]
        if is_custom:
            fw = forget_wc[:, :, di, :].chunk(length)
            rw = reset_wc[:, :, di, :].chunk(length)
        else:
            fw = forget_wc[di].expand(batch, d)
            rw = reset_wc[di].expand(batch, d)
        u0 = U[:, :, di, :, 0].chunk(length)
        u1 = (U[:, :, di, :, 1] + fb).chunk(length)
        u2 = (U[:, :, di, :, 2] + rb).chunk(length)
        if x_prime is not None:
            xp = x_prime[:, :, di, :].chunk(length)
        for t in time_seq:
            if is_custom:
                forget_t = (u1[t] + c_prev * fw[t]).sigmoid()
                reset_t = (u2[t] + c_prev * rw[t]).sigmoid()
            else:
                forget_t = (u1[t] + c_prev * fw).sigmoid()
                reset_t = (u2[t] + c_prev * rw).sigmoid()
            c_t = u0[t] + (c_prev - u0[t]) * forget_t
            if mask_pad is not None:
                c_t = c_t * (1 - mask_pad[t]) + c_prev * mask_pad[t]
            c_prev = c_t
            if activation_type == 0:
                g_c_t = c_t
            elif activation_type == 1:
                g_c_t = c_t.tanh()
            else:
                raise ValueError('Activation type must be 0 or 1, not {}'.format(activation_type))
            if x_prime is not None:
                h_t = xp[t] + (g_c_t - xp[t]) * mask_c_ * reset_t
            else:
                h_t = g_c_t * mask_c_ * reset_t
            if mask_pad is not None:
                h_t = h_t * (1 - mask_pad[t])
            h[t, :, di, :] = h_t
        c_final.append(c_t.view(batch, d))
    return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)


class SRUCell(nn.Module):
    """
    A single SRU layer as per `LSTMCell`, `GRUCell` in Pytorch.
    """
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'rnn_dropout', 'dropout', 'bidirectional', 'has_skip_term', 'highway_bias', 'v1', 'rescale', 'activation_type', 'activation', 'custom_m', 'projection_size', 'num_matrices', 'layer_norm', 'weight_proj', 'scale_x', 'normalize_after', 'weight_c_init']
    scale_x: Tensor
    weight_proj: Optional[Tensor]

    def __init__(self, input_size: int, hidden_size: int, dropout: float=0.0, rnn_dropout: float=0.0, bidirectional: bool=False, n_proj: int=0, use_tanh: bool=False, highway_bias: float=0.0, has_skip_term: bool=True, layer_norm: bool=False, rescale: bool=True, v1: bool=False, custom_m: Optional[nn.Module]=None, amp_recurrence_fp16: bool=False, normalize_after: bool=False, weight_c_init: Optional[float]=None):
        """Initialize the SRUCell module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        n_proj: int, optional
            if non-zero, factorize the ``weight`` parameter matrix as a
            product of two parameter matrices, using an innder dimension
            ``n_proj`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=True)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        custom_m: nn.Module, optional
            use the give module instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool
            if True use post layer norm, else pre layer norm
        weight_c_init: Optional[float]
            if not None, then size of uniform initiatialization of weight_c
        """
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = float(rnn_dropout)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        self.custom_m: Optional[nn.Module] = custom_m
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        self.amp_recurrence_fp16 = amp_recurrence_fp16
        self.normalize_after = normalize_after
        self.weight_c_init = weight_c_init
        self.projection_size = 0
        if n_proj > 0 and n_proj < self.input_size and n_proj < self.output_size:
            self.projection_size = n_proj
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4
        if self.custom_m is None:
            if self.projection_size == 0:
                self.weight_proj = None
                self.weight = nn.Parameter(torch.Tensor(input_size, self.output_size * self.num_matrices))
            else:
                self.weight_proj = nn.Parameter(torch.Tensor(input_size, self.projection_size))
                self.weight = nn.Parameter(torch.Tensor(self.projection_size, self.output_size * self.num_matrices))
        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.register_buffer('scale_x', torch.FloatTensor([0]))
        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            if normalize_after:
                self.layer_norm = nn.LayerNorm(self.output_size)
            else:
                self.layer_norm = nn.LayerNorm(self.input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Properly initialize the weights of SRU, following the same
        recipe as:
        Xavier init:
            http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        Kaiming init:
            https://arxiv.org/abs/1502.01852

        """
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            scale_val = (1 + math.exp(bias_val) * 2) ** 0.5
            self.scale_x.data[0] = scale_val
        if self.custom_m is None:
            d = self.weight.size(0)
            val_range = (3.0 / d) ** 0.5
            self.weight.data.uniform_(-val_range, val_range)
            if self.projection_size > 0:
                val_range = (3.0 / self.weight_proj.size(0)) ** 0.5
                self.weight_proj.data.uniform_(-val_range, val_range)
            w = self.weight.data.view(d, -1, self.hidden_size, self.num_matrices)
            if self.dropout > 0:
                w[:, :, :, 0].mul_((1 - self.dropout) ** 0.5)
            if self.rnn_dropout > 0:
                w.mul_((1 - self.rnn_dropout) ** 0.5)
            if self.layer_norm:
                w.mul_(0.1)
            if self.rescale and self.has_skip_term and self.num_matrices == 4:
                scale_val = (1 + math.exp(bias_val) * 2) ** 0.5
                w[:, :, :, 3].mul_(scale_val)
        elif hasattr(self.custom_m, 'reset_parameters'):
            self.custom_m.reset_parameters()
        else:
            warnings.warn('Unable to reset parameters for custom module. reset_parameters() method not found for custom module. ' + self.custom_m.__class__.__name__)
        if not self.v1:
            if self.weight_c_init is None:
                self.weight_c.data.uniform_(-3.0 ** 0.5, 3.0 ** 0.5)
                self.weight_c.data.mul_(0.5 ** 0.5)
            else:
                self.weight_c.data.uniform_(-self.weight_c_init, self.weight_c_init)
            if self.custom_m is None:
                w[:, :, :, 1].mul_(0.5 ** 0.5)
                w[:, :, :, 2].mul_(0.5 ** 0.5)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self, input: Tensor, c0: Optional[Tensor]=None, mask_pad: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """The forward method of the SRU layer.
        """
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('Input must be 2 or 3 dimensional')
        batch_size = input.size(-2)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size, dtype=input.dtype, device=input.device)
        residual = input
        if self.layer_norm is not None and not self.normalize_after:
            input = self.layer_norm(input)
        if self.training and self.rnn_dropout > 0:
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None
        mask_c: Optional[Tensor] = None
        if self.training and self.dropout > 0:
            mask_c = self.get_dropout_mask_((batch_size, self.output_size), self.dropout)
        U, V = self.compute_UV(input, c0, mask_pad)
        h, c = self.apply_recurrence(U, V, residual, c0, scale_val, mask_c, mask_pad)
        if self.layer_norm is not None and self.normalize_after:
            h = self.layer_norm(h)
        return h, c

    def apply_recurrence(self, U: Tensor, V: Tensor, residual: Tensor, c0: Tensor, scale_val: Optional[Tensor], mask_c: Optional[Tensor], mask_pad: Optional[Tensor]) ->List[Tensor]:
        """
        Apply the elementwise recurrence computation on given input
        tensors

        """
        if not torch.jit.is_scripting():
            if self.bias.is_cuda:
                return elementwise_recurrence_gpu(U, residual, V, self.bias, c0, self.activation_type, self.hidden_size, self.bidirectional, self.has_skip_term, scale_val, mask_c, mask_pad, self.amp_recurrence_fp16)
            else:
                return elementwise_recurrence_naive(U, residual, V, self.bias, c0, self.activation_type, self.hidden_size, self.bidirectional, self.has_skip_term, scale_val, mask_c, mask_pad)
        else:
            return elementwise_recurrence_inference(U, residual, V, self.bias, c0, self.activation_type, self.hidden_size, self.bidirectional, self.has_skip_term, scale_val, mask_c, mask_pad)

    def compute_UV(self, input: Tensor, c0: Optional[Tensor], mask_pad: Optional[Tensor]) ->Tuple[Tensor, Tensor]:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices).

        When a custom module `custom_m` is given, U will be computed by
        the given module. In addition, the module can return an
        additional tensor V (length, batch_size, output_size * 2) that
        will be added to the hidden-to-hidden coefficient terms in
        sigmoid gates, i.e., (V[t, b, d] + weight_c[d]) * c[t-1].

        """
        if self.custom_m is None:
            U = self.compute_U(input)
            V = self.weight_c
        else:
            ret = self.custom_m(input)
            if isinstance(ret, tuple) or isinstance(ret, list):
                if len(ret) > 2:
                    raise Exception('Custom module must return 1 or 2 tensors but got {}.'.format(len(ret)))
                U, V = ret[0], ret[1] + self.weight_c
            else:
                U, V = ret, self.weight_c
            if U.size(-1) != self.output_size * self.num_matrices:
                raise ValueError('U must have a last dimension of {} but got {}.'.format(self.output_size * self.num_matrices, U.size(-1)))
            if V.size(-1) != self.output_size * 2:
                raise ValueError('V must have a last dimension of {} but got {}.'.format(self.output_size * 2, V.size(-1)))
        return U, V

    def compute_U(self, input: Tensor) ->Tensor:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices)
        """
        x = input if input.dim() == 2 else input.contiguous().view(-1, self.input_size)
        weight_proj = self.weight_proj
        if weight_proj is not None:
            x_projected = x.mm(weight_proj)
            U = x_projected.mm(self.weight)
        else:
            U = x.mm(self.weight)
        return U

    def get_dropout_mask_(self, size: Tuple[int, int], p: float) ->Tensor:
        """
        Composes the dropout mask for the `SRUCell`.
        """
        b = self.bias.data
        return b.new_empty(size).bernoulli_(1 - p).div_(1 - p)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.projection_size > 0:
            s += ', projection_size={projection_size}'
        if self.dropout > 0:
            s += ', dropout={dropout}'
        if self.rnn_dropout > 0:
            s += ', rnn_dropout={rnn_dropout}'
        if self.bidirectional:
            s += ', bidirectional={bidirectional}'
        if self.highway_bias != 0:
            s += ', highway_bias={highway_bias}'
        if self.activation_type != 0:
            s += ', activation={activation}'
        if self.v1:
            s += ', v1={v1}'
        s += ', rescale={rescale}'
        if not self.has_skip_term:
            s += ', has_skip_term={has_skip_term}'
        if self.layer_norm:
            s += ', layer_norm=True'
        if self.custom_m is not None:
            s += ',\n  custom_m=' + str(self.custom_m)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return '{}({})'.format(self.__class__.__name__, s)
        else:
            return '{}({}\n)'.format(self.__class__.__name__, s)


class SRU(nn.Module):
    """
    Implementation of Simple Recurrent Unit (SRU)
    """
    __constants__ = ['input_size', 'hidden_size', 'output_size', 'num_layers', 'dropout', 'rnn_dropout', 'projection_size', 'rnn_lst', 'bidirectional', 'use_layer_norm', 'has_skip_term', 'num_directions', 'nn_rnn_compatible_return', 'input_to_hidden']

    def __init__(self, input_size: int, hidden_size: int, num_layers: int=2, dropout: float=0.0, rnn_dropout: float=0.0, bidirectional: bool=False, projection_size: int=0, use_tanh: bool=False, layer_norm: bool=False, highway_bias: float=0.0, has_skip_term: bool=True, rescale: bool=False, v1: bool=False, nn_rnn_compatible_return: bool=False, custom_m: Optional[Union[nn.Module, List[nn.Module]]]=None, proj_input_to_hidden_first: bool=False, amp_recurrence_fp16: bool=False, normalize_after: bool=False, weight_c_init: Optional[float]=None):
        """Initialize the SRU module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        num_layers: int
            the number of stacked SRU layers (default=2)
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        projection_size: int, optional
            if non-zero, factorize the ``weight`` parameter in each
            layeras a product of two parameter matrices, using an innder
            dimension ``projection_size`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=False)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        custom_m: Union[nn.Module, List[nn.Module]], optional
            use the given module(s) instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation.  The
            module must take input x of shape (seq_len, batch_size,
            hidden_size). It returns a tensor U of shape (seq_len,
            batch_size, hidden_size * num_matrices), and one optional
            tensor V of shape (seq_len, batch_size, hidden_size * 2).
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool
            if True use post layer norm, else use pre layer norm
        weight_c_init: Optional[float]
            if not None, then size of uniform initiatialization of weight_c
        """
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden = None
        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
        else:
            first_layer_input_size = input_size
        self.amp_recurrence_fp16 = amp_recurrence_fp16
        if rnn_dropout > 0:
            warnings.warn('rnn_dropout > 0 is deprecated and will be removed innext major version of SRU. Please use dropout instead.')
        if use_tanh:
            warnings.warn('use_tanh = True is deprecated and will be removed innext major version of SRU.')
        rnn_lst = nn.ModuleList()
        for i in range(num_layers):
            custom_m_i = None
            if custom_m is not None:
                custom_m_i = custom_m[i] if isinstance(custom_m, list) else copy.deepcopy(custom_m)
            layer_i = SRUCell(first_layer_input_size if i == 0 else self.output_size, self.hidden_size, dropout=dropout if i + 1 != num_layers else 0, rnn_dropout=rnn_dropout, bidirectional=bidirectional, n_proj=projection_size, use_tanh=use_tanh, layer_norm=layer_norm, highway_bias=highway_bias, has_skip_term=has_skip_term, rescale=rescale, v1=v1, custom_m=custom_m_i, amp_recurrence_fp16=amp_recurrence_fp16, normalize_after=normalize_after, weight_c_init=weight_c_init)
            rnn_lst.append(layer_i)
        self.rnn_lst = rnn_lst

    def forward(self, input: Tensor, c0: Optional[Tensor]=None, mask_pad: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        """The forward method of SRU module

        Parameters
        ----------
        input: Tensor
            the input feature. shape: (length, batch_size, input_size)
        c0: Tensor, optional
            the initial internal hidden state. shape: (num_layers,
            batch_size, output_size) where
            output_size = hidden_size * num_direction
        mask_pad: Tensor, optional
            the mask where a non-zero value indicates if an input token
            is pad token that should be ignored in forward and backward
            computation. shape: (length, batch_size)

        Returns
        ----------
        h: Tensor
            the output hidden state. shape: (length, batch_size,
            output_size) where
            output_size = hidden_size * num_direction
        c: Tensor
            the last internal hidden state. shape: (num_layers,
            batch_size, output_size), or (num_layers * num_directions,
            batch_size, hidden_size) if `nn_rnn_compatible_return` is
            set `True`

        """
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([([0] * length + [1] * (max_length - length)) for length in lengths.tolist()])
            mask_pad = mask_pad.transpose(0, 1).contiguous()
        if input.dim() != 3:
            raise ValueError('There must be 3 dimensions for (length, batch_size, input_size)')
        if c0 is None:
            zeros = torch.zeros(input.size(1), self.output_size, dtype=input.dtype, device=input.device)
            c0_ = [zeros for i in range(self.num_layers)]
        else:
            if c0.dim() != 3:
                raise ValueError('c0 must be 3 dim (num_layers, batch_size, output_size)')
            c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]
        if self.input_to_hidden is None:
            prevx = input
        else:
            prevx = self.input_to_hidden(input)
        lstc = []
        i = 0
        for rnn in self.rnn_lst:
            h, c = rnn(prevx, c0_[i], mask_pad=mask_pad)
            prevx = h
            lstc.append(c)
            i += 1
        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            batch_size = input.size(1)
            lstc_stack = lstc_stack.view(self.num_layers, batch_size, self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        if isinstance(orig_input, PackedSequence):
            prevx = nn.utils.rnn.pack_padded_sequence(prevx, lengths, enforce_sorted=False)
            return prevx, lstc_stack
        else:
            return prevx, lstc_stack

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        if self.input_to_hidden is not None:
            self.input_to_hidden.reset_parameters()

    def make_backward_compatible(self):
        self.nn_rnn_compatible_return = getattr(self, 'nn_rnn_compatible_return', False)
        if hasattr(self, 'n_in'):
            if len(self.ln_lst):
                raise Exception('Layer norm is not backward compatible for sru<=2.1.7')
            if self.use_weight_norm:
                raise Exception('Weight norm removed in sru>=2.1.9')
            self.input_size = self.n_in
            self.hidden_size = self.n_out
            self.output_size = self.out_size
            self.num_layers = self.depth
            self.projection_size = self.n_proj
            self.use_layer_norm = False
            for cell in self.rnn_lst:
                cell.input_size = cell.n_in
                cell.hidden_size = cell.n_out
                cell.output_size = cell.n_out * 2 if cell.bidirectional else cell.n_out
                cell.num_matrices = cell.k
                cell.projection_size = cell.n_proj
                cell.layer_norm = None
                if cell.activation_type > 1:
                    raise Exception('ReLU or SeLU activation removed in sru>=2.1.9')
        if not hasattr(self, 'input_to_hidden'):
            self.input_to_hidden = None
            for cell in self.rnn_lst:
                cell.custom_m = None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SRU,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (StackedBRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_asappresearch_sru(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

