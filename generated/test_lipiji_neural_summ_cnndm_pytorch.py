import sys
_module = sys.modules[__name__]
del sys
bleu = _module
configs = _module
data = _module
gru_dec = _module
lstm_dec_v1 = _module
lstm_dec_v2 = _module
main = _module
model = _module
prepare_data = _module
prepare_rouge = _module
utils_pg = _module
word_prob_layer = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch as T


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


import time


import numpy as np


import copy


import random


from random import shuffle


def init_bias(b):
    nn.init.constant_(b, 0.0)


def init_ortho_weight(w):
    nn.init.orthogonal_(w)


class GRUAttentionDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, ctx_size, device, copy,
        coverage, is_predicting):
        super(GRUAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.is_predicting = is_predicting
        self.device = device
        self.copy = copy
        self.coverage = coverage
        self.W = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.
            input_size))
        self.U = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.
            hidden_size))
        self.b = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        self.Wx = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Ux = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)
            )
        self.bx = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))
        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, self.
            hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))
        self.U_nl = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.
            hidden_size))
        self.b_nl = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        self.Ux_nl = nn.Parameter(torch.Tensor(self.hidden_size, self.
            hidden_size))
        self.bx_nl = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Wc = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.
            ctx_size))
        self.Wcx = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        if self.coverage:
            self.W_coverage = nn.Parameter(torch.Tensor(self.ctx_size, 1))
        self.init_weights()

    def init_weights(self):
        init_ortho_weight(self.W)
        init_ortho_weight(self.U)
        init_bias(self.b)
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)
        init_ortho_weight(self.U_nl)
        init_bias(self.b_nl)
        init_ortho_weight(self.Ux_nl)
        init_bias(self.bx_nl)
        init_ortho_weight(self.Wc)
        init_ortho_weight(self.Wcx)
        if self.coverage:
            init_ortho_weight(self.W_coverage)

    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None,
        init_coverage=None):

        def _get_word_atten(pctx, h1, x_mask, acc_att=None):
            if acc_att is not None:
                h = F.linear(h1, self.W_comb_att) + F.linear(T.transpose(
                    acc_att, 0, 1).unsqueeze(2), self.W_coverage)
            else:
                h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)
            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim=True)[0]
                ) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim=True)
            word_atten = word_atten / sum_word_atten
            return word_atten

        def recurrence(x, xx, y_mask, pre_h, pctx, context, x_mask, acc_att
            =None):
            tmp1 = T.sigmoid(F.linear(pre_h, self.U) + x)
            r1, u1 = tmp1.chunk(2, 1)
            h1 = T.tanh(F.linear(pre_h * r1, self.Ux) + xx)
            h1 = u1 * pre_h + (1.0 - u1) * h1
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
            if self.coverage:
                word_atten = _get_word_atten(pctx, h1, x_mask, acc_att)
            else:
                word_atten = _get_word_atten(pctx, h1, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)
            tmp2 = T.sigmoid(F.linear(atted_ctx, self.Wc) + F.linear(h1,
                self.U_nl) + self.b_nl)
            r2, u2 = tmp2.chunk(2, 1)
            h2 = T.tanh(F.linear(atted_ctx, self.Wcx) + F.linear(h1 * r2,
                self.Ux_nl) + self.bx_nl)
            h2 = u2 * h1 + (1.0 - u2) * h2
            h2 = y_mask * h2 + (1.0 - y_mask) * h1
            word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1
                )
            if self.coverage:
                acc_att += word_atten_
                return h2, h2, atted_ctx, word_atten_, acc_att
            else:
                return h2, h2, atted_ctx, word_atten_
        hs, ss, atts, dists, xids, cs = [], [], [], [], [], []
        hidden = init_state
        acc_att = init_coverage
        if self.copy:
            xid = T.transpose(xid, 0, 1)
        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = F.linear(y_emb, self.W, self.b)
        xx = F.linear(y_emb, self.Wx, self.bx)
        steps = range(y_emb.size(0))
        for i in steps:
            if self.coverage:
                cs += [acc_att]
                hidden, s, att, att_dist, acc_att = recurrence(x[i], xx[i],
                    y_mask[i], hidden, pctx, context, x_mask, acc_att)
            else:
                hidden, s, att, att_dist = recurrence(x[i], xx[i], y_mask[i
                    ], hidden, pctx, context, x_mask)
            hs += [hidden]
            ss += [s]
            atts += [att]
            dists += [att_dist]
            xids += [xid]
        if self.coverage:
            if self.is_predicting:
                cs += [acc_att]
                cs = cs[1:]
            cs = T.stack(cs).view(y_emb.size(0), *cs[0].size())
        hs = T.stack(hs).view(y_emb.size(0), *hs[0].size())
        ss = T.stack(ss).view(y_emb.size(0), *ss[0].size())
        atts = T.stack(atts).view(y_emb.size(0), *atts[0].size())
        dists = T.stack(dists).view(y_emb.size(0), *dists[0].size())
        if self.copy:
            xids = T.stack(xids).view(y_emb.size(0), *xids[0].size())
        if self.copy and self.coverage:
            return hs, ss, atts, dists, xids, cs
        elif self.copy:
            return hs, ss, atts, dists, xids
        elif self.coverage:
            return hs, ss, atts, dists, cs
        else:
            return hs, ss, atts


def init_lstm_weight(lstm):
    for param in lstm.parameters():
        if len(param.shape) >= 2:
            init_ortho_weight(param.data)
        else:
            init_bias(param.data)


class LSTMAttentionDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, ctx_size, device, copy,
        coverage, is_predicting):
        super(LSTMAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.is_predicting = is_predicting
        self.device = device
        self.copy = copy
        self.coverage = coverage
        self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))
        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, 2 * self
            .hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))
        if self.coverage:
            self.W_coverage = nn.Parameter(torch.Tensor(self.ctx_size, 1))
        self.init_weights()

    def init_weights(self):
        init_lstm_weight(self.lstm_1)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)
        if self.coverage:
            init_ortho_weight(self.W_coverage)

    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None,
        init_coverage=None):

        def _get_word_atten(pctx, h1, x_mask, acc_att=None):
            if acc_att is not None:
                h = F.linear(h1, self.W_comb_att) + F.linear(T.transpose(
                    acc_att, 0, 1).unsqueeze(2), self.W_coverage)
            else:
                h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)
            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim=True)[0]
                ) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim=True)
            word_atten = word_atten / sum_word_atten
            return word_atten

        def recurrence(x, y_mask, hidden, pctx, context, x_mask, acc_att=None):
            pre_h, pre_c = hidden
            h1, c1 = self.lstm_1(x, hidden)
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
            c1 = y_mask * c1 + (1.0 - y_mask) * pre_c
            s = T.cat((h1.view(-1, self.hidden_size), c1.view(-1, self.
                hidden_size)), 1)
            if self.coverage:
                word_atten = _get_word_atten(pctx, s, x_mask, acc_att)
            else:
                word_atten = _get_word_atten(pctx, s, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)
            word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1
                )
            if self.coverage:
                acc_att += word_atten_
                return (h1, c1), h1, atted_ctx, word_atten_, acc_att
            else:
                return (h1, c1), h1, atted_ctx, word_atten_
        hs, cs, ss, atts, dists, xids, Cs = [], [], [], [], [], [], []
        hidden = init_state
        acc_att = init_coverage
        if self.copy:
            xid = T.transpose(xid, 0, 1)
        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = y_emb
        steps = range(y_emb.size(0))
        for i in steps:
            if self.coverage:
                Cs += [acc_att]
                hidden, s, att, att_dist, acc_att = recurrence(x[i], y_mask
                    [i], hidden, pctx, context, x_mask, acc_att)
            else:
                hidden, s, att, att_dist = recurrence(x[i], y_mask[i],
                    hidden, pctx, context, x_mask)
            hs += [hidden[0]]
            cs += [hidden[1]]
            ss += [s]
            atts += [att]
            dists += [att_dist]
            xids += [xid]
        if self.coverage:
            if self.is_predicting:
                Cs += [acc_att]
                Cs = Cs[1:]
            Cs = T.stack(Cs).view(y_emb.size(0), *Cs[0].size())
        hs = T.stack(hs).view(y_emb.size(0), *hs[0].size())
        cs = T.stack(cs).view(y_emb.size(0), *cs[0].size())
        ss = T.stack(ss).view(y_emb.size(0), *ss[0].size())
        atts = T.stack(atts).view(y_emb.size(0), *atts[0].size())
        dists = T.stack(dists).view(y_emb.size(0), *dists[0].size())
        if self.copy:
            xids = T.stack(xids).view(y_emb.size(0), *xids[0].size())
        if self.copy and self.coverage:
            return (hs, cs), ss, atts, dists, xids, Cs
        elif self.copy:
            return (hs, cs), ss, atts, dists, xids
        elif self.coverage:
            return (hs, cs), ss, atts, dists, Cs
        else:
            return (hs, cs), ss, atts


class LSTMAttentionDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, ctx_size, device, copy,
        coverage, is_predicting):
        super(LSTMAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.is_predicting = is_predicting
        self.device = device
        self.copy = copy
        self.coverage = coverage
        self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)
        self.Wx = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.
            ctx_size))
        self.Ux = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.
            hidden_size))
        self.bx = nn.Parameter(torch.Tensor(4 * self.hidden_size))
        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))
        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, 2 * self
            .hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))
        if self.coverage:
            self.W_coverage = nn.Parameter(torch.Tensor(self.ctx_size, 1))
        self.init_weights()

    def init_weights(self):
        init_lstm_weight(self.lstm_1)
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)
        if self.coverage:
            init_ortho_weight(self.W_coverage)

    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None,
        init_coverage=None):

        def _get_word_atten(pctx, h1, x_mask, acc_att=None):
            if acc_att is not None:
                h = F.linear(h1, self.W_comb_att) + F.linear(T.transpose(
                    acc_att, 0, 1).unsqueeze(2), self.W_coverage)
            else:
                h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)
            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim=True)[0]
                ) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim=True)
            word_atten = word_atten / sum_word_atten
            return word_atten

        def recurrence(x, y_mask, hidden, pctx, context, x_mask, acc_att=None):
            pre_h, pre_c = hidden
            h1, c1 = self.lstm_1(x, hidden)
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
            c1 = y_mask * c1 + (1.0 - y_mask) * pre_c
            s = T.cat((h1.view(-1, self.hidden_size), c1.view(-1, self.
                hidden_size)), 1)
            if self.coverage:
                word_atten = _get_word_atten(pctx, s, x_mask, acc_att)
            else:
                word_atten = _get_word_atten(pctx, s, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)
            ifoc_preact = F.linear(h1, self.Ux) + F.linear(atted_ctx, self.
                Wx, self.bx)
            x4i, x4f, x4o, x4c = ifoc_preact.chunk(4, 1)
            i = torch.sigmoid(x4i)
            f = torch.sigmoid(x4f)
            o = torch.sigmoid(x4o)
            c2 = f * c1 + i * torch.tanh(x4c)
            h2 = o * torch.tanh(c2)
            c2 = y_mask * c2 + (1.0 - y_mask) * c1
            h2 = y_mask * h2 + (1.0 - y_mask) * h1
            word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1
                )
            if self.coverage:
                acc_att += word_atten_
                return (h2, c2), h2, atted_ctx, word_atten_, acc_att
            else:
                return (h2, c2), h2, atted_ctx, word_atten_
        hs, cs, ss, atts, dists, xids, Cs = [], [], [], [], [], [], []
        hidden = init_state
        acc_att = init_coverage
        if self.copy:
            xid = T.transpose(xid, 0, 1)
        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = y_emb
        steps = range(y_emb.size(0))
        for i in steps:
            if self.coverage:
                Cs += [acc_att]
                hidden, s, att, att_dist, acc_att = recurrence(x[i], y_mask
                    [i], hidden, pctx, context, x_mask, acc_att)
            else:
                hidden, s, att, att_dist = recurrence(x[i], y_mask[i],
                    hidden, pctx, context, x_mask)
            hs += [hidden[0]]
            cs += [hidden[1]]
            ss += [s]
            atts += [att]
            dists += [att_dist]
            xids += [xid]
        if self.coverage:
            if self.is_predicting:
                Cs += [acc_att]
                Cs = Cs[1:]
            Cs = T.stack(Cs).view(y_emb.size(0), *Cs[0].size())
        hs = T.stack(hs).view(y_emb.size(0), *hs[0].size())
        cs = T.stack(cs).view(y_emb.size(0), *cs[0].size())
        ss = T.stack(ss).view(y_emb.size(0), *ss[0].size())
        atts = T.stack(atts).view(y_emb.size(0), *atts[0].size())
        dists = T.stack(dists).view(y_emb.size(0), *dists[0].size())
        if self.copy:
            xids = T.stack(xids).view(y_emb.size(0), *xids[0].size())
        if self.copy and self.coverage:
            return (hs, cs), ss, atts, dists, xids, Cs
        elif self.copy:
            return (hs, cs), ss, atts, dists, xids
        elif self.coverage:
            return (hs, cs), ss, atts, dists, Cs
        else:
            return (hs, cs), ss, atts


def init_gru_weight(gru):
    for param in gru.parameters():
        if len(param.shape) >= 2:
            init_ortho_weight(param.data)
        else:
            init_bias(param.data)


def init_xavier_weight(w):
    nn.init.xavier_normal_(w)


def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)


def init_uniform_weight(w):
    nn.init.uniform_(w, -0.1, 0.1)


class Model(nn.Module):

    def __init__(self, modules, consts, options):
        super(Model, self).__init__()
        self.has_learnable_w2v = options['has_learnable_w2v']
        self.is_predicting = options['is_predicting']
        self.is_bidirectional = options['is_bidirectional']
        self.beam_decoding = options['beam_decoding']
        self.cell = options['cell']
        self.device = options['device']
        self.copy = options['copy']
        self.coverage = options['coverage']
        self.avg_nll = options['avg_nll']
        self.dim_x = consts['dim_x']
        self.dim_y = consts['dim_y']
        self.len_x = consts['len_x']
        self.len_y = consts['len_y']
        self.hidden_size = consts['hidden_size']
        self.dict_size = consts['dict_size']
        self.pad_token_idx = consts['pad_token_idx']
        self.ctx_size = (self.hidden_size * 2 if self.is_bidirectional else
            self.hidden_size)
        self.w_rawdata_emb = nn.Embedding(self.dict_size, self.dim_x, self.
            pad_token_idx)
        if self.cell == 'gru':
            self.encoder = nn.GRU(self.dim_x, self.hidden_size,
                bidirectional=self.is_bidirectional)
            self.decoder = GRUAttentionDecoder(self.dim_y, self.hidden_size,
                self.ctx_size, self.device, self.copy, self.coverage, self.
                is_predicting)
        else:
            self.encoder = nn.LSTM(self.dim_x, self.hidden_size,
                bidirectional=self.is_bidirectional)
            self.decoder = LSTMAttentionDecoder(self.dim_y, self.
                hidden_size, self.ctx_size, self.device, self.copy, self.
                coverage, self.is_predicting)
        self.get_dec_init_state = nn.Linear(self.ctx_size, self.hidden_size)
        self.word_prob = WordProbLayer(self.hidden_size, self.ctx_size,
            self.dim_y, self.dict_size, self.device, self.copy, self.coverage)
        self.init_weights()

    def init_weights(self):
        init_uniform_weight(self.w_rawdata_emb.weight)
        if self.cell == 'gru':
            init_gru_weight(self.encoder)
        else:
            init_lstm_weight(self.encoder)
        init_linear_weight(self.get_dec_init_state)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -T.log(T.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 0) / T.sum(y_mask, 0)
        else:
            cost = T.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return T.mean(cost)

    def encode(self, x, len_x, mask_x):
        self.encoder.flatten_parameters()
        emb_x = self.w_rawdata_emb(x)
        emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x)
        hs, hn = self.encoder(emb_x, None)
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)
        dec_init_state = T.sum(hs * mask_x, 0) / T.sum(mask_x, 0)
        dec_init_state = T.tanh(self.get_dec_init_state(dec_init_state))
        return hs, dec_init_state

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None,
        max_ext_len=None, acc_att=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((1, batch_size, self.dim_y))).to(self.
                device)
        else:
            y_emb = self.w_rawdata_emb(y)
        mask_y = Variable(T.ones((1, batch_size, 1))).to(self.device)
        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(
                y_emb, hs, dec_init_state, mask_x, mask_y, x, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_emb
                , hs, dec_init_state, mask_x, mask_y, xid=x)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(y_emb,
                hs, dec_init_state, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_emb, hs,
                dec_init_state, mask_x, mask_y)
        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_emb,
                att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb)
        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs

    def forward(self, x, len_x, y, mask_x, mask_y, x_ext, y_ext, max_ext_len):
        hs, dec_init_state = self.encode(x, len_x, mask_x)
        y_emb = self.w_rawdata_emb(y)
        y_shifted = y_emb[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).
            to(self.device), y_shifted), 0)
        h0 = dec_init_state
        if self.cell == 'lstm':
            h0 = dec_init_state, dec_init_state
        if self.coverage:
            acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(
                self.device)
        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(
                y_shifted, hs, h0, mask_x, mask_y, x_ext, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(
                y_shifted, hs, h0, mask_x, mask_y, xid=x_ext)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(
                y_shifted, hs, h0, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_shifted, hs, h0,
                mask_x, mask_y)
        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_shifted,
                att_dist, xids, max_ext_len)
            cost = self.nll_loss(y_pred, y_ext, mask_y, self.avg_nll)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_shifted)
            cost = self.nll_loss(y_pred, y, mask_y, self.avg_nll)
        if self.coverage:
            cost_c = T.mean(T.sum(T.min(att_dist, C), 2))
            return y_pred, cost, cost_c
        else:
            return y_pred, cost, None


class WordProbLayer(nn.Module):

    def __init__(self, hidden_size, ctx_size, dim_y, dict_size, device,
        copy, coverage):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.dim_y = dim_y
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.coverage = coverage
        self.w_ds = nn.Parameter(torch.Tensor(self.hidden_size, self.
            hidden_size + self.ctx_size + self.dim_y))
        self.b_ds = nn.Parameter(torch.Tensor(self.hidden_size))
        self.w_logit = nn.Parameter(torch.Tensor(self.dict_size, self.
            hidden_size))
        self.b_logit = nn.Parameter(torch.Tensor(self.dict_size))
        if self.copy:
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_size + self.
                ctx_size + self.dim_y))
            self.bv = nn.Parameter(torch.Tensor(1))
        self.init_weights()

    def init_weights(self):
        init_xavier_weight(self.w_ds)
        init_bias(self.b_ds)
        init_xavier_weight(self.w_logit)
        init_bias(self.b_logit)
        if self.copy:
            init_xavier_weight(self.v)
            init_bias(self.bv)

    def forward(self, ds, ac, y_emb, att_dist=None, xids=None, max_ext_len=None
        ):
        h = T.cat((ds, ac, y_emb), 2)
        logit = T.tanh(F.linear(h, self.w_ds, self.b_ds))
        logit = F.linear(logit, self.w_logit, self.b_logit)
        y_dec = T.softmax(logit, dim=2)
        if self.copy:
            if max_ext_len > 0:
                ext_zeros = Variable(torch.zeros(y_dec.size(0), y_dec.size(
                    1), max_ext_len)).to(self.device)
                y_dec = T.cat((y_dec, ext_zeros), 2)
            g = T.sigmoid(F.linear(h, self.v, self.bv))
            y_dec = (g * y_dec).scatter_add(2, xids, (1 - g) * att_dist)
        return y_dec


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lipiji_neural_summ_cnndm_pytorch(_paritybench_base):
    pass
