import sys
_module = sys.modules[__name__]
del sys
data_util = _module
batcher = _module
config = _module
data = _module
utils = _module
training_ptr_gen = _module
decode = _module
eval = _module
model = _module
train = _module
train_util = _module
transformer_encoder = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from numpy import random


import time


from torch.nn.utils import clip_grad_norm_


from torch.optim import Adagrad


import logging


import math


_global_config['rand_unif_init_mag'] = 4


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.
                    rand_unif_init_mag)
            elif name.startswith('bias_'):
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.0)
                bias.data[start:end].fill_(1.0)


_global_config['trunc_norm_init_std'] = 4


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


_global_config['vocab_size'] = 4


_global_config['hidden_dim'] = 4


_global_config['emb_dim'] = 4


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2,
            bias=False)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)
        encoder_outputs = encoder_outputs.contiguous()
        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)
        return encoder_outputs, encoder_feature, hidden


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


class ReduceState(nn.Module):

    def __init__(self):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        return hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)


_global_config['is_coverage'] = 4


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.
            hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature,
        enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat)
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()
        dec_fea_expanded = dec_fea_expanded.view(-1, n)
        att_features = encoder_feature + dec_fea_expanded
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)
            coverage_feature = self.W_c(coverage_input)
            att_features = att_features + coverage_feature
        e = F.tanh(att_features)
        scores = self.v(e)
        scores = scores.view(-1, t_k)
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        attn_dist = attn_dist.unsqueeze(1)
        c_t = torch.bmm(attn_dist, encoder_outputs)
        c_t = c_t.view(-1, config.hidden_dim * 2)
        attn_dist = attn_dist.view(-1, t_k)
        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist
        return c_t, attn_dist, coverage


_global_config['pointer_gen'] = 4


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim,
            config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1,
            batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)
        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.
                emb_dim, 1)
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature,
        enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab,
        coverage, step):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                c_decoder.view(-1, config.hidden_dim)), 1)
            c_t, _, coverage_next = self.attention_network(s_t_hat,
                encoder_outputs, encoder_feature, enc_padding_mask, coverage)
            coverage = coverage_next
        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)
        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
            c_decoder.view(-1, config.hidden_dim)), 1)
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat,
            encoder_outputs, encoder_feature, enc_padding_mask, coverage)
        if self.training or step > 0:
            coverage = coverage_next
        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)
        output = self.out1(output)
        output = self.out2(output)
        vocab_dist = F.softmax(output, dim=1)
        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab,
                attn_dist_)
        else:
            final_dist = vocab_dist
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.
            log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.h = num_head
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask):
        nbatches = query.size(0)
        query = self.linear_query(query).view(nbatches, -1, self.h, self.d_k
            ).transpose(1, 2)
        key = self.linear_key(key).view(nbatches, -1, self.h, self.d_k
            ).transpose(1, 2)
        value = self.linear_value(value).view(nbatches, -1, self.h, self.d_k
            ).transpose(1, 2)
        mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k
            )
        return self.linear_out(x)


class AffineLayer(nn.Module):

    def __init__(self, dropout, d_model, d_ff):
        super(AffineLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, num_head, dropout, d_model, d_ff):
        super(EncoderLayer, self).__init__()
        self.att_layer = MultiHeadedAttention(num_head, d_model, dropout)
        self.norm_att = nn.LayerNorm(d_model)
        self.dropout_att = nn.Dropout(dropout)
        self.affine_layer = AffineLayer(dropout, d_model, d_ff)
        self.norm_affine = nn.LayerNorm(d_model)
        self.dropout_affine = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_att = self.norm_att(x * mask)
        x_att = self.att_layer(x_att, x_att, x_att, mask)
        x = x + self.dropout_att(x_att)
        x_affine = self.norm_affine(x * mask)
        x_affine = self.affine_layer(x_affine)
        return x + self.dropout_affine(x_affine)


class Encoder(nn.Module):

    def __init__(self, N, num_head, dropout, d_model, d_ff):
        super(Encoder, self).__init__()
        self.position = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList()
        for _ in range(N):
            self.layers.append(EncoderLayer(num_head, dropout, d_model, d_ff))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, word_embed, mask):
        x = self.position(word_embed)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x * mask)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_atulkum_pointer_summarizer(_paritybench_base):
    pass
    def test_000(self):
        self._check(AffineLayer(*[], **{'dropout': 0.5, 'd_model': 4, 'd_ff': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Encoder(*[], **{'N': 4, 'num_head': 4, 'dropout': 0.5, 'd_model': 4, 'd_ff': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(EncoderLayer(*[], **{'num_head': 4, 'dropout': 0.5, 'd_model': 4, 'd_ff': 4}), [torch.rand([4, 4]), torch.rand([4, 1])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MultiHeadedAttention(*[], **{'num_head': 4, 'd_model': 4}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 1])], {})

    def test_004(self):
        self._check(PositionalEncoding(*[], **{'d_model': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

