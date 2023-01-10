import sys
_module = sys.modules[__name__]
del sys
DataLoader = _module
Dataset = _module
predict = _module
random_character_loader = _module
test_broadcast = _module
test_config = _module
test_dataloader = _module
test_global_parameters = _module
test_logger = _module
test_multi_gpu = _module
test_processing = _module
test_save_model = _module
test_train = _module
apply_bpe = _module
train = _module
train_attn_and_ctc = _module
train_multi = _module
Attention = _module
Beam = _module
Constants = _module
Decode = _module
Embedding = _module
Layers = _module
Loss = _module
Models = _module
Optim = _module
SubLayers = _module
Utils = _module
transformer = _module

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


import random


import numpy as np


import torch


import time


import string


import math


import torch.nn as nn


import torch.optim as optim


import torch.utils.data


import torch.utils.data.distributed


import torch.nn.init as init


from torch.autograd import Variable


import torch.nn.functional as func


import logging


import matplotlib.pyplot as plt


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.scaled = math.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Q/K/V [batch_size, time_step, d_model]
        Args:
        Q: queue matrix
        K: key matrix
        V: value matrix
        QK^T:[batch_size, q_time_step, d_model]X[batch_size, d_model, k_time_step]
                        =[batch_size, q_time_step, k_time_step]
        """
        attn = torch.bmm(q, k.transpose(1, 2)).div(self.scaled)
        if mask is not None:
            assert mask.size() == attn.size()
            attn.data.masked_fill_(mask, -float('inf'))
        attn_weights = self.softmax(attn)
        attn_weights = self.dropout(attn_weights)
        output = torch.bmm(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        assert d_v == int(d_model / n_head)
        assert d_k == int(d_model / n_head)
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scaled = math.sqrt(d_k)
        self.linear_q = nn.Linear(d_model, n_head * d_k)
        self.linear_k = nn.Linear(d_model, n_head * d_k)
        self.linear_v = nn.Linear(d_model, n_head * d_v)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        def shape(x):
            return x.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)
        query = shape(query)
        key = shape(key)
        value = shape(value)
        scores = torch.matmul(query, key.transpose(2, 3)).div(self.scaled)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            scores.masked_fill_(mask, -float('inf'))
        attns = self.dropout(self.softmax(scores))
        context = unshape(torch.matmul(attns, value))
        output = self.output_linear(context)
        norm_output = self.layernorm(output + v)
        return norm_output, attns


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=600):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, inputs_length, step=None):
        batch_size = inputs_length.size(0)
        time_steps = inputs_length.max().item()
        if step is None:
            pos_enc = self.pe[:, :time_steps].repeat(batch_size, 1, 1)
        else:
            pos_enc = self.pe[:, step].repeat(batch_size, 1, 1)
        return pos_enc


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-06)
        init.xavier_normal_(self.fc1.weight.data)
        init.xavier_normal_(self.fc2.weight.data)

    def forward(self, inputs):
        relu_output = self.dropout1(self.relu(self.fc1(inputs)))
        ffn_output = self.fc2(relu_output)
        output = self.dropout2(self.layernorm(inputs + ffn_output))
        return output


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, inputs, slf_attn_mask=None):
        attn_output, slf_attn_weight = self.slf_attn(inputs, inputs, inputs, mask=slf_attn_mask)
        pffn_output = self.pos_ffn(attn_output)
        return pffn_output, slf_attn_weight


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, inputs, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        slf_attn_output, slf_attn_weight = self.slf_attn(inputs, inputs, inputs, mask=slf_attn_mask)
        sre_attn_output, sre_attn_weight = self.enc_attn(slf_attn_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        pffn_output = self.pos_ffn(sre_attn_output)
        return pffn_output, (slf_attn_weight, sre_attn_weight)


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, target):
        assert inputs.dim() == 2
        assert target.dim() == 2
        batch_size = inputs.size(0)
        output_dim = inputs.size(1)
        input_log_softmax = self.log_softmax(inputs)
        weight = self.weight.repeat(inputs.size(0), 1)
        tmp = torch.addcmul(torch.zeros(batch_size, output_dim), -1, input_log_softmax, target)
        tmp_weighted = torch.addcmul(torch.zeros(batch_size, output_dim), 1, weight, tmp)
        if self.size_average:
            loss = torch.sum(tmp_weighted).div(batch_size)
        else:
            loss = torch.sum(tmp_weighted)
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, vocab_size, weight=None, size_average=True, ignore_index=-1):
        assert 0.0 <= label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        smoothing_value = label_smoothing / (vocab_size - 1)
        one_hot = torch.full((vocab_size,), smoothing_value)
        if not ignore_index:
            one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.criterion = CrossEntropyLoss(weight=weight, size_average=size_average)

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        if self.padding_idx >= 0:
            model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)
        return self.criterion(output, model_prob)


def padding_info_mask(seq_q_length, seq_k_length):
    """ Indicate the padding-related part to mask """
    assert seq_q_length.dim() == 1 and seq_k_length.dim() == 1
    batch_size = seq_k_length.size(0)
    len_q = seq_q_length.max().item()
    len_k = seq_k_length.max().item()
    mask_mat = []
    for i in range(batch_size):
        max_len = len_k
        length = seq_k_length[i].item()
        mask_mat.append(np.column_stack([np.zeros([1, length]), np.ones([1, max_len - length])]))
    mask_mat = np.row_stack(mask_mat).astype('uint8')
    pad_attn_mask = torch.from_numpy(mask_mat).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    if seq_q_length.is_cuda:
        return pad_attn_mask
    return pad_attn_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, input_size, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64, d_model=512, d_inner_hid=1024, dropout=0.1, emb_scale=1):
        super(Encoder, self).__init__()
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.emb_scale = emb_scale
        self.position_enc = PositionalEncoding(dropout, d_model, self.n_max_seq)
        self.input_proj = nn.Sequential(nn.Linear(input_size, d_model, bias=True), nn.ReLU(), nn.Dropout(), nn.LayerNorm(d_model, eps=1e-06))
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, inputs, inputs_length, return_attns=False):
        enc_input = self.input_proj(inputs)
        pos_emb = self.position_enc(inputs_length)
        enc_input += pos_emb
        enc_slf_attn_mask = padding_info_mask(inputs_length, inputs_length)
        enc_slf_attns = []
        enc_output = enc_input
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        return enc_output, enc_slf_attns


def feature_info_mask(seq_length):
    """ Get an attention mask to avoid using the subsequent info."""
    assert seq_length.dim() == 1
    batch_size = seq_length.size(0)
    max_len = seq_length.max().item()
    attn_shape = batch_size, max_len, max_len
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq_length.is_cuda:
        return subsequent_mask
    return subsequent_mask


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, vocab_size, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64, d_model=512, d_inner_hid=1024, dropout=0.1, emb_scale=1):
        super(Decoder, self).__init__()
        self.n_max_seq = n_max_seq
        self.output_dim = vocab_size
        self.d_model = d_model
        self.emb_scale = emb_scale
        self.position_enc = PositionalEncoding(dropout, d_model, self.n_max_seq)
        self.tgt_word_emb = nn.Embedding(vocab_size, d_model, Constants.PAD)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, outputs_data, outputs_pos, input_pos, enc_output, return_attns=False):
        dec_input = self.tgt_word_emb(outputs_data)
        dec_input = self.position_enc(dec_input)
        dec_slf_attn_pad_mask = padding_info_mask(outputs_data, outputs_data)
        dec_slf_attn_sub_mask = feature_info_mask(outputs_data)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = padding_info_mask(outputs_data, input_pos)
        dec_slf_attns, dec_enc_attns = [], []
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=dec_slf_attn_mask, dec_enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]
        return dec_output, dec_slf_attns, dec_enc_attns


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.return_attns = config.return_attns
        self.encoder = Encoder(input_size=config.feature_dim, n_max_seq=config.max_inputs_length, n_layers=config.num_enc_layer, n_head=config.n_heads, d_k=config.d_k, d_v=config.d_v, d_model=config.d_model, d_inner_hid=config.d_inner_hid, dropout=config.dropout, emb_scale=config.emb_scale)
        self.decoder = Decoder(vocab_size=config.vocab_size, n_max_seq=config.max_target_length, n_layers=config.num_dec_layer, n_head=config.n_heads, d_k=config.d_k, d_v=config.d_v, d_model=config.d_model, d_inner_hid=config.d_inner_hid, dropout=config.dropout, emb_scale=config.emb_scale)
        self.tgt_word_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, inputs, inputs_pos, targets=None, targets_pos=None):
        enc_output, enc_slf_attn = self.encoder(inputs, inputs_pos, self.return_attns)
        dec_output, dec_slf_attn, dec_enc_attn = self.decoder(targets, targets_pos, inputs_pos, enc_output, self.return_attns)
        seq_logit = self.tgt_word_proj(dec_output)
        return seq_logit, (enc_slf_attn, dec_slf_attn, dec_enc_attn)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'d_k': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_ZhengkunTian_Speech_Tranformer_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

