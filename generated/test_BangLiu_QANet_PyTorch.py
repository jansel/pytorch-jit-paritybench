import sys
_module = sys.modules[__name__]
del sys
QANet_main = _module
SQuAD = _module
data_loader = _module
config = _module
QANet = _module
model = _module
modules = _module
attention = _module
cnn = _module
ema = _module
highway = _module
normalize = _module
position = _module
rnn = _module
treelstm = _module
treelstm_utils = _module
QANet_trainer = _module
trainer = _module
loss = _module
metric = _module
util = _module
dict_utils = _module
file_utils = _module
list_utils = _module
ml_utils = _module
nlp_utils = _module
pd_utils = _module
str_utils = _module
tfidf_utils = _module
vis_utils = _module
visualize = _module

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


import math


import random


import torch


import numpy as np


import torch.nn as nn


import torch.optim as optim


from collections import Counter


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn.functional as F


import copy


from torch import nn


from torch.autograd import Variable


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn import init


from torch.nn import functional


import time


import torch.nn.functional as f


import re


import string


import scipy.sparse as sp


from collections import OrderedDict


class Initialized_Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution uses less parameters
    to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input = torch.randn(32, 300, 20)
        >>> output = m(input)
    """

    def __init__(self, in_ch, out_ch, k, dim=1, bias=False):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        :param bias: default False. Add bias or not
        """
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception('Incorrect dimension!')

    def forward(self, x):
        """
        :Input: (batch_num, in_ch, seq_length)
        :Output: (batch_num, out_ch, seq_length)
        """
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    """
    Applies highway transformation to the incoming data.
    It is like LSTM that uses gates. Highway network is
    helpful to train very deep neural networks.
    y = H(x, W_H) * T(x, W_T) + x * C(x, W_C)
    C = 1 - T
    :Examples:
        >>> m = Highway(2, 300)
        >>> x = torch.randn(32, 20, 300)
        >>> y = m(x)
        >>> print(y.size())
    """

    def __init__(self, layer_num, size):
        """
        :param layer_num: number of highway transform layers
        :param size: size of the last dimension of input
        """
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :Input: (N, *, size) * means any number of additional dimensions.
        :Output: (N, *, size) * means any number of additional dimensions.
        """
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * -1e+30


class SelfAttention(nn.Module):

    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries
        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]
        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [(x if x != None else -1) for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Embedding(nn.Module):

    def __init__(self, wemb_dim, cemb_dim, d_model, dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = nn.Conv2d(cemb_dim, d_model, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=10000.0):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, channels % 2, 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


def PosEncoder(x, min_timescale=1.0, max_timescale=10000.0):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal).transpose(1, 2)


class EncoderBlock(nn.Module):

    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model * 2, 1)
        self.w2 = Initialized_Conv1d(d_model * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANet(nn.Module):

    def __init__(self, word_mat, char_mat, c_max_len, q_max_len, d_model, train_cemb=False, pad=0, dropout=0.1, num_head=1):
        super().__init__()
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_mat)
        self.word_emb = nn.Embedding.from_pretrained(word_mat)
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)
        self.num_head = num_head
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1) for _ in range(7)])
        self.out = Pointer(d_model)
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.ones_like(Cwid) * self.PAD != Cwid).float()
        maskQ = (torch.ones_like(Qwid) * self.PAD != Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        None


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self, dropout=0.0):
        """
        :param dropout: attention dropout rate
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=self.dropout)
        return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    """
    Compute 'Multi-Head Attention'
    When we calculate attentions, usually key and value are the same tensor.
    For self-attention, query, key, value are all the same tensor.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: hidden size
        :param dropout: attention dropout rate
        """
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = ScaledDotProductAttention(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn


class LayerNormalization(nn.Module):
    """Construct a layernorm module."""

    def __init__(self, features, eps=1e-06):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    """
    Add position information to input tensor.
    :Examples:
        >>> m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        >>> input = torch.randn(3, 10, 6)
        >>> output = m(input)
    """

    def __init__(self, d_model, dropout=0, max_len=5000):
        """
        :param d_model: same with input hidden size
        :param dropout: dropout rate
        :param max_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :Input: (batch_num, seq_length, hidden_size)
        :Output: (batch_num, seq_length, hidden_size)
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class RNN(nn.Module):
    """
    General Recurrent Neural Network module.
    Input: tensor of shape (seq_len, batch, input_size)
    Output: tensor of shape (seq_len, batch, hidden_size * num_directions)
    """

    def __init__(self, input_size, hidden_size, output_projection_size=None, num_layers=1, bidirectional=True, cell_type='lstm', dropout=0, pack=False, batch_first=False, init_method='default'):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        if output_projection_size is not None:
            self.output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_projection_size)
        self.pack = pack
        network = self._get_rnn(cell_type)
        self.network = network(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

    def forward(self, input_variable):
        outputs, hidden = self.network(input_variable)
        if self.pack:
            padded_outputs, lengths = pad_packed_sequence(outputs)
            if hasattr(self, 'output_layer'):
                outputs = pack_padded_sequence(self.output_layer(padded_outputs), lengths)
        elif hasattr(self, 'output_layer'):
            outputs = self.output_layer(outputs)
        return outputs, hidden

    def _get_rnn(self, rnn_type):
        rnn_type = rnn_type.lower()
        if rnn_type == 'gru':
            network = torch.nn.GRU
        elif rnn_type == 'lstm':
            network = torch.nn.LSTM
        else:
            raise ValueError('Invalid RNN type %s' % rnn_type)
        return network


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim, out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, left=None, right=None):
        """
        :param left: A (h_l, c_l) tuple, where each value has the size
                     (batch_size, max_length, hidden_dim).
        :param right: A (h_r, c_r) tuple, where each value has the size
                      (batch_size, max_length, hidden_dim).
        :returns: h, c, The hidden and cell state of the composed parent,
                  each of which has the size
                  (batch_size, max_length, hidden_dim).
        """
        hl, cl = left
        hr, cr = right
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = treelstm_utils.apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid() + u.tanh() * i.sigmoid()
        h = o.sigmoid() * c.tanh()
        return h, c


class BinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, use_leaf_rnn, intra_attention, gumbel_temperature, bidirectional):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.gumbel_temperature = gumbel_temperature
        self.bidirectional = bidirectional
        assert not (self.bidirectional and not self.use_leaf_rnn)
        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim, out_features=2 * hidden_dim)
        if self.bidirectional:
            self.treelstm_layer = BinaryTreeLSTMLayer(2 * hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * hidden_dim))
        else:
            self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal_(self.word_linear.weight.data)
            init.constant_(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal_(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2).expand_as(new_h)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = treelstm_utils.dot_nd(query=self.comp_query, candidates=new_h)
        if self.training:
            select_mask = treelstm_utils.st_gumbel_softmax(logits=comp_weights, temperature=self.gumbel_temperature, mask=mask)
        else:
            select_mask = treelstm_utils.greedy_select(logits=comp_weights, mask=mask)
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = select_mask_cumsum.data.new(new_h.size(0), 1).zero_()
        right_mask = torch.cat([right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = select_mask_expand * new_h + left_mask_expand * old_h_left + right_mask_expand * old_h_right
        new_c = select_mask_expand * new_c + left_mask_expand * old_c_left + right_mask_expand * old_c_right
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, length, return_select_masks=True):
        """
        :param input: (Tensor) a batch of text representations, with size
                      (batch_size, max_length, hidden_dim).
        :param length: (Tensor) a vector of length batch_size that records
                       each sample's length in the input batch.
        :param return_select_masks: whether return the inter-media select masks
                                    that record how the tree is composed.
        :returns: h, (batch_size, hidden_dim), the final hidden states.
                  c, (batch_size, hidden_dim), the final cell states.
                  nodes, (batch_size, 2 * max_length - 1, hidden_dim),
                  all the hidden states during the composition.
                  select_masks, record how the trees are composed.
                  boundaries, (batch_size, 2 * max_length - 1, 2)
                  It is calculated from select_masks.
                  It record each node's cover boundary in the original
                  sentence, from bottom to top root node.
        """
        batch_size, max_length, _ = input.size()
        max_depth = input.size(1)
        length_mask = treelstm_utils.sequence_mask(seq_length=length, max_length=max_depth)
        select_masks = []
        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = input.size()
            zero_state = input.data.new(batch_size, self.hidden_dim).zero_()
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(input=input[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)
            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = treelstm_utils.reverse_padded_sequence(inputs=input, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = treelstm_utils.reverse_padded_sequence(inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = treelstm_utils.reverse_padded_sequence(inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = hs, cs
        else:
            state = treelstm_utils.apply_nd(fn=self.word_linear, input=input)
            state = state.chunk(chunks=2, dim=2)
        nodes = []
        nodes.append(state[0])
        for i in range(max_depth - 1):
            h, c = state
            new_state = self.treelstm_layer(left=(h[:, :-1, :], c[:, :-1, :]), right=(h[:, 1:, :], c[:, 1:, :]))
            if i < max_depth - 2:
                new_h, new_c, select_mask, selected_h = self.select_composition(old_state=state, new_state=new_state, mask=length_mask[:, i + 1:])
                new_state = new_h, new_c
                select_masks.append(select_mask)
                nodes.append(selected_h.unsqueeze_(1))
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state, done_mask=done_mask)
            if i >= max_depth - 2:
                nodes.append(state[0])
        h, c = state
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.float()
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = treelstm_utils.masked_softmax(logits=att_weights, mask=att_mask)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            h = (att_weights_expand * nodes).sum(1).unsqueeze(1)
        assert h.size(1) == 1 and c.size(1) == 1
        median_boundaries = self._select_masks_to_boundaries(select_masks)
        leaf_boundaries = torch.rand(batch_size, max_length, 2).long()
        for j in range(max_length):
            leaf_boundaries[:, j, :] = j
        root_boundaries = torch.rand(batch_size, 1, 2).long()
        root_boundaries[:, :, 1] = max_length - 1
        boundaries = torch.cat([leaf_boundaries, median_boundaries, root_boundaries], dim=1)
        if not return_select_masks:
            return h.squeeze(1), c.squeeze(1)
        else:
            return h.squeeze(1), c.squeeze(1), select_masks, nodes, boundaries

    def _select_masks_to_boundaries(self, select_masks):
        """
        Transform select_masks to boundaries.
        :param select_masks: a list of select_mask, shapes are
                             [(batch_size, max_length - 1),
                              (batch_size, max_length - 2),
                              ...,
                              (batch_size, 2)]
        :return: inter-media nodes boundaries,
                 shape is (batch_size, max_length - 2, 2),
                 each element in last dimension is [start, end] index.
        """

        def _merge(node_covers, select_idx):
            """
            node_covers [[s1, e1], [s2, e2], ..., [sn, en]]
            """
            new_node_covers = []
            for i in range(len(node_covers) - 1):
                if i == select_idx:
                    merge_node_boundary = [node_covers[select_idx][0], node_covers[select_idx + 1][1]]
                    new_node_covers.append(merge_node_boundary)
                elif i < select_idx:
                    new_node_covers.append(node_covers[i])
                else:
                    new_node_covers.append(node_covers[i + 1])
            return new_node_covers, merge_node_boundary
        batch_size = select_masks[0].size()[0]
        max_length = select_masks[0].size()[1] + 1
        combine_matrix = torch.rand(batch_size, max_length, 2).long()
        for j in range(max_length):
            combine_matrix[:, j, :] = j
        results = []
        for batch_idx in range(batch_size):
            node_covers = combine_matrix[batch_idx, :, :].numpy().tolist()
            result = []
            for node_idx in range(max_length - 2):
                select = select_masks[node_idx][batch_idx, :]
                select_idx = torch.nonzero(select).data[0][0]
                node_covers, merge_boundary = _merge(node_covers, select_idx)
                result.append(merge_boundary)
            results.append(result)
        results = torch.LongTensor(results)
        return results


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CQAttention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 1]), torch.rand([4, 1, 4])], {}),
     True),
    (DepthwiseSeparableConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'k': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Highway,
     lambda: ([], {'layer_num': 1, 'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Initialized_Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (LayerNormalization,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Pointer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'d_model': 4, 'num_head': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_BangLiu_QANet_PyTorch(_paritybench_base):
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

