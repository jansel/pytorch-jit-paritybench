import sys
_module = sys.modules[__name__]
del sys
const = _module
corpus = _module
data_loader = _module
fine_tuning = _module
fuel = _module
fuel_cnn = _module
model = _module
optimization = _module
pretrain = _module
train = _module
Embedding = _module
Highway = _module
LSTM = _module
pre_data = _module
segmenter = _module
model = _module
optim = _module
train = _module
caption = _module
data_loader = _module
model = _module
optim = _module
rouge = _module
train = _module
corpus = _module
data_loader = _module
model = _module
train = _module
const = _module
data_loader = _module
game = _module
mcts = _module
net = _module
play = _module
train = _module
corpus = _module
data_loader = _module
model = _module
segment = _module
train = _module
corpus = _module
data_loader = _module
model = _module
predict = _module
train = _module
base_layer = _module
model = _module
module_utils = _module
corpus = _module
data_loader = _module
main = _module
model = _module
main = _module
corpus = _module
data_loader = _module
generate = _module
model = _module
train = _module
corpus = _module
data_loader = _module
main = _module
model = _module
corpus = _module
data_loader = _module
model = _module
prepare_data = _module
skeleton2conll = _module
train = _module
utils = _module
data_loader = _module
model = _module
train = _module
corpus = _module
data_loader = _module
example = _module
model = _module
train = _module
img_loader = _module
predict = _module
train = _module
dqn = _module
reinforce = _module
corpus = _module
data_loader = _module
model = _module
train = _module
common = _module
corpus = _module
data_loader = _module
download = _module
model = _module
predict = _module
train = _module
corpus = _module
data_loader = _module
main = _module
model = _module
img_loader = _module
model = _module
train = _module
main = _module
corpus = _module
data_loader = _module
module = _module
predict = _module
train = _module
common = _module
corpus = _module
data_loader = _module
layers = _module
model = _module
predict = _module
train = _module
corpus = _module
data_loader = _module
model = _module
rouge = _module
train = _module
transform = _module
corpus = _module
data_loader = _module
model = _module
train = _module
corpus = _module
data_loader = _module
model = _module
train = _module
corpus = _module
data_loader = _module
layers = _module
model = _module
train = _module
transform = _module
utils = _module
corpus = _module
data_loader = _module
highway = _module
model = _module
optim = _module
train = _module
embedding = _module
encode = _module
data_loader = _module
model = _module
radam = _module
writer = _module
preprocess_training = _module
speaker_embedding = _module
hparams = _module
voice_encoder = _module
train = _module
models = _module
causal_conv = _module
upsample = _module
melgan = _module
parallel_wavegan = _module
pqmf = _module
preprocess = _module
utils = _module
darknet = _module
detect = _module
img_loader = _module
layer = _module
train = _module
utils = _module

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


import random


import numpy as np


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.nn import init


from torch.optim import Optimizer


from torch.utils.data import DataLoader


import torch.nn.init as init


from torch.nn.parallel.data_parallel import DataParallel


from collections import OrderedDict


import torchvision.datasets as td


import torchvision.transforms as transforms


import time


from torch.autograd import Variable


import re


import torchvision


from torchvision import transforms


from torch.nn.parameter import Parameter


from torchvision.models import inception_v3


from torch.optim import Adam


import logging


import copy


from collections import deque


import torch.autograd as autograd


from torch.nn.functional import cosine_similarity


import torch.optim as optim


import collections


from torch.nn.functional import binary_cross_entropy


from collections import defaultdict


from copy import deepcopy


from torchvision import transforms as T


from torchvision.models import resnet50


from collections import namedtuple


from itertools import count


from torch.distributions import Categorical


import pandas as pd


from torchvision.models import vgg19


from itertools import groupby


from torch.utils import data


from torch.optim.optimizer import Optimizer


from typing import Union


from typing import List


from torch import nn


from scipy.signal import kaiser


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, dropout):
        super().__init__()
        self.temper = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn.view(-1, attn.size(2))).view(*attn.size())
        attn = self.dropout(attn)
        return torch.bmm(attn, v)


class MultiHeadAtt(nn.Module):

    def __init__(self, n_head, d_model, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_v = self.d_k = d_k = d_model // n_head
        for name in ['w_qs', 'w_ks', 'w_vs']:
            self.__setattr__(name, nn.Parameter(torch.FloatTensor(n_head, d_model, d_k)))
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.lm = LayerNorm(d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self._init_weight()

    def forward(self, q, k, v, attn_mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q
        bsz, len_q, d_model = q.size()
        len_k, len_v = k.size(1), v.size(1)

        def reshape(x):
            """[bsz, len, d_*] -> [n_head x (bsz*len) x d_*]"""
            return x.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        q_s, k_s, v_s = map(reshape, [q, k, v])
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)
        outputs = self.attention(q_s, k_s, v_s, attn_mask.repeat(n_head, 1, 1))
        outputs = torch.cat(torch.split(outputs, bsz, dim=0), dim=-1).view(-1, n_head * d_v)
        outputs = F.dropout(self.w_o(outputs), p=self.dropout).view(bsz, len_q, -1)
        return self.lm(outputs + residual)

    def _init_weight(self):
        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)
        init.xavier_normal(self.w_o.weight)


class PositionWise(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv1d(d_model, d_ff, 1), nn.ReLU(), nn.Conv1d(d_ff, d_model, 1), nn.Dropout(dropout))
        self.lm = LayerNorm(d_model)

    def forward(self, input):
        residual = input
        out = self.seq(input.transpose(1, 2)).transpose(1, 2)
        return self.lm(residual + out)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        self.mh = MultiHeadAtt(n_head, d_model, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, enc_input, slf_attn_mask):
        enc_output = self.mh(enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.pw(enc_output)
        return enc_output


class GELU(nn.Module):
    """
    different from 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


INIT_RANGE = 0.02


class Pooler(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.linear.weight.data.normal_(std=INIT_RANGE)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x[:, 0])
        return F.tanh(x)


def get_attn_padding_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    bsz, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(const.PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(bsz, len_q, len_k)
    return pad_attn_mask


def position(n_position, d_model):
    position_enc = np.array([[(pos / np.power(10000, 2 * i / d_model)) for i in range(d_model)] for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).float()


class BERT(nn.Module):

    def __init__(self, args):
        super().__init__()
        n_position = args.max_len + 1
        self.enc_ebd = nn.Embedding(args.vsz, args.d_model)
        self.seg_ebd = nn.Embedding(3, args.d_model)
        self.pos_ebd = nn.Embedding(n_position, args.d_model)
        self.pos_ebd.weight.data = position(n_position, args.d_model)
        self.pos_ebd.weight.requires_grad = False
        self.dropout = nn.Dropout(p=args.dropout)
        self.ebd_normal = LayerNorm(args.d_model)
        self.out_normal = LayerNorm(args.d_model)
        self.encodes = nn.ModuleList([EncoderLayer(args.d_model, args.d_ff, args.n_head, args.dropout) for _ in range(args.n_stack_layers)])
        self.pooler = Pooler(args.d_model)
        self.transform = nn.Linear(args.d_model, args.d_model)
        self.gelu = GELU()

    def reset_parameters(self):
        self.enc_ebd.weight.data.normal_(std=INIT_RANGE)
        self.seg_ebd.weight.data.normal_(std=INIT_RANGE)
        self.transform.weight.data.normal_(std=INIT_RANGE)
        self.transform.bias.data.zero_()

    def forward(self, inp, pos, segment_label):
        encode = self.enc_ebd(inp) + self.seg_ebd(segment_label) + self.pos_ebd(pos)
        encode = self.dropout(self.ebd_normal(encode))
        slf_attn_mask = get_attn_padding_mask(inp)
        for layer in self.encodes:
            encode = layer(encode, slf_attn_mask)
        sent_encode = self.pooler(encode)
        word_encode = self.out_normal(self.gelu(self.transform(encode)))
        return word_encode, sent_encode

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def parameters_count(self):
        return sum(x.numel() for x in self.parameters())

    def save_model(self, args, data, path='model.pt'):
        torch.save({'args': args, 'weights': self.state_dict(), 'dict': data['dict'], 'max_len': data['max_len']}, path)

    def load_model(self, weights):
        self.load_state_dict(weights)
        self


class Classifier(nn.Module):

    def __init__(self, lsz, args):
        super().__init__()
        self.bert = BERT(args)
        self.sent_predict = nn.Linear(args.d_model, lsz)
        self.sent_predict.weight.data.normal_(INIT_RANGE)
        self.sent_predict.bias.data.zero_()

    def get_trainable_parameters(self):
        return self.bert.get_trainable_parameters()

    def forward(self, inp, pos, segment_label):
        _, sent_encode = self.bert(inp, pos, segment_label)
        return F.log_softmax(self.sent_predict(sent_encode), dim=-1)

    def load_model(self, path='model.pt'):
        data = torch.load(path)
        self.bert.load_model(data['weights'])


class WordCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        tgt_sum = mask.sum()
        loss = -(tgt_props * mask).sum() / tgt_sum
        _, index = torch.max(props, -1)
        corrects = ((index.data == tgt).float() * mask).sum()
        return loss, corrects, tgt_sum


NOT_USE_WEIGHT_DECAY = ['bias', 'gamma', 'beta']


class Pretraining(nn.Module):

    def __init__(self, lsz, args):
        super().__init__()
        self.bert = BERT(args)
        self.sent_predict = nn.Linear(args.d_model, lsz)
        self.word_predict = nn.Linear(args.d_model, args.vsz)
        self.reset_parameters()

    def reset_parameters(self):
        self.bert.reset_parameters()
        self.sent_predict.weight.data.normal_(INIT_RANGE)
        self.sent_predict.bias.data.zero_()
        self.word_predict.weight = self.bert.enc_ebd.weight
        self.word_predict.bias.data.zero_()

    def get_optimizer_parameters(self, decay):
        return [{'params': [p for n, p in self.named_parameters() if n.split('.')[-1] not in NOT_USE_WEIGHT_DECAY and p.requires_grad], 'weight_decay': decay}, {'params': [p for n, p in self.named_parameters() if n.split('.')[-1] in NOT_USE_WEIGHT_DECAY and p.requires_grad], 'weight_decay': 0.0}]

    def forward(self, inp, pos, segment_label):
        word_encode, sent_encode = self.bert(inp, pos, segment_label)
        sent = F.log_softmax(self.sent_predict(sent_encode), dim=-1)
        word = F.log_softmax(self.word_predict(word_encode), dim=-1)
        return word, sent


class PreEmbedding(nn.Module):

    def __init__(self, pre_w2v, vocab_size, ebd_dim):
        self.lookup_table = nn.Embedding(vocab_size, ebd_dim)
        assert isinstance(pre_w2v, np.ndarray)
        self.lookup_table.weight.data.copy_(torch.from_numpy(pre_w2v))
        self.lookup_table.weight.requires_grad = False

    def forward(self, x):
        return self.lookup_table(x)


class highway_layer(nn.Module):

    def __init__(self, hsz, active):
        super().__init__()
        self.hsz = hsz
        self.active = active
        self.gate = nn.Linear(hsz, hsz)
        self.h = nn.Linear(hsz, hsz)

    def _init_weight(self):
        stdv = 1.0 / math.sqrt(self.hsz)
        self.gate.weight.data.uniform_(-stdv, stdv)
        self.gate.bias.data.fill_(-1)
        if active.__name__ == 'relu':
            init.xavier_normal(self.h.weight)
        else:
            self.h.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        gate = F.sigmoid(self.gate(x))
        return torch.mul(self.active(self.h(x)), gate) + torch.mul(x, 1 - gate)


class Highway(nn.Module):

    def __init__(self, num_layers, hsz, active):
        super().__init__()
        self.layers = nn.ModuleList([highway_layer(hsz, active) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class C_LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.b_ih = None
            self.b_hh = None
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = F.linear(input, self.w_ih, self.b_ih) + F.linear(hx, self.w_hh, self.b_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)
        in_gate, forget_gate, out_gate = map(F.sigmoid, [in_gate, forget_gate, out_gate])
        cell_gate = F.tanh(cell_gate)
        cy = forget_gate * cx + in_gate * cell_gate
        hy = out_gate * F.tanh(cy)
        return hy, cy


class ConvNorm(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(torch.nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Decoder(nn.Module):
    """Decoder module:
    """

    def __init__(self, hparams):
        super().__init__()
        inp = hparams.dim_neck * 2 + hparams.dim_emb + hparams.f0_dim
        self.lstm1 = nn.LSTM(inp, hparams.dim_pre, 1, batch_first=True)
        convolutions = []
        for _ in range(3):
            conv_layer = nn.Sequential(ConvNorm(hparams.dim_pre, hparams.dim_pre, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='relu'), nn.BatchNorm1d(hparams.dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm2 = nn.LSTM(hparams.dim_pre, 1024, 2, batch_first=True)
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        outputs, _ = self.lstm2(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output


class Encoder(nn.Module):
    """Encoder module:
    """

    def __init__(self, hparams):
        super().__init__()
        self.dim_neck = hparams.dim_neck
        self.freq = hparams.freq
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(ConvNorm(80 + hparams.dim_emb if i == 0 else 512, 512, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='relu'), nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(512, hparams.dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        None
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))
        return codes


class Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.enc = Encoder(self.enc_vocab_size, self.max_word_len, self.n_stack_layers, self.d_model, self.d_ff, self.n_head, self.dropout)
        self.dec = Decoder(self.dec_vocab_size, self.max_word_len, self.n_stack_layers, self.d_model, self.d_ff, self.n_head, self.dropout)
        self.linear = nn.Linear(self.d_model, self.dec_vocab_size, bias=False)
        self._init_weight()

    def _init_weight(self):
        if self.share_linear:
            self.linear.weight = self.dec.dec_ebd.weight
        else:
            init.xavier_normal(self.linear.weight)

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())

    def forward(self, src, src_pos, tgt, tgt_pos):
        tgt, tgt_pos = tgt[:, :-1], tgt_pos[:, :-1]
        enc_outputs = self.enc(src, src_pos)
        dec_output = self.dec(enc_outputs, src, tgt, tgt_pos)
        out = self.linear(dec_output)
        return F.log_softmax(out.view(-1, self.dec_vocab_size))


class _Transition(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.AvgPool2d(2, stride=2))
        self.dropout = dropout

    def forward(self, input):
        out = self.layer(input)
        if self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout)
        return out


class _DenseBLayer(nn.Module):

    def __init__(self, in_channels, growth_rate, dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False), nn.BatchNorm2d(4 * growth_rate), nn.ReLU(inplace=True), nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
        self.dropout = dropout

    def forward(self, input):
        out = self.layer(input)
        out = torch.cat([out, input], 1)
        if self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout)
        return out


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, growth_rate, in_channels, dropout):
        super().__init__()
        self.bottleneck = nn.Sequential(OrderedDict([('dbl_{}'.format(l), _DenseBLayer(in_channels + growth_rate * l, growth_rate, dropout)) for l in range(num_layers)]))

    def forward(self, input):
        return self.bottleneck(input)


class DenseNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.init_cnn_layer = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, args.channels, kernel_size=3, padding=1, bias=False)), ('norm0', nn.BatchNorm2d(args.channels)), ('relu0', nn.ReLU(inplace=True))]))
        denseblocks = []
        for l, nums in enumerate(args.layer_nums):
            denseblocks += [('db_{}'.format(l), _DenseBlock(nums, args.growth_rate, args.channels, args.dropout))]
            _in_channels = args.channels + args.growth_rate * nums
            args.channels = _in_channels // 2
            if l != len(args.layer_nums) - 1:
                denseblocks += [('t_{}'.format(l), _Transition(_in_channels, args.channels, args.dropout))]
        denseblocks += [('nb_5', nn.BatchNorm2d(_in_channels))]
        denseblocks += [('relu_5', nn.ReLU(inplace=True))]
        if args.dropout != 0.0:
            denseblocks += [('dropout_5', nn.Dropout(args.dropout))]
        self.denseblocks = nn.Sequential(OrderedDict(denseblocks))
        self.lr = nn.Linear(_in_channels, args.num_class)
        self.lr.bias.data.fill_(0)

    def forward(self, input):
        out = self.init_cnn_layer(input)
        out = self.denseblocks(out)
        out = F.avg_pool2d(out, 8).squeeze()
        return self.lr(out)


class Attention(nn.Module):

    def __init__(self, hsz):
        super().__init__()
        self.hsz = hsz
        self.sigma = nn.Linear(hsz, hsz)
        self.beta = nn.Linear(hsz, hsz, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hsz)
        self.sigma.weight.data.uniform_(-stdv, stdv)
        self.beta.weight.data.uniform_(-stdv, stdv)

    def forward(self, hiddens, hidden):
        hiddens.append(hidden)
        sigma = torch.tanh(self.sigma(hidden))
        _hiddens = torch.stack(hiddens, dim=1)
        _betas = torch.sum(torch.exp(self.beta(_hiddens)), dim=1)
        beta = torch.exp(self.beta(sigma)) / _betas
        return (beta * hidden).unsqueeze(1)


BOS = 2


PAD = 0


class Actor(nn.Module):

    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()
        self.torch = torch.cuda if use_cuda else torch
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.enc = inception_v3(True)
        self.enc_out = nn.Linear(1000, dec_hsz)
        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(dec_hsz + dec_hsz, dec_hsz, rnn_layers, batch_first=True, dropout=dropout)
        self.attn = Attention(dec_hsz)
        self.out = nn.Linear(self.dec_hsz, vocab_size)
        self._reset_parameters()

    def forward(self, hidden, labels=None):
        word = Variable(self.torch.LongTensor([[BOS]] * self.bsz))
        emb_enc = self.lookup_table(word)
        hiddens = [hidden[0].squeeze()]
        attn = torch.transpose(hidden[0], 0, 1)
        outputs, words = [], []
        for i in range(self.max_len):
            _, hidden = self.rnn(torch.cat([emb_enc, attn], -1), hidden)
            h_state = F.dropout(hidden[0], p=self.dropout)
            props = F.log_softmax(self.out(h_state[-1]), dim=-1)
            attn = self.attn(hiddens, h_state[-1])
            if labels is not None:
                emb_enc = self.lookup_table(labels[:, i]).unsqueeze(1)
            else:
                _props = props.data.clone().exp()
                word = Variable(_props.multinomial(1), requires_grad=False)
                words.append(word)
                emb_enc = self.lookup_table(word)
            outputs.append(props.unsqueeze(1))
        if labels is not None:
            return torch.cat(outputs, 1)
        else:
            return torch.cat(outputs, 1), torch.cat(words, 1)

    def encode(self, imgs):
        if self.training:
            enc = self.enc(imgs)[0]
        else:
            enc = self.enc(imgs)
        enc = self.enc_out(enc)
        return enc

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(self.rnn_layers, self.bsz, self.dec_hsz).zero_())
        h = Variable(enc.data.unsqueeze(0).expand(self.rnn_layers, *enc.size()))
        return h.contiguous(), c.contiguous()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.vocab_size)
        self.enc_out.weight.data.uniform_(-stdv, stdv)
        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.out.weight.data.uniform_(-stdv, stdv)
        for p in self.enc.parameters():
            p.requires_grad = False

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())


class Critic(nn.Module):

    def __init__(self, vocab_size, dec_hsz, rnn_layers, bsz, max_len, dropout, use_cuda):
        super().__init__()
        self.use_cuda = use_cuda
        self.dec_hsz = dec_hsz
        self.rnn_layers = rnn_layers
        self.bsz = bsz
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.lookup_table = nn.Embedding(vocab_size, dec_hsz, padding_idx=PAD)
        self.rnn = nn.LSTM(self.dec_hsz, self.dec_hsz, self.rnn_layers, batch_first=True, dropout=dropout)
        self.value = nn.Linear(self.dec_hsz, 1)
        self._reset_parameters()

    def feed_enc(self, enc):
        weight = next(self.parameters()).data
        c = Variable(weight.new(self.rnn_layers, self.bsz, self.dec_hsz).zero_())
        h = Variable(enc.data.unsqueeze(0).expand(self.rnn_layers, *enc.size()))
        return h.contiguous(), c.contiguous()

    def forward(self, inputs, hidden):
        emb_enc = self.lookup_table(inputs.clone()[:, :-1])
        _, out = self.rnn(emb_enc, hidden)
        out = F.dropout(out[0][-1], p=self.dropout)
        return self.value(out).squeeze()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.vocab_size)
        self.lookup_table.weight.data.uniform_(-stdv, stdv)
        self.value.weight.data.uniform_(-stdv, stdv)


START = 1


STOP = 2


def gather_index(encode, k1, k2, n=6):
    x = torch.arange(start=0, end=n / (n - 1.0), step=1.0 / (n - 1), dtype=torch.float)
    if k1.is_cuda:
        x = x
    k1 = x * k1.float()
    k2 = (1 - x) * k2.float()
    index = torch.round(k1 + k2).long()
    return torch.stack([torch.index_select(encode[idx], 0, index[idx]) for idx in range(encode.size(0))], dim=0)


def log_sum_exp(input, keepdim=False):
    assert input.dim() == 2
    max_scores, _ = input.max(dim=-1, keepdim=True)
    output = input - max_scores
    return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))


class CRF(nn.Module):

    def __init__(self, label_size, is_cuda):
        super().__init__()
        self.label_size = label_size
        self.transitions = nn.Parameter(torch.randn(label_size, label_size))
        self._init_weight()
        self.torch = torch.cuda if is_cuda else torch

    def _init_weight(self):
        init.xavier_uniform_(self.transitions)
        self.transitions.data[START, :].fill_(-10000.0)
        self.transitions.data[:, STOP].fill_(-10000.0)

    def _score_sentence(self, input, tags):
        bsz, sent_len, l_size = input.size()
        score = Variable(self.torch.FloatTensor(bsz).fill_(0.0))
        s_score = Variable(self.torch.LongTensor([[START]] * bsz))
        tags = torch.cat([s_score, tags], dim=-1)
        input_t = input.transpose(0, 1)
        for i, words in enumerate(input_t):
            temp = self.transitions.index_select(1, tags[:, i])
            bsz_t = gather_index(temp.transpose(0, 1), tags[:, i + 1])
            w_step_score = gather_index(words, tags[:, i + 1])
            score = score + bsz_t + w_step_score
        temp = self.transitions.index_select(1, tags[:, -1])
        bsz_t = gather_index(temp.transpose(0, 1), Variable(self.torch.LongTensor([STOP] * bsz)))
        return score + bsz_t

    def forward(self, input):
        bsz, sent_len, l_size = input.size()
        init_alphas = self.torch.FloatTensor(bsz, self.label_size).fill_(-10000.0)
        init_alphas[:, START].fill_(0.0)
        forward_var = Variable(init_alphas)
        input_t = input.transpose(0, 1)
        for words in input_t:
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = words[:, next_tag].view(-1, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var, True))
            forward_var = torch.cat(alphas_t, dim=-1)
        forward_var = forward_var + self.transitions[STOP].view(1, -1)
        return log_sum_exp(forward_var)

    def viterbi_decode(self, input):
        backpointers = []
        bsz, sent_len, l_size = input.size()
        init_vvars = self.torch.FloatTensor(bsz, self.label_size).fill_(-10000.0)
        init_vvars[:, START].fill_(0.0)
        forward_var = Variable(init_vvars)
        input_t = input.transpose(0, 1)
        for words in input_t:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.label_size):
                _trans = self.transitions[next_tag].view(1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                best_tag_scores, best_tag_ids = torch.max(next_tag_var, 1, keepdim=True)
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)
            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))
        terminal_var = forward_var + self.transitions[STOP].view(1, -1)
        _, best_tag_ids = torch.max(terminal_var, 1)
        best_path = [best_tag_ids.view(-1, 1)]
        for bptrs_t in reversed(backpointers):
            best_tag_ids = gather_index(bptrs_t, best_tag_ids)
            best_path.append(best_tag_ids.contiguous().view(-1, 1))
        best_path.pop()
        best_path.reverse()
        return torch.cat(best_path, dim=-1)


class BiLSTM(nn.Module):

    def __init__(self, word_size, word_ebd_dim, kernel_num, lstm_hsz, lstm_layers, dropout, batch_size):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hsz = lstm_hsz
        self.batch_size = batch_size
        self.word_ebd = nn.Embedding(word_size, word_ebd_dim)
        self.lstm = nn.LSTM(word_ebd_dim + kernel_num, hidden_size=lstm_hsz // 2, num_layers=lstm_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self._init_weights()

    def _init_weights(self, scope=1.0):
        self.word_ebd.weight.data.uniform_(-scope, scope)

    def forward(self, words, char_feats, hidden=None):
        encode = self.word_ebd(words)
        encode = torch.cat((char_feats, encode), dim=-1)
        output, hidden = self.lstm(encode, hidden)
        return output, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.lstm_layers * 2, self.batch_size, self.lstm_hsz // 2).zero_()), Variable(weight.new(self.lstm_layers * 2, self.batch_size, self.lstm_hsz // 2).zero_())


class CNN(nn.Module):

    def __init__(self, char_size, char_ebd_dim, kernel_num, filter_size, dropout):
        super().__init__()
        self.char_size = char_size
        self.char_ebd_dim = char_ebd_dim
        self.kernel_num = kernel_num
        self.filter_size = filter_size
        self.dropout = dropout
        self.char_ebd = nn.Embedding(self.char_size, self.char_ebd_dim)
        self.char_cnn = nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(self.filter_size, self.char_ebd_dim))
        self._init_weight()

    def _init_weight(self, scope=1.0):
        init.xavier_uniform_(self.char_ebd.weight)

    def forward(self, input):
        bsz, word_len, char_len = input.size()
        encode = input.view(-1, char_len)
        encode = self.char_ebd(encode).unsqueeze(1)
        encode = F.relu(self.char_cnn(encode))
        encode = F.max_pool2d(encode, kernel_size=(encode.size(2), 1))
        encode = F.dropout(encode.squeeze(), p=self.dropout)
        return encode.view(bsz, word_len, -1)


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(nn.Sequential(ConvNorm(80, 512, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(512)))
        for i in range(1, 5 - 1):
            self.convolutions.append(nn.Sequential(ConvNorm(512, 512, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='tanh'), nn.BatchNorm1d(512)))
        self.convolutions.append(nn.Sequential(ConvNorm(512, 80, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='linear'), nn.BatchNorm1d(80)))

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)
        return x


class Quantinizer(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.size = hparams.f0_dim

    def forward(self, x):
        x = (x * self.size * 0.999).long()
        return F.one_hot(x, num_classes=self.size).float()


class Model(nn.Module):
    """Generator network."""

    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet()
        self.quant = Quantinizer(hparams)

    def forward(self, x, c_org, c_trg, f0):
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1), self.quant(f0)), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)


RES_BLOCK_FILLTERS = 128


class ResBlockNet(nn.Module):

    def __init__(self, ind=RES_BLOCK_FILLTERS, block_filters=RES_BLOCK_FILLTERS, kr_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(ind, block_filters, kr_size, stride=stride, padding=padding, bias=bias), nn.BatchNorm2d(block_filters), nn.ReLU(), nn.Conv2d(block_filters, block_filters, kr_size, stride=stride, padding=padding, bias=bias), nn.BatchNorm2d(block_filters))

    def forward(self, x):
        res = x
        out = self.layers(x) + x
        return F.relu(out)


BLOCKS = 10


HISTORY = 3


IND = HISTORY * 2 + 2


class Feature(nn.Module):

    def __init__(self, ind=IND, outd=RES_BLOCK_FILLTERS):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(ind, outd, stride=1, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(outd), nn.ReLU())
        self.encodes = nn.ModuleList([ResBlockNet() for _ in range(BLOCKS)])

    def forward(self, x):
        x = self.layer(x)
        for enc in self.encodes:
            x = enc(x)
        return x


SIZE = 8


OUTD = SIZE ** 2


class Policy(nn.Module):

    def __init__(self, ind=RES_BLOCK_FILLTERS, outd=OUTD, kernels=2):
        super().__init__()
        self.out = outd * kernels
        self.conv = nn.Sequential(nn.Conv2d(ind, kernels, kernel_size=1), nn.BatchNorm2d(kernels), nn.ReLU())
        self.linear = nn.Linear(kernels * outd, outd)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.out)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class Value(nn.Module):

    def __init__(self, ind=RES_BLOCK_FILLTERS, outd=OUTD, hsz=256, kernels=1):
        super().__init__()
        self.outd = outd
        self.conv = nn.Sequential(nn.Conv2d(ind, kernels, kernel_size=1), nn.BatchNorm2d(kernels), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(outd, hsz), nn.ReLU(), nn.Linear(hsz, 1), nn.Tanh())
        self._reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.outd)
        return self.linear(x)

    def _reset_parameters(self):
        for layer in self.modules():
            if type(layer) == nn.Linear:
                layer.weight.data.uniform_(-0.1, 0.1)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.feat = Feature()
        self.value = Value()
        self.policy = Policy()

    def forward(self, x):
        feats = self.feat(x)
        winners = self.value(feats)
        props = self.policy(feats)
        return winners, props

    def save_model(self, path='model.pt'):
        torch.save(self.state_dict(), path)

    def load_model(self, path='model.pt', cuda=True):
        if cuda:
            self.load_state_dict(torch.load(path))
            self
        else:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            self.cpu()


class AlphaEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.v_loss = nn.MSELoss()

    def forward(self, props, v, pi, reward):
        v_loss = self.v_loss(v, reward)
        p_loss = -torch.mean(torch.sum(props * pi, 1))
        return p_loss + v_loss


class BiLSTM_Cut(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bi_lstm = nn.LSTM(self.embed_dim, self.lstm_hsz, num_layers=self.lstm_layers, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.logistic = nn.Linear(2 * self.lstm_hsz, self.tag_size)
        self._init_weights(scope=self.w_init)

    def forward(self, sentences):
        sents_ebd = self.lookup_table(sentences)
        output, _ = self.bi_lstm(sents_ebd)
        output = self.logistic(output).view(-1, self.tag_size)
        return F.log_softmax(output)

    def _init_weights(self, scope=0.25):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        init.xavier_uniform(self.logistic.weight)


class BiLSTM_CRF_Size(nn.Module):

    def __init__(self, args):
        super(BiLSTM_CRF_Size, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.bi_lstm = nn.LSTM(self.embed_dim, self.lstm_hsz, num_layers=self.lstm_layers, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.logistic = nn.Linear(2 * self.lstm_hsz, self.tag_size)
        self._init_weights(scope=self.w_init)

    def forward(self, sentences):
        sents_ebd = self.lookup_table(sentences)
        output, _ = self.bi_lstm(sents_ebd)
        output = self.logistic(output).view(-1, self.tag_size)
        return F.log_softmax(output)

    def _init_weights(self, scope=0.25):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        init.xavier_uniform(self.logistic.weight)


def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    """
    Return: [batch_size, decompse_dim, dim]
    """
    in_tensor = in_tensor.unsqueeze(1)
    decompose_params = decompose_params.unsqueeze(0)
    return torch.mul(in_tensor, decompose_params)


class FullMatchLay(nn.Module):

    def __init__(self, mp_dim, cont_dim):
        super().__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim
        self.register_parameter('weight', nn.Parameter(torch.Tensor(mp_dim, cont_dim)))
        self.weight.data.uniform_(-1.0, 1.0)

    def forward(self, cont_repres, other_cont_first):
        """
        Args:
            cont_repres - [batch_size, this_len, context_lstm_dim]
            other_cont_first - [batch_size, context_lstm_dim]
        Return:
            size - [batch_size, this_len, mp_dim]
        """

        def expand(context, weight):
            """
            Args:
                [batch_size, this_len, context_lstm_dim]
                [mp_dim, context_lstm_dim]
            Return:
                [batch_size, this_len, mp_dim, context_lstm_dim]
            """
            weight = weight.unsqueeze(0)
            weight = weight.unsqueeze(0)
            context = context.unsqueeze(2)
            return torch.mul(context, weight)
        cont_repres = expand(cont_repres, self.weight)
        other_cont_first = multi_perspective_expand_for_2D(other_cont_first, self.weight)
        other_cont_first = other_cont_first.unsqueeze(1)
        return cosine_similarity(cont_repres, other_cont_first, cont_repres.dim() - 1)


class MaxpoolMatchLay(nn.Module):

    def __init__(self, mp_dim, cont_dim):
        super().__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim
        self.register_parameter('weight', nn.Parameter(torch.Tensor(mp_dim, cont_dim)))
        self.weight.data.uniform_(-1.0, 1.0)

    def forward(self, cont_repres, other_cont_repres):
        """
        Args:
            cont_repres - [batch_size, this_len, context_lstm_dim]
            other_cont_repres - [batch_size, other_len, context_lstm_dim]
        Return:
            size - [bsz, this_len, mp_dim*2]
        """
        bsz = cont_repres.size(0)
        this_len = cont_repres.size(1)
        other_len = other_cont_repres.size(1)
        cont_repres = cont_repres.view(-1, self.cont_dim)
        other_cont_repres = other_cont_repres.view(-1, self.cont_dim)
        cont_repres = multi_perspective_expand_for_2D(cont_repres, self.weight)
        other_cont_repres = multi_perspective_expand_for_2D(other_cont_repres, self.weight)
        cont_repres = cont_repres.view(bsz, this_len, self.mp_dim, self.cont_dim)
        other_cont_repres = other_cont_repres.view(bsz, other_len, self.mp_dim, self.cont_dim)
        cont_repres = cont_repres.unsqueeze(2)
        other_cont_repres = other_cont_repres.unsqueeze(1)
        simi = cosine_similarity(cont_repres, other_cont_repres, cont_repres.dim() - 1)
        t_max, _ = simi.max(2)
        t_mean = simi.mean(2)
        return torch.cat((t_max, t_mean), 2)


class AtteMatchLay(nn.Module):

    def __init__(self, mp_dim, cont_dim):
        super(AtteMatchLay, self).__init__()
        self.cont_dim = cont_dim
        self.mp_dim = mp_dim
        self.register_parameter('weight', nn.Parameter(torch.Tensor(mp_dim, cont_dim)))
        self.weight.data.uniform_(-1.0, 1.0)

    def forward(self, repres, max_att):
        """
        Args:
            repres - [bsz, a_len|q_len, cont_dim]
            max_att - [bsz, q_len|a_len, cont_dim]
        Return:
            size - [bsz, sentence_len, mp_dim]
        """
        bsz = repres.size(0)
        sent_len = repres.size(1)
        repres = repres.view(-1, self.cont_dim)
        max_att = max_att.view(-1, self.cont_dim)
        repres = multi_perspective_expand_for_2D(repres, self.weight)
        max_att = multi_perspective_expand_for_2D(max_att, self.weight)
        temp = cosine_similarity(repres, max_att, repres.dim() - 1)
        return temp.view(bsz, sent_len, self.mp_dim)


PF_POS = 1


def cosine_cont(repr_context, relevancy, norm=False):
    """
    cosine siminlarity betwen context and relevancy
    Args:
        repr_context - [batch_size, other_len, context_lstm_dim]
        relevancy - [batch_size, this_len, other_len]
    Return:
        size - [batch_size, this_len, context_lstm_dim]
    """
    dim = repr_context.dim()
    temp_relevancy = relevancy.unsqueeze(dim)
    buff = repr_context.unsqueeze(1)
    buff = torch.mul(buff, temp_relevancy)
    buff = buff.sum(2)
    if norm:
        relevancy = relevancy.sum(dim - 1).clamp(min=1e-06)
        relevancy = relevancy.unsqueeze(2)
        buff = buff.div(relevancy)
    return buff


eps = 1e-12


def max_repres(repre_cos):
    """
    Args:
        repre_cos - (q_repres, cos_simi_q)|(a_repres, cos_simi)
        Size: ([bsz, q_len, context_dim], [bsz, a_len, question_len])| ...
    Return:
        size - [bsz, a_len, context_dim] if question else [bsz, q_len, context_dim]
    """

    def tf_gather(input, index):
        """
        The same as tensorflow gather sometimes...
        Args:
            - input: dim - 3
            - index: dim - 2
        Return: [input.size(0), index.size(1), input.size(2)]
        """
        bsz = input.size(0)
        sent_size = input.size(1)
        dim_size = input.size(2)
        for n, i in enumerate(index):
            index.data[n] = i.data.add(n * sent_size)
        input = input.view(-1, dim_size)
        index = index.view(-1)
        temp = input.index_select(0, index)
        return temp.view(bsz, -1, dim_size)
    repres, cos_simi = repre_cos
    index = torch.max(cos_simi, 2)[1]
    return tf_gather(repres, index)


class biMPModule(nn.Module):
    """
    Word Representation Layer
        - Corpus Embedding(Word Embedding)
        - Word Embedding(Character Embedding)
    from biMPM TensorFlow, there is a layer called Highway which is a f**king lstmcell implement, do not know y?
    Context Representation Layer
    Matching Layer
    Aggregation Layer
    Prediction Layer
    """

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.context_dim = self.corpus_emb_dim + self.word_lstm_dim
        self.corpus_emb = nn.Embedding(self.corpora_len, self.corpus_emb_dim)
        self._init_corpus_embedding()
        self.word_emb = nn.Embedding(self.words_len, self.word_emb_dim)
        self.word_lstm = nn.LSTM(self.word_emb_dim, self.word_lstm_dim, num_layers=self.word_layer_num, dropout=self.dropout, batch_first=True)
        self.context_lstm = nn.LSTM(self.context_dim, self.context_lstm_dim, num_layers=self.context_layer_num, dropout=self.dropout, batch_first=True, bidirectional=True)
        self.pre_q_attmath_layer = AtteMatchLay(self.mp_dim, self.context_dim)
        self.pre_a_attmath_layer = AtteMatchLay(self.mp_dim, self.context_dim)
        self.f_full_layer = FullMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_full_layer = FullMatchLay(self.mp_dim, self.context_lstm_dim)
        self.f_max_layer = MaxpoolMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_max_layer = MaxpoolMatchLay(self.mp_dim, self.context_lstm_dim)
        self.f_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)
        self.f_max_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)
        self.b_max_att_layer = AtteMatchLay(self.mp_dim, self.context_lstm_dim)
        self.aggre_lstm = nn.LSTM(11 * self.mp_dim + 6, self.aggregation_lstm_dim, num_layers=self.aggregation_layer_num, dropout=self.dropout, batch_first=True, bidirectional=True)
        self.l1 = nn.Linear(self.aggregation_lstm_dim * 4, self.aggregation_lstm_dim * 2)
        self.l2 = nn.Linear(self.aggregation_lstm_dim * 2, self.num_class)
        self._init_weights_and_bias()

    def forward(self, q_corpora, q_words, a_corpora, a_words):
        """
        Module main forward
        """
        self.q_mask = q_corpora.ge(PF_POS)
        self.a_mask = a_corpora.ge(PF_POS)
        self.q_repres = self._word_repre_layer((q_corpora, q_words))
        self.a_repres = self._word_repre_layer((a_corpora, a_words))
        iqr_temp = self.q_repres.unsqueeze(1)
        ipr_temp = self.a_repres.unsqueeze(2)
        simi = F.cosine_similarity(iqr_temp, ipr_temp, dim=3)
        simi_mask = self._cosine_similarity_mask(simi)
        q_aware_reps, a_aware_reps = self._bilateral_match(simi_mask)
        q_aware_reps = F.dropout(q_aware_reps, p=self.dropout)
        a_aware_reps = F.dropout(a_aware_reps, p=self.dropout)
        aggre = self._aggre(q_aware_reps, a_aware_reps)
        predict = F.tanh(self.l1(aggre))
        predict = F.dropout(predict, p=self.dropout)
        return F.softmax(self.l2(predict))

    def _word_repre_layer(self, input):
        """
        args:
            - input: (q_sentence, q_words)|(a_sentence, a_words)
              q_sentence - [batch_size, sent_length]
              q_words - [batch_size, sent_length, words_len]
        return:
            - output: [batch_size, sent_length, context_dim]
        """
        sentence, words = input
        s_encode = self.corpus_emb(sentence)
        w_encode = self._word_repre_forward(words)
        w_encode = F.dropout(w_encode, p=self.dropout, training=True, inplace=False)
        out = torch.cat((s_encode, w_encode), 2)
        return out

    def _word_repre_forward(self, input):
        """
        args:
            - input: q_words|a_words size: [batch_size, sent_length, words_len]
                ps: q_words|a_words is matrix: corpus * words
        return:
            - output: [batch_size, sent_length, word_lstm_dim]
        """
        bsz = input.size(0)
        sent_length = input.size(1)
        words_len = input.size(2)
        input = input.view(-1, words_len)
        encode = self.word_emb(input)
        _, hidden = self.word_lstm(encode)
        output = hidden[0].view(bsz, sent_length, self.word_lstm_dim)
        return output

    def _cosine_similarity_mask(self, simi):
        simi = torch.mul(simi, self.q_mask.unsqueeze(1).float()).clamp(min=eps)
        simi = torch.mul(simi, self.a_mask.unsqueeze(2).float()).clamp(min=eps)
        return simi

    def _bilateral_match(self, cos_simi):
        """
        Args:
            cos_simi: [bsz, a_len, q_len]
        Return:
            q_aware_reps: [bsz, q_len, mp_dim*11+6]
            a_aware_reps: [bsz, a_len, mp_dim*11+6]
        """
        cos_simi_q = cos_simi.permute(0, 2, 1)
        q_aware_reps = [torch.max(cos_simi, 2, keepdim=True)[0], torch.mean(cos_simi, 2, keepdim=True)]
        a_aware_reps = [torch.max(cos_simi_q, 2, keepdim=True)[0], torch.mean(cos_simi_q, 2, keepdim=True)]
        q_max_att = max_repres((self.q_repres, cos_simi))
        q_max_att_rep = self.pre_q_attmath_layer(self.a_repres, q_max_att)
        q_aware_reps.append(q_max_att_rep)
        a_max_att = max_repres((self.a_repres, cos_simi_q))
        a_max_att_rep = self.pre_a_attmath_layer(self.q_repres, a_max_att)
        a_aware_reps.append(a_max_att_rep)
        q_repr_context_f, q_repr_context_b = self._context_repre_forward(self.q_repres)
        a_repr_context_f, a_repr_context_b = self._context_repre_forward(self.a_repres)
        left_match = self._all_match_layer(a_repr_context_f, a_repr_context_b, self.a_mask, q_repr_context_f, q_repr_context_b, self.q_mask)
        right_match = self._all_match_layer(q_repr_context_f, q_repr_context_b, self.q_mask, a_repr_context_f, a_repr_context_b, self.a_mask)
        q_aware_reps.extend(left_match)
        a_aware_reps.extend(right_match)
        q_aware_reps = torch.cat(q_aware_reps, dim=2)
        a_aware_reps = torch.cat(a_aware_reps, dim=2)
        return q_aware_reps, a_aware_reps

    def _context_repre_forward(self, input):
        """
        Args:
            - input: [bsz, sent_length, context_dim]]
        Return:
            - output: size - ([bsz, sent_length, context_lstm_dim], [bsz, sent_length, context_lstm_dim])
        """
        output, _ = self.context_lstm(input)
        return output.split(self.context_lstm_dim, 2)

    def _aggre(self, q_aware_reps, a_aware_reps):
        """
        Aggregation Layer handle
        Args:
            q_aware_reps - [batch_size, question_len, 11*mp_dim+6]
            a_aware_reps - [batch_size, answer_len, 11*mp_dim+6]
        Return:
            size - [batch_size, aggregation_lstm_dim*4]
        """
        _aggres = []
        _, (q_hidden, _) = self.aggre_lstm(q_aware_reps)
        _, (a_hidden, _) = self.aggre_lstm(a_aware_reps)
        _aggres.append(q_hidden[-2])
        _aggres.append(q_hidden[-1])
        _aggres.append(a_hidden[-2])
        _aggres.append(a_hidden[-1])
        return torch.cat(_aggres, dim=1)

    def _all_match_layer(self, repr_context_f, repr_context_b, mask, other_repr_context_f, other_repr_context_b, other_mask):
        """
        Args:
            repr_context_f, repr_context_b|other_repr_context_f, other_repr_context_b - size: [bsz, this_len, context_lstm_dim], [bsz, other_len, context_lstm_dim]
            mask|other_mask - size: [bsz, this_len]|[bsz, other_len]
        Return:
            List - size: [bsz, sentence_len, mp_dim] * 10*mp_dim+4
        """
        repr_context_f = repr_context_f.contiguous()
        repr_context_b = repr_context_b.contiguous()
        other_repr_context_f = other_repr_context_f.contiguous()
        other_repr_context_b = other_repr_context_b.contiguous()
        all_aware_repres = []
        this_cont_dim = repr_context_f.dim()
        repr_context_f = torch.mul(repr_context_f, mask.unsqueeze(this_cont_dim - 1).float()).clamp(min=eps)
        repr_context_b = torch.mul(repr_context_b, mask.unsqueeze(this_cont_dim - 1).float()).clamp(min=eps)
        other_repr_context_f = torch.mul(other_repr_context_f, other_mask.unsqueeze(this_cont_dim - 1).float()).clamp(min=eps)
        other_repr_context_b = torch.mul(other_repr_context_b, other_mask.unsqueeze(this_cont_dim - 1).float()).clamp(min=eps)
        f_relevancy = F.cosine_similarity(other_repr_context_f.unsqueeze(1), repr_context_f.unsqueeze(2), dim=this_cont_dim)
        f_relevancy = torch.mul(f_relevancy, mask.unsqueeze(this_cont_dim - 1).float()).clamp(min=eps)
        f_relevancy = torch.mul(f_relevancy, other_mask.unsqueeze(this_cont_dim - 2).float()).clamp(min=eps)
        b_relevancy = F.cosine_similarity(other_repr_context_b.unsqueeze(1), repr_context_b.unsqueeze(2), dim=this_cont_dim)
        b_relevancy = torch.mul(b_relevancy, mask.unsqueeze(this_cont_dim - 1).float()).clamp(min=eps)
        b_relevancy = torch.mul(b_relevancy, other_mask.unsqueeze(this_cont_dim - 2).float()).clamp(min=eps)
        other_context_f_first = other_repr_context_f[:, -1, :]
        other_context_b_first = other_repr_context_b[:, 0, :]
        f_full_match = self.f_full_layer(repr_context_f, other_context_f_first)
        b_full_match = self.b_full_layer(repr_context_b, other_context_b_first)
        all_aware_repres.append(f_full_match)
        all_aware_repres.append(b_full_match)
        f_max_match = self.f_max_layer(repr_context_f, other_repr_context_f)
        b_max_match = self.b_max_layer(repr_context_b, other_repr_context_b)
        all_aware_repres.append(f_max_match)
        all_aware_repres.append(b_max_match)
        f_att_cont = cosine_cont(other_repr_context_f, f_relevancy)
        f_att_repre = self.f_att_layer(repr_context_f, f_att_cont)
        b_att_cont = cosine_cont(other_repr_context_b, b_relevancy)
        b_att_repre = self.b_att_layer(repr_context_b, b_att_cont)
        all_aware_repres.append(f_att_repre)
        all_aware_repres.append(b_att_repre)
        f_max_att = max_repres((other_repr_context_f, f_relevancy))
        f_max_att_repres = self.f_max_att_layer(repr_context_f, f_max_att)
        b_max_att = max_repres((other_repr_context_b, b_relevancy))
        b_max_att_repres = self.b_max_att_layer(repr_context_b, b_max_att)
        all_aware_repres.append(f_max_att_repres)
        all_aware_repres.append(b_max_att_repres)
        all_aware_repres.append(f_relevancy.max(2, keepdim=True)[0])
        all_aware_repres.append(f_relevancy.mean(2, keepdim=True))
        all_aware_repres.append(b_relevancy.max(2, keepdim=True)[0])
        all_aware_repres.append(b_relevancy.mean(2, keepdim=True))
        return all_aware_repres

    def _init_weights_and_bias(self, scope=1.0):
        """
        initialise weight and bias
        """
        self.word_emb.weight.data.uniform_(-scope, scope)
        self.l1.weight.data.uniform_(-scope, scope)
        self.l1.bias.data.fill_(0)
        self.l2.weight.data.uniform_(-scope, scope)
        self.l2.bias.data.fill_(0)

    def _init_corpus_embedding(self):
        """
        corpus embedding is a fixed vector for each individual corpus,
        which is pre-trained with word2vec
        """
        self.corpus_emb.weight.data.copy_(torch.from_numpy(self.corpora_emb))
        self.corpus_emb.weight.requires_grad = False


class BiRNN(nn.Module):

    def __init__(self, vsz, embed_dim, dropout, hsz, layers):
        super().__init__()
        self.lookup_table = nn.Embedding(vsz, embed_dim, padding_idx=const.PAD)
        self.lstm = nn.LSTM(embed_dim, hsz, layers, dropout=dropout, batch_first=True, bidirectional=True)
        scope = 1.0 / math.sqrt(vsz)
        self.lookup_table.weight.data.uniform_(-scope, scope)

    def forward(self, input):
        encode = self.lookup_table(input)
        lstm_out, _ = self.lstm(encode)
        feats = lstm_out.mean(1)
        return lstm_out, feats


class ConvUnit(nn.Module):

    def __init__(self):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=5, stride=1)

    def forward(self, x):
        return self.conv(x)


def squash(input):
    mag_sq = torch.sum(input ** 2, dim=2, keepdim=True)
    mag = torch.sqrt(mag_sq)
    out = mag_sq / (1.0 + mag_sq) * (input / mag)
    return out


class PrimaryCap(nn.Module):

    def __init__(self, num_primary_units):
        super().__init__()
        self.num_primary_units = num_primary_units
        self.convUnits = nn.ModuleList([ConvUnit() for _ in range(num_primary_units)])

    def forward(self, input):
        bsz = input.size(0)
        units = [unit(input) for unit in self.convUnits]
        units = torch.stack(units, dim=1)
        units = units.view(bsz, self.num_primary_units, -1)
        return squash(units)


class DigitCap(nn.Module):

    def __init__(self, use_cuda, num_primary_units, labels, output_unit_size, primary_unit_size, iterations):
        super().__init__()
        self.labels = labels
        self.use_cuda = use_cuda
        self.primary_unit_size = primary_unit_size
        self.iterations = iterations
        self.W = nn.Parameter(torch.randn(1, primary_unit_size, labels, output_unit_size, num_primary_units))

    def forward(self, input):
        bsz = input.size(0)
        input_t = input.transpose(1, 2)
        u = torch.stack([input_t] * self.labels, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * bsz, dim=0)
        u_hat = torch.matmul(W, u)
        b_ij = Variable(torch.zeros(1, self.primary_unit_size, self.labels, 1))
        if self.use_cuda:
            b_ij = b_ij
        for _ in range(self.iterations):
            c_ij = F.softmax(b_ij, dim=-1)
            c_ij = torch.cat([c_ij] * bsz, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)
            v_j1 = torch.cat([v_j] * self.primary_unit_size, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1
        return v_j.squeeze()


def to_one_hot(x, length, use_cuda, is_zero=True):
    bsz, x_list = x.size(0), x.data.tolist()
    x_one_hot = torch.zeros(bsz, length)
    if is_zero:
        for i in range(bsz):
            x_one_hot[i, x_list[i]] = 1.0
    else:
        x_one_hot = x_one_hot + 0.1
        for i in range(bsz):
            x_one_hot[i, x_list[i]] = -1.0
    x_one_hot = Variable(x_one_hot)
    if use_cuda:
        x_one_hot = x_one_hot
    return x_one_hot


class Capsule(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.rnn = BiRNN(self.vsz, self.embed_dim, self.dropout, self.hsz, self.layers)
        self.fc = nn.Linear(self.hsz * 2, self.max_len)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 256, kernel_size=7), nn.ReLU(inplace=True))
        self.pCap = PrimaryCap(self.num_primary_units)
        self.dCap = DigitCap(self.use_cuda, self.num_primary_units, self.labels, self.output_unit_size, self.primary_unit_size, self.iterations)
        self.recon = nn.Sequential(nn.Linear(self.output_unit_size, self.hsz), nn.ReLU(inplace=True), nn.Linear(self.hsz, self.hsz * 2))
        self._reset_parameters()

    def _reset_parameters(self):
        scope = 1.0 / math.sqrt(self.vsz)
        for m in self.modules():
            if type(m) == nn.Linear:
                m.weight.data.uniform_(-scope, scope)

    def forward(self, input):
        lstm_out, lstm_feats = self.rnn(input)
        in_capsule = self.fc(lstm_out)
        in_capsule = in_capsule.unsqueeze(1)
        conv1_out = self.conv1(in_capsule)
        pCap_out = self.pCap(conv1_out)
        dCap_out = self.dCap(pCap_out)
        return dCap_out, lstm_feats

    def loss(self, props, target, lstm_feats):
        zero_t = to_one_hot(target, self.labels, self.use_cuda)
        unzero_t = to_one_hot(target, self.labels, self.use_cuda, False)
        return self.margin_loss(props, zero_t) + self.reconstruction_loss(lstm_feats, props, unzero_t) * 0.05

    def margin_loss(self, props, target):
        bsz = props.size(0)
        v_abs = torch.sqrt((props ** 2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        if self.use_cuda:
            zero = zero
        m_plus, m_minus = 0.9, 0.1
        max_pos = torch.max(zero, m_plus - v_abs).view(bsz, -1) ** 2
        max_neg = torch.max(zero, v_abs - m_minus).view(bsz, -1) ** 2
        loss = target * max_pos + 0.5 * (1.0 - target) * max_neg
        return loss.mean()

    def reconstruction_loss(self, lstm_feats, props, target):
        bsz, target = props.size(0), target.unsqueeze(2)
        v = torch.sqrt((props ** 2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1))
        if self.use_cuda:
            zero = zero
        r = self.recon(props)
        lstm_feats = lstm_feats.unsqueeze(1)
        _temp = (r * target * v * lstm_feats).sum(2, keepdim=True)
        loss = torch.max(zero, 1.0 + _temp)
        return loss.mean()


class CBOW(nn.Module):

    def __init__(self, vocab_size, ebd_size, cont_size):
        super(CBOW, self).__init__()
        self.ebd = nn.Embedding(vocab_size, ebd_size)
        self.lr1 = nn.Linear(ebd_size * cont_size * 2, 128)
        self.lr2 = nn.Linear(128, vocab_size)
        self._init_weight()

    def forward(self, inputs):
        out = self.ebd(inputs).view(1, -1)
        out = F.relu(self.lr1(out))
        out = self.lr2(out)
        out = F.log_softmax(out)
        return out

    def _init_weight(self, scope=0.1):
        self.ebd.weight.data.uniform_(-scope, scope)
        self.lr1.weight.data.uniform_(0, scope)
        self.lr1.bias.data.fill_(0)
        self.lr2.weight.data.uniform_(0, scope)
        self.lr2.bias.data.fill_(0)


class CNN_Text(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = 'encoder_%d' % i
            self.__setattr__(enc_attr_name, nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(filter_size, self.embed_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))
        self.logistic = nn.Linear(len(self.filter_sizes) * self.kernel_num, self.label_size)
        self.dropout = nn.Dropout(self.dropout)
        self._init_weight()

    def forward(self, x):
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        x = self.lookup_table(x)
        x = x.unsqueeze(c_idx)
        enc_outs = []
        for encoder in self.encoders:
            enc_ = F.relu(encoder(x))
            k_h = enc_.size()[h_idx]
            enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        return F.log_softmax(self.logistic(encoding))

    def _init_weight(self, scope=0.1):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)


class Score(nn.Module):

    def __init__(self, in_dim, hidden_dim=150):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(in_dim, 2 * hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.score(x)


class Distance(nn.Module):
    bins = [1, 2, 3, 4, 8, 16, 32, 64]

    def __init__(self, distance_dim=20):
        super().__init__()
        self.dim = distance_dim
        self.embeds = nn.Sequential(nn.Embedding(len(self.bins) + 1, distance_dim), nn.Dropout(0.2))

    def forward(self, lengths):
        return self.embeds(lengths)


class RnnEncoder(nn.Module):

    def __init__(self, d_model, embedding_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_size=d_model, batch_first=True, bidirectional=True)
        self.ln = LayerNorm(d_model * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        encode, _ = self.rnn(x)
        encode = self.ln(encode)
        return self.dropout(encode)[:, -1, :]


class MentionPairScore(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.position_embedding = nn.Embedding(args.max_len + 1, args.pos_dim)
        self.word_embedding = nn.Embedding(args.word_ebd_weight.shape[0], args.word_ebd_weight.shape[1])
        self.word_embedding.weight.data.copy_(torch.from_numpy(args.word_ebd_weight))
        self.embedding_transform = nn.Linear(args.pos_dim + args.word_ebd_weight.shape[1], args.d_model)
        self.transform_activate = GELU()
        self.rnn_rncoder = RnnEncoder(args.rnn_hidden_size, args.word_ebd_weight.shape[1], args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.head_att = MultiHeadAtt(args.n_head, args.d_model, args.dropout)
        self.distance_embedding = Distance()
        score_in_dim = 4 * args.d_model + 4 * args.rnn_hidden_size + args.pos_dim
        self.score = Score(score_in_dim)
        self._reset_parameters()

    def forward(self, doc, word2idx):
        doc_encoding = self.doc_encode(doc)
        mention_rnn_encode, coref_rnn_encode, distances_embedding_encode, corefs_idxs, mention_idxs, labels = self.mention_encode(doc, word2idx)
        doc_features = torch.stack([torch.cat((doc_encoding[mention_start], doc_encoding[mention_end], doc_encoding[coref_start], doc_encoding[coref_end])) for (mention_start, mention_end), (coref_start, coref_end) in zip(mention_idxs, corefs_idxs)], dim=0)
        mention_features = torch.cat((doc_features, mention_rnn_encode, coref_rnn_encode, distances_embedding_encode), dim=1)
        scores = self.score(mention_features).squeeze()
        return torch.sigmoid(scores), labels

    def doc_encode(self, doc):
        doc_embedding_encode = self.word_embedding(doc.token_tensors)
        doc_postion_encode = self.position_embedding(doc.pos2tensor(self.args.use_cuda))
        doc_encode = self.dropout(torch.cat((doc_embedding_encode, doc_postion_encode), dim=-1))
        transform_encode = self.embedding_transform(doc_encode)
        transform_encode = self.transform_activate(transform_encode).unsqueeze(0)
        return self.head_att(transform_encode, transform_encode, transform_encode).squeeze()

    def mention_encode(self, doc, word2idx):
        corefs_idxs, mention_idxs, mention_spans, labels, distances, corefs = doc.sample(self.args.use_cuda, self.args.batch_size)
        distances_embedding_encode = self.distance_embedding(distances)
        mention_embedding_encode = self.word_embedding(mention_spans)
        coref_embedding_encode = self.word_embedding(corefs)
        distances_embedding_encode = self.dropout(distances_embedding_encode)
        mention_embedding_encode = self.dropout(mention_embedding_encode)
        coref_embedding_encode = self.dropout(coref_embedding_encode)
        mention_rnn_encode = self.rnn_rncoder(mention_embedding_encode)
        coref_rnn_encode = self.rnn_rncoder(coref_embedding_encode)
        return mention_rnn_encode, coref_rnn_encode, distances_embedding_encode, corefs_idxs, mention_idxs, labels

    def mention_predict(self, tokens, positions, mention, coref_idx, mention_idx, distance, coref):
        doc_embedding_encode = self.word_embedding(tokens)
        doc_postion_encode = self.position_embedding(positions)
        doc_encode = self.dropout(torch.cat((doc_embedding_encode, doc_postion_encode), dim=-1))
        transform_encode = self.embedding_transform(doc_encode)
        transform_encode = self.transform_activate(transform_encode).unsqueeze(0)
        doc_encoding = self.head_att(transform_encode, transform_encode, transform_encode).squeeze()
        distance_embedding_encode = self.distance_embedding(distance)
        mention_embedding_encode = self.word_embedding(mention)
        coref_embedding_encode = self.word_embedding(coref)
        distance_embedding_encode = self.dropout(distance_embedding_encode)
        mention_embedding_encode = self.dropout(mention_embedding_encode)
        coref_embedding_encode = self.dropout(coref_embedding_encode)
        mention_rnn_encode = self.rnn_rncoder(mention_embedding_encode)
        coref_rnn_encode = self.rnn_rncoder(coref_embedding_encode)
        doc_feature = torch.cat((doc_encoding[mention_idx[0]], doc_encoding[mention_idx[1]], doc_encoding[coref_idx[0]], doc_encoding[coref_idx[1]]))
        mention_feature = torch.cat((doc_feature.unsqueeze(0), mention_rnn_encode, coref_rnn_encode, distance_embedding_encode.squeeze(0)), dim=1)
        score = self.score(mention_feature).squeeze()
        return torch.sigmoid(score)

    def _reset_parameters(self):
        self.position_embedding.weight.data.uniform_(-0.1, 0.1)
        for layer in self.modules():
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(std=const.INIT_RANGE)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, cuda):
        if cuda:
            self.load_state_dict(torch.load(path))
            self
        else:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            self.cpu()


def conv_size(args):
    try:
        reduce
    except NameError:
        from functools import reduce
    out_size, num_filter = args
    _size = [out_size] + [2] * num_filter
    return reduce(lambda x, y: x // y, _size)


class Generator(nn.Module):

    def __init__(self, out_h, out_w, channel_dims, z_dim=100):
        super().__init__()
        assert len(channel_dims) == 4, 'length of channel dims should be 4'
        conv1_dim, conv2_dim, conv3_dim, conv4_dim = channel_dims
        conv1_h, conv2_h, conv3_h, conv4_h = map(conv_size, [(out_h, step) for step in [4, 3, 2, 1]])
        conv1_w, conv2_w, conv3_w, conv4_w = map(conv_size, [(out_w, step) for step in [4, 3, 2, 1]])
        self.fc = nn.Linear(z_dim, conv1_dim * conv1_h * conv1_w)
        self.deconvs = nn.Sequential(nn.BatchNorm2d(conv1_dim), nn.ReLU(), nn.ConvTranspose2d(conv1_dim, conv2_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(conv2_dim), nn.ReLU(), nn.ConvTranspose2d(conv2_dim, conv3_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(conv3_dim), nn.ReLU(), nn.ConvTranspose2d(conv3_dim, conv4_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(conv4_dim), nn.ReLU(), nn.ConvTranspose2d(conv4_dim, 3, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())
        self.conv1_size = conv1_dim, conv1_h, conv1_w
        self._init_weight()

    def _init_weight(self):
        self.fc.weight.data.normal_(0.0, 0.02)
        for layer in self.deconvs:
            if isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data.normal_(0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, *self.conv1_size)
        return self.deconvs(out)


class Discriminator(nn.Module):

    def __init__(self, out_h, out_w, channel_dims, relu_leak):
        super().__init__()
        assert len(channel_dims) == 4, 'length of channel dims should be 4'
        conv4_dim, conv3_dim, conv2_dim, conv1_dim = channel_dims
        conv4_h, conv3_h, conv2_h, conv1_h = map(conv_size, [(out_h, step) for step in [4, 3, 2, 1]])
        conv4_w, conv3_w, conv2_w, conv1_w = map(conv_size, [(out_w, step) for step in [4, 3, 2, 1]])
        self.convs = nn.Sequential(nn.Conv2d(3, conv1_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(relu_leak), nn.Conv2d(conv1_dim, conv2_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(conv2_dim), nn.LeakyReLU(relu_leak), nn.Conv2d(conv2_dim, conv3_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(conv3_dim), nn.LeakyReLU(relu_leak), nn.Conv2d(conv3_dim, conv4_dim, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(conv4_dim), nn.LeakyReLU(relu_leak))
        self.fc = nn.Linear(conv4_dim * conv4_h * conv4_w, 1)
        self.fc_dim = conv4_dim * conv4_h * conv4_w
        self._init_weight()

    def _init_weight(self):
        self.fc.weight.data.normal_(0.0, 0.02)
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, input):
        out = self.convs(input)
        linear = self.fc(out.view(-1, self.fc_dim))
        return F.sigmoid(linear)


class RnnDropout(nn.Module):

    def __init__(self, dropout_prob, hidden_size, is_cuda):
        super().__init__()
        self.mask = torch.bernoulli(torch.Tensor(1, hidden_size).fill_(1.0 - dropout_prob))
        if is_cuda:
            self.mask = self.mask
        self.dropout_prob = dropout_prob

    def forward(self, input):
        input = input * self.mask
        input *= 1.0 / (1.0 - self.dropout_prob)
        return input


class HwLSTMCell(nn.Module):

    def __init__(self, isz, hsz, dropout_prob, is_cuda):
        super().__init__()
        self.hsz = hsz
        self.w_ih = nn.Parameter(torch.Tensor(6 * hsz, isz))
        self.w_hh = nn.Parameter(torch.Tensor(5 * hsz, hsz))
        self.b_ih = nn.Parameter(torch.Tensor(6 * hsz))
        self.rdropout = RnnDropout(dropout_prob, hsz, is_cuda)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hsz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hsz)
            hidden = hidden, hidden
        hx, cx = hidden
        input = F.linear(input, self.w_ih, self.b_ih)
        gates = F.linear(hx, self.w_hh) + input[..., :-self.hsz]
        in_gate, forget_gate, cell_gate, out_gate, r_gate = gates.chunk(5, 1)
        in_gate, forget_gate, out_gate, r_gate = map(torch.sigmoid, [in_gate, forget_gate, out_gate, r_gate])
        cell_gate = torch.tanh(cell_gate)
        k = input[..., -self.hsz:]
        cy = forget_gate * cx + in_gate * cell_gate
        hy = r_gate * out_gate * F.tanh(cy) + (1.0 - r_gate) * k
        if self.training:
            hy = self.rdropout(hy)
        return hy, cy


class HwLSTMlayer(nn.Module):

    def __init__(self, isz, hsz, dropout_prob, is_cuda):
        super().__init__()
        self.cell = HwLSTMCell(isz, hsz, dropout_prob, is_cuda)

    def forward(self, input, reverse=True):
        output, hidden = [], None
        for i in range(len(input)):
            hidden = self.cell(input[i], hidden)
            output.append(hidden[0])
        if reverse:
            output.reverse()
        return torch.stack(output)


class DeepBiLSTMModel(nn.Module):

    def __init__(self, vsz, lsz, ebd_dim, lstm_hsz, lstm_layers, dropout_prob, is_cuda, ebd_weights=None):
        super().__init__()
        self.ebd_weights = ebd_weights
        self.ebd = nn.Embedding(vsz, ebd_dim, padding_idx=PAD)
        self.lstms = nn.ModuleList([(HwLSTMlayer(lstm_hsz, lstm_hsz, dropout_prob, is_cuda) if layer != 0 else HwLSTMlayer(ebd_dim, lstm_hsz, dropout_prob, is_cuda)) for layer in range(lstm_layers)])
        self.logistic = nn.Linear(lstm_hsz, lsz)
        self.reset_parameters(ebd_dim)

    def reset_parameters(self, ebd_dim):
        stdv = 1.0 / math.sqrt(ebd_dim)
        self.logistic.weight.data.uniform_(-stdv, stdv)
        if self.ebd_weights is None:
            self.ebd.weight.data.uniform_(-stdv, stdv)
        else:
            self.ebd.weight.data.copy_(torch.from_numpy(self.ebd_weights))

    def forward(self, inp):
        inp = self.ebd(inp)
        inp = inp.permute(1, 0, 2)
        for rnn in self.lstms:
            inp = rnn(inp)
        inp = inp.permute(1, 0, 2).contiguous().view(-1, inp.size(2))
        out = self.logistic(inp)
        return F.log_softmax(out, dim=-1)


class Beauty(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet50(True)
        self.predict = nn.Linear(1000, 1)
        self._reset_parameters()

    def forward(self, x):
        out = nn.functional.relu(self.resnet(x))
        score = self.predict(out)
        return score.squeeze()

    def _reset_parameters(self):
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.predict.weight.data.uniform_(-0.1, 0.1)

    def get_trainable_parameters(self):
        return filter(lambda m: m.requires_grad, self.parameters())


FINAL_EPSILON = 0.01


INITIAL_EPSILON = 0.5


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self, state_dim, out_dim, capacity, bsz, epsilon):
        super().__init__()
        self.steps_done = 0
        self.position = 0
        self.pool = []
        self.capacity = capacity
        self.bsz = bsz
        self.epsilon = epsilon
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, out_dim)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def action(self, state):
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if random.random() > self.epsilon:
            return self(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            return longTensor([[random.randrange(2)]])

    def push(self, *args):
        if len(self) < self.capacity:
            self.pool.append(None)
        self.pool[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.pool, self.bsz)

    def __len__(self):
        return len(self.pool)


class ActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, state, values, select_props):
        state = torch.from_numpy(state).float()
        props, value = self(Variable(state))
        dist = Categorical(props)
        action = dist.sample()
        log_props = dist.log_prob(action)
        values.append(value)
        select_props.append(log_props)
        return action.data[0]


class NlpCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        return -(tgt_props * mask).sum() / mask.sum()


class DilatedGatedConv1D(nn.Module):

    def __init__(self, dilation_rate, dim):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=0.1)
        self.cnn = nn.Conv1d(dim, dim * 2, 3, padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        residual = x
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x1, x2 = x[:, :, :self.dim], x[:, :, self.dim:]
        x1 = torch.sigmoid(self.dropout(x1))
        return residual * (1 - x1) + x2 * x1


class DgCNN(nn.Module):

    def __init__(self, dim, dilation_rates: list):
        super().__init__()
        self.cnn1ds = nn.ModuleList([DilatedGatedConv1D(dilation_rate, dim) for dilation_rate in dilation_rates])

    def forward(self, x, mask):
        for layer in self.cnn1ds:
            x = layer(x) * mask
        return x


class SubjectLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.seq(x)


class SubModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Conv1d(2 * dim, dim, 3, padding=1)
        self.lr1 = nn.Linear(dim, 1)
        self.lr2 = nn.Linear(dim, 1)

    def forward(self, x):
        x = F.relu(self.cnn(x.transpose(1, 2))).transpose(1, 2)
        x1 = torch.sigmoid(self.lr1(x))
        x2 = torch.sigmoid(self.lr2(x))
        return x1, x2


class ObjModel(nn.Module):

    def __init__(self, dim, num_classes):
        super().__init__()
        self.cnn = nn.Conv1d(4 * dim, dim, 3, padding=1)
        self.lr1 = nn.Linear(dim, 1)
        self.lr2 = nn.Linear(dim, num_classes)
        self.lr3 = nn.Linear(dim, num_classes)

    def forward(self, x, shareFeat1, shareFeat2):
        x = F.relu(self.cnn(x.transpose(1, 2))).transpose(1, 2)
        x1 = torch.sigmoid(self.lr1(x))
        x2 = torch.sigmoid(self.lr2(x))
        x3 = torch.sigmoid(self.lr3(x))
        x2 = x2 * shareFeat1 * x1
        x3 = x3 * shareFeat2 * x1
        return x2, x3


class ObjectRnn(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.rnn = nn.GRU(d_model, hidden_size=d_model, batch_first=True, bidirectional=True)
        self.ln = LayerNorm(d_model * 2)

    def forward(self, x, sub_slidx, sub_elidx, pos_ebd):
        idx = gather_index(x, sub_slidx, sub_elidx)
        encode, _ = self.rnn(idx)
        encode = self.ln(encode)[:, -1, :].unsqueeze(1)
        pos_ebd = self.position(x, sub_slidx, sub_elidx, pos_ebd)
        return encode + pos_ebd

    def position(self, x, sidx, eidx, pos_ebd):
        bsz, length, _ = x.size()
        pos_idx = torch.arange(0, length).repeat(bsz, 1)
        if x.is_cuda:
            pos_idx = pos_idx
        s_pos = pos_ebd(torch.abs(pos_idx - sidx.long()))
        e_pos = pos_ebd(torch.abs(pos_idx - eidx.long()))
        return torch.cat((s_pos, e_pos), dim=-1)


class LSTM_Text(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.num_directions = 2 if self.bidirectional else 1
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, self.lstm_layers, dropout=self.dropout, bidirectional=self.bidirectional)
        self.ln = LayerNorm(self.hidden_size * self.num_directions)
        self.logistic = nn.Linear(self.hidden_size * self.num_directions, self.label_size)
        self._init_weights()

    def _init_weights(self, scope=1.0):
        self.lookup_table.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self):
        num_layers = self.lstm_layers * self.num_directions
        weight = next(self.parameters()).data
        return Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_()), Variable(weight.new(num_layers, self.batch_size, self.hidden_size).zero_())

    def forward(self, input, hidden):
        encode = self.lookup_table(input)
        lstm_out, hidden = self.lstm(encode.transpose(0, 1), hidden)
        output = self.ln(lstm_out)[-1]
        return F.log_softmax(self.logistic(output)), hidden


class GramMatrix(nn.Module):

    def forward(self, input):
        _, channels, h, w = input.size()
        out = input.view(-1, h * w)
        out = torch.mm(out, out.t())
        return out.div(channels * h * w)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super().__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()
        self.gm = GramMatrix()

    def forward(self, input):
        gm = self.gm(input.clone())
        loss = self.criterion(gm * self.weight, self.target)
        return loss


def check_layers(layers):
    """
    relu1_* - 2， relu2_* - 2， relu3_* - 4， relu4_* - 4， relu5_* - 4
    """
    in_layers = []
    for layer in layers:
        layer = layer[-3:]
        if layer[0] == '1' or layer[0] == '2':
            in_layers += [2 * (int(layer[0]) - 1) + int(layer[2]) - 1]
        else:
            in_layers += [4 * (int(layer[0]) - 3) + int(layer[2]) + 3]
    return in_layers


class Vgg_Model(nn.Module):

    def __init__(self, vgg):
        super().__init__()
        self.layers = copy.deepcopy(vgg)

    def forward(self, input, out_layers):
        relu_outs, out = [], input
        out_layers = check_layers(out_layers)
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                relu_outs.append(out)
        outs = [relu_outs[index - 1] for index in out_layers]
        return outs


class NGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim=16, context_size=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.l1 = nn.Linear(context_size * embedding_dim, 128)
        self.l2 = nn.Linear(128, vocab_size)
        self._init_weight()

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.l1(embeds))
        out = self.l2(out)
        log_probs = F.log_softmax(out)
        return log_probs

    def _init_weight(self, scope=0.1):
        self.embeddings.weight.data.uniform_(-scope, scope)
        self.l1.weight.data.uniform_(0, scope)
        self.l1.bias.data.fill_(0)
        self.l2.weight.data.uniform_(0, scope)
        self.l2.bias.data.fill_(0)


class CNN_Ranking(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.embedded_chars_left = nn.Embedding(self.src_vocab_size, self.embed_dim)
        self.embedded_chars_right = nn.Embedding(self.tgt_vocab_size, self.embed_dim)
        self.conv_left, self.conv_right = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            conv_left_name = 'conv_left_%d' % i
            conv_right_name = 'conv_right_%d' % i
            self.__setattr__(conv_left_name, nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(filter_size, self.embed_dim)))
            self.conv_left.append(self.__getattr__(conv_left_name))
            self.__setattr__(conv_right_name, nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(filter_size, self.embed_dim)))
            self.conv_right.append(self.__getattr__(conv_right_name))
        ins = len(self.filter_sizes) * self.num_filters
        self.simi_weight = nn.Parameter(torch.zeros(ins, ins))
        self.out_lr = nn.Linear(2 * ins + 1, self.hidden_size)
        self.logistic = nn.Linear(self.hidden_size, 2)
        self._init_weights()

    def forward(self, input_left, input_right):
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        enc_left = self.embedded_chars_left(input_left)
        enc_right = self.embedded_chars_right(input_right)
        enc_left = enc_left.unsqueeze(c_idx)
        enc_right = enc_right.unsqueeze(c_idx)
        enc_outs_left, enc_outs_right = [], []
        for index, (encoder_left, encoder_right) in enumerate(zip(self.conv_left, self.conv_right)):
            enc_left_ = F.relu(encoder_left(enc_left))
            enc_right_ = F.relu(encoder_right(enc_right))
            h_left = enc_left_.size()[h_idx]
            h_right = enc_right_.size()[h_idx]
            enc_left_ = F.max_pool2d(enc_left_, kernel_size=(h_left, 1))
            enc_right_ = F.max_pool2d(enc_right_, kernel_size=(h_right, 1))
            enc_left_ = enc_left_.squeeze(w_idx)
            enc_left_ = enc_left_.squeeze(h_idx)
            enc_right_ = enc_right_.squeeze(w_idx)
            enc_right_ = enc_right_.squeeze(h_idx)
            enc_outs_left.append(enc_left_)
            enc_outs_right.append(enc_right_)
        hid_in_left = torch.cat(enc_outs_left, c_idx)
        enc_outs_right = torch.cat(enc_outs_right, c_idx)
        transform_left = torch.mm(hid_in_left, self.simi_weight)
        sims = torch.sum(torch.mm(transform_left, enc_outs_right.t()), dim=c_idx, keepdim=True)
        new_input = torch.cat([hid_in_left, sims, enc_outs_right], c_idx)
        out = F.dropout(self.out_lr(new_input), p=self.dropout)
        return F.log_softmax(self.logistic(out))

    def _init_weights(self, scope=1.0):
        self.embedded_chars_left.weight.data.uniform_(-scope, scope)
        self.embedded_chars_right.weight.data.uniform_(-scope, scope)
        init.xavier_uniform(self.simi_weight)
        init.xavier_uniform(self.out_lr.weight)
        init.xavier_uniform(self.logistic.weight)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(d_k)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.layer_norm(q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual
        return output, attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super().__init__()
        self.slf_mh = MultiHeadAtt(n_head, d_model, dropout)
        self.dec_mh = MultiHeadAtt(n_head, d_model, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask, dec_enc_attn_mask):
        dec_output = self.slf_mh(dec_input, dec_input, dec_input, slf_attn_mask)
        dec_output = self.dec_mh(dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output = self.pw(dec_output)
        return dec_output


class CrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, props, tgt):
        tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        return -(tgt_props * mask).sum() / mask.sum()


class SelfCriticCriterion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, props, s_words, tgt, advantage):
        advantage = (advantage - advantage.mean()) / advantage.std().clamp(min=1e-08)
        s_props = props.gather(2, s_words.unsqueeze(2)).squeeze()
        mask = (tgt > 0).float()
        advantage = advantage.unsqueeze(1).repeat(1, mask.size(1))
        advantage = advantage.detach()
        return -(s_props * mask * advantage).sum() / mask.sum()


class RelationNet(nn.Module):

    def __init__(self, word_size, answer_size, max_s_len, max_q_len, use_cuda, story_len=20, emb_dim=32, story_hsz=32, story_layers=1, question_hsz=32, question_layers=1):
        super().__init__()
        self.use_cuda = use_cuda
        self.max_s_len = max_s_len
        self.max_q_len = max_q_len
        self.story_len = story_len
        self.emb_dim = emb_dim
        self.story_hsz = story_hsz
        self.emb = nn.Embedding(word_size, emb_dim)
        self.story_rnn = torch.nn.LSTM(input_size=emb_dim, hidden_size=story_hsz, num_layers=story_layers, batch_first=True)
        self.question_rnn = torch.nn.LSTM(input_size=emb_dim, hidden_size=question_hsz, num_layers=question_layers, batch_first=True)
        self.g1 = nn.Linear(2 * story_len + 2 * story_hsz + question_hsz, 256)
        self.g2 = nn.Linear(256, 256)
        self.g3 = nn.Linear(256, 256)
        self.g4 = nn.Linear(256, 256)
        self.f1 = nn.Linear(256, 256)
        self.f2 = nn.Linear(256, 512)
        self.f3 = nn.Linear(512, answer_size)
        self._reset_parameters()

    def _reset_parameters(self, stddev=0.1):
        self.emb.weight.data.normal_(std=stddev)
        self.g1.weight.data.normal_(std=stddev)
        self.g1.bias.data.fill_(0)
        self.g2.weight.data.normal_(std=stddev)
        self.g2.bias.data.fill_(0)
        self.g3.weight.data.normal_(std=stddev)
        self.g3.bias.data.fill_(0)
        self.g4.weight.data.normal_(std=stddev)
        self.g4.bias.data.fill_(0)
        self.f1.weight.data.normal_(std=stddev)
        self.f1.bias.data.fill_(0)
        self.f2.weight.data.normal_(std=stddev)
        self.f2.bias.data.fill_(0)
        self.f3.weight.data.normal_(std=stddev)
        self.f3.bias.data.fill_(0)

    def g_theta(self, x):
        x = F.relu_(self.g1(x))
        x = F.relu_(self.g2(x))
        x = F.relu_(self.g3(x))
        x = F.relu_(self.g4(x))
        return x

    def init_tags(self):
        tags = torch.zeros((self.story_len, self.story_len))
        if self.use_cuda:
            tags = tags
        for i in range(self.story_len):
            tags[i, i].fill_(1)
        return tags

    def forward(self, story, question):
        tags = self.init_tags()
        bsz = story.shape[0]
        s_emb = self.emb(story)
        s_emb = s_emb.view(-1, self.max_s_len, self.emb_dim)
        _, (s_state, _) = self.story_rnn(s_emb)
        s_state = s_state[-1, :, :]
        s_state = s_state.view(-1, self.story_len, self.story_hsz)
        s_tags = tags.unsqueeze(0)
        s_tags = s_tags.repeat((bsz, 1, 1))
        story_objects = torch.cat((s_state, s_tags), dim=2)
        q_emb = self.emb(question)
        _, (q_state, _) = self.question_rnn(q_emb)
        q_state = q_state[-1, :, :]
        sum_g_theta = 0
        for i in range(self.story_len):
            this_tensor = story_objects[:, i, :]
            for j in range(self.story_len):
                u = torch.cat((this_tensor, story_objects[:, j, :], q_state), dim=1)
                g = self.g_theta(u)
                sum_g_theta = torch.add(sum_g_theta, g)
        out = F.relu(self.f1(sum_g_theta))
        out = F.relu(self.f2(out))
        out = self.f3(out)
        return out


class VAE(nn.Module):

    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lookup_table.weight.data.copy_(torch.from_numpy(self.pre_w2v))
        self.hw = Highway(self.hw_layers, self.hw_hsz, F.relu)
        self.encode = Encoder(self.embed_dim, self.enc_hsz, self.enc_layers, self.dropout)
        self._enc_mu = nn.Linear(self.enc_hsz * 2, self.latent_dim)
        self._enc_log_sigma = nn.Linear(self.enc_hsz * 2, self.latent_dim)
        self.decode = Decoder(self.embed_dim, self.latent_dim, self.dec_hsz, self.dec_layers, self.dropout, self.vocab_size)
        self._init_weight()

    def forward(self, enc_input, dec_input, enc_hidden, dec_hidden):
        enc_ = self.lookup_table(enc_input)
        enc_ = F.dropout(self.hw(enc_), p=self.dropout)
        enc_output, enc_hidden = self.encode(enc_, enc_hidden)
        z = self._gaussian(enc_output)
        dec_ = self.lookup_table(dec_input)
        dec, dec_hidden = self.decode(dec_, z, dec_hidden)
        return dec, self.latent_loss, enc_hidden, dec_hidden

    def _gaussian(self, enc_output):

        def latent_loss(mu, sigma):
            pow_mu = mu * mu
            pow_sigma = sigma * sigma
            return 0.5 * torch.sum(pow_mu + pow_sigma - torch.log(pow_sigma) - 1, dim=-1).mean()
        mu = self._enc_mu(enc_output)
        sigma = torch.exp(0.5 * self._enc_log_sigma(enc_output))
        self.latent_loss = latent_loss(mu, sigma)
        weight = next(self.parameters()).data
        std_z = Variable(weight.new(*sigma.size()), requires_grad=False)
        std_z.data.copy_(torch.from_numpy(np.random.normal(size=sigma.size())))
        return mu + sigma * std_z

    def _init_weight(self):
        init.xavier_normal(self._enc_mu.weight)
        init.xavier_normal(self._enc_log_sigma.weight)

    def generate(self, max_len):
        size = 1, self.latent_dim
        weight = next(self.parameters()).data
        z = Variable(weight.new(*size), volatile=True)
        z.data.copy_(torch.from_numpy(np.random.normal(size=size)))
        prob = torch.LongTensor([BOS])
        input = Variable(prob.unsqueeze(1), volatile=True)
        if weight.is_cuda:
            input = input
        portry = ''
        hidden = self.decode.init_hidden(1)
        for index in range(1, max_len + 1):
            encode = self.lookup_table(input)
            output, hidden = self.decode(encode, z, hidden)
            prob = output.squeeze().data
            next_word = torch.max(prob, -1)[1].tolist()[0]
            input.data.fill_(next_word)
            if index % 5 == 0:
                portry += self.idx2word[next_word]
                portry += '，'
            else:
                portry += self.idx2word[next_word]
        return portry[:-1] + '。'


mel_n_channels = 40


mel_window_step = 10


model_embedding_size = 256


model_hidden_size = 256


model_num_layers = 3


partials_n_frames = 160


sampling_rate = 16000


class VoiceEncoder(nn.Module):

    def __init__(self, model='../model/speaker.pt', device='cpu'):
        """
        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). 
        If None, defaults to cuda if it is available on your machine, otherwise the model will 
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        """
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        checkpoint = torch.load(model, map_location='cpu')
        self.load_state_dict(checkpoint['model_state'], strict=False)
        self

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape 
        (batch_size, n_frames, n_channels) 
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size). 
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to 
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the 
        mel spectrogram slices are returned, so as to make each partial utterance waveform 
        correspond to its spectrogram.
    
        The returned ranges may be indexing further than the length of the waveform. It is 
        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.
    
        :param n_samples: the number of samples in the waveform
        :param rate: how many partial utterances should occur per second. Partial utterances must 
        cover the span of the entire utterance, thus the rate should not be lower than the inverse 
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and 
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have 
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present, 
        then the last partial utterance will be considered by zero-padding the audio. Otherwise, 
        it will be discarded. If there aren't enough frames for one partial utterance, 
        this parameter is ignored so that the function always returns at least one slice.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
        respectively the waveform and the mel spectrogram with these slices to obtain the partial 
        utterances.
        """
        assert 0 < min_coverage <= 1
        samples_per_frame = int(sampling_rate * mel_window_step / 1000)
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round(sampling_rate / rate / samples_per_frame))
        assert 0 < frame_step, 'The rate is too high'
        assert frame_step <= partials_n_frames, 'The rate is too low, it should be %f at least' % (sampling_rate / (samples_per_frame * partials_n_frames))
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]
        return wav_slices, mel_slices

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
        """
        Computes an embedding for a single utterance. The utterance is divided in partial 
        utterances and an embedding is computed for each. The complete utterance embedding is the 
        L2-normed average embedding of the partial utterances.
        
        TODO: independent batched version of this function
    
        :param wav: a preprocessed utterance waveform as a numpy array of float32
        :param return_partials: if True, the partial embeddings will also be returned along with 
        the wav slices corresponding to each partial utterance.
        :param rate: how many partial utterances should occur per second. Partial utterances must 
        cover the span of the entire utterance, thus the rate should not be lower than the inverse 
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and 
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have 
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present, 
        then the last partial utterance will be considered by zero-padding the audio. Otherwise, 
        it will be discarded. If there aren't enough frames for one partial utterance, 
        this parameter is ignored so that the function always returns at least one slice.
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
        returned.
        """
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), 'constant')
        mel = utils.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels)
            partial_embeds = self(mels).cpu().numpy()
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        """
        Compute the embedding of a collection of wavs (presumably from the same speaker) by 
        averaging their embedding and L2-normalizing it.
        
        :param wavs: list of wavs a numpy arrays of float32.
        :param kwargs: extra arguments to embed_utterance()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).
        """
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, pad='ConstantPad1d', pad_params={'value': 0.0}):
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.conv(self.pad(x))[:, :, :x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        return self.deconv(x)[:, :, :-self.stride]


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(self, kernel_size=3, channels=32, dilation=1, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_causal_conv=False):
        """Initialize ResidualStack module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.
        """
        super(ResidualStack, self).__init__()
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
            self.stack = torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params), torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, 1, bias=bias))
        else:
            self.stack = torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), CausalConv1d(channels, channels, kernel_size, dilation=dilation, bias=bias, pad=pad, pad_params=pad_params), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, 1, bias=bias))
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, chennels, T).
        """
        return self.stack(c) + self.skip_layer(c)


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self, kernel_size=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, dropout=0.0, dilation=1, bias=True, use_causal_conv=False):
        """Initialize ResidualBlock module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.
        """
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).
        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)
        return x, s


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode='nearest'):
        """Initialize Stretch2d module.
        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.
        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, C, F, T).
        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),
        """
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(self, upsample_scales, nonlinear_activation=None, nonlinear_activation_params={}, interpolate_mode='nearest', freq_axis_kernel_size=1, use_causal_conv=False):
        """Initialize upsampling network module.
        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]
            assert (freq_axis_kernel_size - 1) % 2 == 0, 'Not support even number freq axis kernel size.'
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = freq_axis_kernel_size, scale * 2 + 1
            if use_causal_conv:
                padding = freq_axis_padding, scale * 2
            else:
                padding = freq_axis_padding, scale
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, C, T).
        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).
        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., :c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(self, upsample_scales, nonlinear_activation=None, nonlinear_activation_params={}, interpolate_mode='nearest', freq_axis_kernel_size=1, aux_channels=80, aux_context_window=0, use_causal_conv=False):
        """Initialize convolution + upsampling network module.
        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.
        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        self.conv_in = Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(upsample_scales=upsample_scales, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, interpolate_mode=interpolate_mode, freq_axis_kernel_size=freq_axis_kernel_size, use_causal_conv=use_causal_conv)

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, C, T').
        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).
        Note:
            The length of inputs considers the context window size.
        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self, in_channels=80, out_channels=1, kernel_size=7, channels=512, bias=True, upsample_scales=[8, 8, 2, 2], stack_kernel_size=3, stacks=3, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}, use_final_nonlinear_activation=True, use_weight_norm=True, use_causal_conv=False):
        """Initialize MelGANGenerator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.
        """
        super(MelGANGenerator, self).__init__()
        assert channels >= np.prod(upsample_scales)
        assert channels % 2 ** len(upsample_scales) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        layers = []
        if not use_causal_conv:
            layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias)]
        else:
            layers += [CausalConv1d(in_channels, channels, kernel_size, bias=bias, pad=pad, pad_params=pad_params)]
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [torch.nn.ConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_scale * 2, stride=upsample_scale, padding=upsample_scale // 2 + upsample_scale % 2, output_padding=upsample_scale % 2, bias=bias)]
            else:
                layers += [CausalConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_scale * 2, stride=upsample_scale, bias=bias)]
            for j in range(stacks):
                layers += [ResidualStack(kernel_size=stack_kernel_size, channels=channels // 2 ** (i + 1), dilation=stack_kernel_size ** j, bias=bias, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params, pad=pad, pad_params=pad_params, use_causal_conv=use_causal_conv)]
        layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
        if not use_causal_conv:
            layers += [getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params), torch.nn.Conv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias)]
        else:
            layers += [CausalConv1d(channels // 2 ** (i + 1), out_channels, kernel_size, bias=bias, pad=pad, pad_params=pad_params)]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]
        self.melgan = torch.nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).
        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f'Reset parameters in {m}.')
        self.apply(_reset_parameters)

    def inference(self, c):
        """Perform inference.
        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).
        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float)
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, layers=30, stacks=3, residual_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, aux_context_window=2, dropout=0.0, bias=True, use_weight_norm=True, use_causal_conv=False, upsample_conditional_features=True, upsample_net='ConvInUpsampleNetwork', upsample_params={'upsample_scales': [4, 4, 4, 4]}):
        """Initialize Parallel WaveGAN Generator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.
        """
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)
        if upsample_conditional_features:
            upsample_params.update({'use_causal_conv': use_causal_conv})
            if upsample_net == 'MelGANGenerator':
                assert aux_context_window == 0
                upsample_params.update({'use_weight_norm': False, 'use_final_nonlinear_activation': False})
                self.upsample_net = getattr(models, upsample_net)(**upsample_params)
            else:
                if upsample_net == 'ConvInUpsampleNetwork':
                    upsample_params.update({'aux_channels': aux_channels, 'aux_context_window': aux_context_window})
                self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            self.upsample_factor = np.prod(upsample_params['upsample_scales'])
        else:
            self.upsample_net = None
            self.upsample_factor = 1
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(kernel_size=kernel_size, residual_channels=residual_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=aux_channels, dilation=dilation, dropout=dropout, bias=bias, use_causal_conv=use_causal_conv)
            self.conv_layers += [conv]
        self.last_conv_layers = torch.nn.ModuleList([torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, skip_channels, bias=True), torch.nn.ReLU(inplace=True), Conv1d1x1(skip_channels, out_channels, bias=True)])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
        Returns:
            Tensor: Output tensor (B, out_channels, T)
        """
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')
        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2 ** x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def inference(self, c=None, x=None):
        """Perform inference.
        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
        Returns:
            Tensor: Output tensor (T, out_channels)
        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float)
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float)
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    assert taps % 2 == 0, 'The number of taps mush be even number.'
    assert 0.0 < cutoff_ratio < 1.0, 'Cutoff ratio must be > 0.0 and < 1.0.'
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio
    w = kaiser(taps + 1, beta)
    h = h_i * w
    return h


class PQMF(torch.nn.Module):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Initilize PQMF module.
        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.
        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super(PQMF, self).__init__()
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - taps / 2) + (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (np.arange(taps + 1) - taps / 2) - (-1) ** k * np.pi / 4)
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)
        self.register_buffer('analysis_filter', analysis_filter)
        self.register_buffer('synthesis_filter', synthesis_filter)
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer('updown_filter', updown_filter)
        self.subbands = subbands
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


class BasicConv(nn.Module):

    def __init__(self, ind, outd, kr_size, stride, padding, lr=0.1, bias=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(ind, outd, kr_size, stride, padding, bias=bias), nn.BatchNorm2d(outd), nn.LeakyReLU(lr))

    def forward(self, x):
        return self.layers(x)


def load_classes(inp='data/coco.names'):
    return [c.strip() for c in open(inp)]


OUT_DIM = 3 * (len(load_classes()) + 5)


DETECT_DICT = {'first': [1024, (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)], 'second': [768, (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)], 'third': [384, (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)]}


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


class BasicPred(nn.Module):

    def __init__(self, structs, use_cuda, anchors, classes, height=416, route_index=0):
        super().__init__()
        self.ri = route_index
        self.classes = classes
        self.height = height
        self.anchors = anchors
        self.torch = torch.cuda if use_cuda else torch
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        in_dim = structs[0]
        self.layers = nn.ModuleList()
        for s in structs[1:]:
            if len(s) == 4:
                out_dim, kr_size, stride, padding = s
                layer = BasicConv(in_dim, out_dim, kr_size, stride, padding)
            else:
                out_dim, kr_size, stride, padding, _ = s
                layer = nn.Conv2d(in_dim, out_dim, kr_size, stride, padding)
            in_dim = out_dim
            self.layers.append(layer)

    def forward(self, x, targets=None):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if self.ri != 0 and index == self.ri:
                output = x
        detections = self.predict_transform(x, targets)
        if self.ri != 0:
            return detections, output
        else:
            return detections

    def predict_transform(self, inp, targets):
        """
        Code originally from https://github.com/eriklindernoren/PyTorch-YOLOv3.
        """

        def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
            """
            target - [bsz, max_obj, 5]
            """
            nB = target.size(0)
            nA = num_anchors
            nC = num_classes
            nG = grid_size
            mask = torch.zeros(nB, nA, nG, nG)
            conf_mask = torch.ones(nB, nA, nG, nG)
            tx = torch.zeros(nB, nA, nG, nG)
            ty = torch.zeros(nB, nA, nG, nG)
            tw = torch.zeros(nB, nA, nG, nG)
            th = torch.zeros(nB, nA, nG, nG)
            tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
            tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)
            nGT = 0
            nCorrect = 0
            for b in range(nB):
                for t in range(target.shape[1]):
                    if target[b, t].sum() == 0:
                        continue
                    nGT += 1
                    gx = target[b, t, 1] * nG
                    gy = target[b, t, 2] * nG
                    gw = target[b, t, 3] * nG
                    gh = target[b, t, 4] * nG
                    gi = int(gx)
                    gj = int(gy)
                    gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                    anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
                    anch_ious = bbox_iou(gt_box, anchor_shapes)
                    conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
                    best_n = np.argmax(anch_ious)
                    gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
                    pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
                    mask[b, best_n, gj, gi] = 1
                    conf_mask[b, best_n, gj, gi] = 1
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                    target_label = int(target[b, t, 0])
                    tcls[b, best_n, gj, gi, target_label] = 1
                    tconf[b, best_n, gj, gi] = 1
                    iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                    pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
                    score = pred_conf[b, best_n, gj, gi]
                    if iou > 0.5 and pred_label == target_label and score > 0.5:
                        nCorrect += 1
            return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls
        bsz = inp.size(0)
        grid_size = inp.size(2)
        stride = self.height // grid_size
        bbox_attrs = 5 + self.classes
        num_anchors = len(self.anchors)
        prediction = inp.view(bsz, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        anchors = self.torch.FloatTensor([(a[0] / stride, a[1] / stride) for a in self.anchors])
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(self.torch.FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(self.torch.FloatTensor)
        anchor_w = anchors[:, 0].view((1, num_anchors, 1, 1))
        anchor_h = anchors[:, 1].view((1, num_anchors, 1, 1))
        pred_boxes = self.torch.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        if targets is not None:
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes=pred_boxes.cpu().data, pred_conf=pred_conf.cpu().data, pred_cls=pred_cls.cpu().data, target=targets.cpu().data, anchors=anchors.cpu().data, num_anchors=num_anchors, num_classes=self.classes, grid_size=grid_size, ignore_thres=0.5, img_dim=self.height)
            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)
            mask = mask.type(self.torch.ByteTensor)
            conf_mask = conf_mask.type(self.torch.ByteTensor)
            with torch.no_grad():
                tx = tx.type(self.torch.FloatTensor)
                ty = ty.type(self.torch.FloatTensor)
                tw = tw.type(self.torch.FloatTensor)
                th = th.type(self.torch.FloatTensor)
                tconf = tconf.type(self.torch.FloatTensor)
                tcls = tcls.type(self.torch.LongTensor)
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall, precision
        else:
            output = torch.cat((pred_boxes.view(bsz, -1, 4) * stride, pred_conf.view(bsz, -1, 1), pred_cls.view(bsz, -1, self.classes)), -1)
            return output


class FirstPred(BasicPred):

    def __init__(self, structs, use_cuda, classes, route_index=4, anchors=[(116, 90), (156, 198), (373, 326)]):
        super().__init__(structs, use_cuda, anchors, classes, route_index=route_index)


LOSS_NAMES = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall', 'precision']


class BasicLayer(nn.Module):

    def __init__(self, conv_1, conv_2, times):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(times):
            self.layers.append(BasicConv(*conv_1))
            self.layers.append(BasicConv(*conv_2))

    def forward(self, x):
        residual = x
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index % 2 == 1:
                x += residual
                residual = x
        return x


class LayerFive(BasicLayer):

    def __init__(self):
        super().__init__((1024, 512, 1, 1, 0), (512, 1024, 3, 1, 1), 4)


class LayerFour(BasicLayer):

    def __init__(self):
        super().__init__((512, 256, 1, 1, 0), (256, 512, 3, 1, 1), 8)


class LayerOne(BasicLayer):

    def __init__(self):
        super().__init__((64, 32, 1, 1, 0), (32, 64, 3, 1, 1), 1)


class LayerThree(BasicLayer):

    def __init__(self):
        super().__init__((256, 128, 1, 1, 0), (128, 256, 3, 1, 1), 8)


class LayerTwo(BasicLayer):

    def __init__(self):
        super().__init__((128, 64, 1, 1, 0), (64, 128, 3, 1, 1), 2)


class SecondPred(BasicPred):

    def __init__(self, structs, use_cuda, classes, route_index=4, anchors=[(30, 61), (62, 45), (59, 119)]):
        super().__init__(structs, use_cuda, anchors, classes, route_index=route_index)


class ThirdPred(BasicPred):

    def __init__(self, structs, use_cuda, classes, height=416, anchors=[(10, 13), (16, 30), (33, 23)]):
        super().__init__(structs, use_cuda, anchors, classes)


class DarkNet(nn.Module):

    def __init__(self, use_cuda, nClasses):
        super().__init__()
        self.conv_1 = BasicConv(256, 512, 3, 2, 1)
        self.seq_1 = nn.Sequential(BasicConv(3, 32, 3, 1, 1), BasicConv(32, 64, 3, 2, 1), LayerOne(), BasicConv(64, 128, 3, 2, 1), LayerTwo(), BasicConv(128, 256, 3, 2, 1), LayerThree())
        self.seq_2 = nn.Sequential(BasicConv(512, 1024, 3, 2, 1), LayerFive())
        self.layer_4 = LayerFour()
        self.uns_1 = nn.Sequential(BasicConv(512, 256, 1, 1, 0), nn.Upsample(scale_factor=2, mode='bilinear'))
        self.uns_2 = nn.Sequential(BasicConv(256, 128, 1, 1, 0), nn.Upsample(scale_factor=2, mode='bilinear'))
        self.pred_1 = FirstPred(DETECT_DICT['first'], use_cuda, nClasses)
        self.pred_2 = SecondPred(DETECT_DICT['second'], use_cuda, nClasses)
        self.pred_3 = ThirdPred(DETECT_DICT['third'], use_cuda, nClasses)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                layer.weight.data.normal_(0.0, 0.02)
            if type(layer) == nn.BatchNorm2d:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, x, targets=None):
        gather_losses = defaultdict(float)
        x = self.seq_1(x)
        r_0 = x
        x = self.layer_4(self.conv_1(x))
        r_1 = x
        x = self.seq_2(x)
        if targets is not None:
            (sum_loss, *losses), x = self.pred_1(x, targets)
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss
        else:
            det_1, x = self.pred_1(x)
        x = self.uns_1(x)
        x = torch.cat((x, r_1), 1)
        if targets is not None:
            (this_loss, *losses), x = self.pred_2(x, targets)
            sum_loss += this_loss
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss
        else:
            det_2, x = self.pred_2(x)
        x = self.uns_2(x)
        x = torch.cat((x, r_0), 1)
        if targets is not None:
            this_loss, *losses = self.pred_3(x, targets)
            sum_loss += this_loss
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss
            gather_losses['recall'] /= 3
            gather_losses['precision'] /= 3
            return sum_loss, gather_losses
        else:
            det_3 = self.pred_3(x)
            return torch.cat((det_1, det_2, det_3), 1)

