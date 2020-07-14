import sys
_module = sys.modules[__name__]
del sys
Simulate_Extreme_Cases_Jump = _module
Simulate_Extreme_Cases_Zero_Epoch = _module
char_data_iterator = _module
layers_torch = _module
yellowfin_backup = _module
long_stress_test = _module
nn1 = _module
nn1_stress_test = _module
main = _module
models = _module
resnext = _module
utils = _module
debug_plot = _module
yellowfin = _module
yellowfin_test = _module
data = _module
generate = _module
main = _module
model = _module

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


import numpy as np


from torch.autograd import Variable


import torch.nn as nn


import math


import copy


import time


import pandas as pd


import matplotlib


import matplotlib.pyplot as plt


import torch.nn.functional as F


from scipy import sparse


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


import logging


import torch.nn.init as init


class RNN_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size))
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size))
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [c_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp = outputs.size()[0], outputs.size()[1]
        out = outputs.contiguous().view(shp[0] * shp[1], self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)
        return out

    def print_log(self):
        model_name = '_regular-LSTM_'
        model_log = ' Regular LSTM.......'
        return model_name, model_log


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0)


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if nonlinearity == 'tanh':
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if s is None:
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    return s.squeeze()


class AttentionWordRNN(nn.Module):

    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional=True, init_range=0.1, use_lstm=False):
        super(AttentionWordRNN, self).__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.use_lstm = use_lstm
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            if use_lstm:
                None
                self.word_gru = nn.LSTM(embed_size, word_gru_hidden, bidirectional=True)
            else:
                self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 2 * word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
        else:
            if use_lstm:
                self.word_gru = nn.LSTM(embed_size, word_gru_hidden, bidirectional=False)
            else:
                self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-init_range, init_range)
        self.weight_proj_word.data.uniform_(-init_range, init_range)

    def forward(self, embed, state_word):
        embedded = self.lookup(embed)
        output_word, state_word = self.word_gru(embedded, state_word)
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1, 0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1, 0))
        return word_attn_vectors, state_word, word_attn_norm

    def init_hidden(self):
        if self.bidirectional == True:
            if self.use_lstm == True:
                return [Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden)), Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))]
            else:
                return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        elif self.use_lstm == True:
            return [Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden)), Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))]
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))


class MixtureSoftmax(nn.Module):

    def __init__(self, batch_size, word_gru_hidden, feature_dim, n_classes, bidirectional=True):
        super(MixtureSoftmax, self).__init__()
        word_gru_hidden = 0
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.feature_dim = feature_dim
        if bidirectional == True:
            self.linear = nn.Linear(2 * 2 * word_gru_hidden + feature_dim, n_classes)
        else:
            self.linear = nn.Linear(2 * word_gru_hidden + feature_dim, n_classes)

    def forward(self, word_attention_vectors, features):
        mixture_input = features
        final_map = self.linear(mixture_input)
        return final_map


class Block(nn.Module):
    """Grouped convolution block."""
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * group_width))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionWordRNN,
     lambda: ([], {'batch_size': 4, 'num_tokens': 4, 'embed_size': 4, 'word_gru_hidden': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.rand([2, 4, 4])], {}),
     False),
    (Block,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixtureSoftmax,
     lambda: ([], {'batch_size': 4, 'word_gru_hidden': 4, 'feature_dim': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_JianGoForIt_YellowFin_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

