import sys
_module = sys.modules[__name__]
del sys
config = _module
evaluate = _module
load = _module
main = _module
model = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import re


import torch.nn as nn


import torch.nn.functional as F


from torch import optim


import torch.backends.cudnn as cudnn


import itertools


import math


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0 if n_layers == 1 else dropout, bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


USE_CUDA = torch.cuda.is_available()


device = torch.device('cuda' if USE_CUDA else 'cpu')


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attn_energies = torch.zeros(batch_size, max_len)
        attn_energies = attn_energies
        for b in range(batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, (b)], encoder_outputs[i, b].unsqueeze(0))
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy


class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0 if n_layers == 1 else dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        if embedded.size(0) != 1:
            raise ValueError('Decoder input sequence length should be 1')
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, attn_weights


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'embedding': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_ywk991112_pytorch_chatbot(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

