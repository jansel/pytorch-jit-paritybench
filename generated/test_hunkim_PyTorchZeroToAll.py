import sys
_module = sys.modules[__name__]
del sys
name_dataset = _module
seq2seq_models = _module
text_loader = _module

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


from torch import nn


from torch import tensor


from torch import sigmoid


import torch.nn.functional as F


import torch.optim as optim


from torch import optim


from torch import from_numpy


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch import max


from torch import cuda


from torch.utils import data


from torchvision import datasets


from torchvision import transforms


import time


import torch.nn as nn


from torch.autograd import Variable


import math


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import itertools


batch_size = 1


embedding_size = 3


hidden_size = 100


input_size = 5


num_classes = 5


sequence_length = 6


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size, hidden_size=5, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        emb = self.embedding(x)
        emb = emb.view(batch_size, sequence_length, -1)
        out, _ = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes))


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.log_softmax(x)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embed = self.embedding(input.view(1, -1))
        embed = embed.view(1, 1, -1)
        output, hidden = self.gru(embed, hidden)
        output = self.linear(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        if torch.cuda.is_available():
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        else:
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        return Variable(hidden)


def create_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor)
    else:
        return Variable(tensor)


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        embedded = self.embedding(input)
        gru_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)
        fc_output = self.fc(hidden[-1])
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_variable(hidden)


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor)
    else:
        return Variable(tensor)


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def init_hidden(self):
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_hiddens):
        rnn_input = self.embedding(word_input).view(1, 1, -1)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        attn_weights = self.get_att_weight(rnn_output.squeeze(0), encoder_hiddens)
        context = attn_weights.bmm(encoder_hiddens.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = self.out(torch.cat((rnn_output, context), 1))
        return output, hidden, attn_weights

    def get_att_weight(self, hidden, encoder_hiddens):
        seq_len = len(encoder_hiddens)
        attn_scores = cuda_variable(torch.zeros(seq_len))
        for i in range(seq_len):
            attn_scores[i] = self.get_att_score(hidden, encoder_hiddens[i])
        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, hidden, encoder_hidden):
        score = self.attn(encoder_hidden)
        return torch.dot(hidden.view(-1), score.view(-1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.rand([1, 1, 4])], {}),
     True),
    (InceptionA,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RNNClassifier,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     False),
]

class Test_hunkim_PyTorchZeroToAll(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

