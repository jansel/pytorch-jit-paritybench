import sys
_module = sys.modules[__name__]
del sys
chatspace = _module
data = _module
corpus = _module
dataset = _module
indexer = _module
vocab = _module
inference = _module
model = _module
components = _module
char_conv = _module
char_lstm = _module
embed = _module
projection = _module
seq_fnn = _module
time_distributed = _module
model = _module
resource = _module
train = _module
metric = _module
trainer = _module
setup = _module
tests = _module
evaluation_test = _module
inference_test = _module
jit_evaluation_test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import logging


import re


from typing import Dict


from typing import Generator


from typing import Iterable


from typing import List


from typing import Optional


from typing import Union


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import numpy as np


from torch.optim.adam import Adam


class CharConvolution(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=config['embedding_dim'],
            out_channels=config['cnn_features'], kernel_size=config[
            'cnn_filter'])
        self.conv2 = nn.Conv1d(in_channels=config['cnn_features'],
            out_channels=config['cnn_features'], kernel_size=config[
            'cnn_filter'] * 2 + 1)
        self.conv3 = nn.Conv1d(in_channels=config['cnn_features'] * 2,
            out_channels=config['cnn_features'], kernel_size=config[
            'cnn_filter'] * 2 - 1)
        self.padding_1 = nn.ConstantPad1d(1, 0)
        self.padding_2 = nn.ConstantPad1d(3, 0)
        self.padding_3 = nn.ConstantPad1d(2, 0)

    def forward(self, embed_input):
        embed_input = torch.transpose(embed_input, dim0=1, dim1=2)
        conv1_output = self.conv1(embed_input)
        conv1_paded = self.padding_1(conv1_output)
        conv2_output = self.conv2(conv1_paded)
        conv2_paded = self.padding_2(conv2_output)
        conv3_input = torch.cat((conv1_paded, conv2_paded), dim=1)
        conv3_output1 = self.conv3(conv3_input)
        conv3_paded1 = self.padding_3(conv3_output1)
        conv3_output2 = self.conv3(embed_input)
        conv3_paded2 = self.padding_3(conv3_output2)
        return torch.cat((conv3_paded1, conv3_paded2, conv3_input), dim=1)


class CharLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(input_size=config['cnn_features'] // 2,
            hidden_size=config['cnn_features'] // 2, num_layers=config[
            'lstm_layers'], bidirectional=config['lstm_bidirectional'],
            batch_first=True)

    def forward(self, x, length):
        return self.lstm(x)[0]


class CharEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'],
            embedding_dim=config['embedding_dim'])

    def forward(self, input_seq):
        return self.embedding(input_seq)


class Projection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_fnn = TimeDistributed(nn.Linear(config['cnn_features'], 3))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = torch.transpose(x, 1, 0)
        x = self.seq_fnn(x)
        x = torch.transpose(x, 1, 0)
        x = self.softmax(x)
        return x


class SequentialFNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.time_distributed_1 = TimeDistributed(nn.Linear(in_features=
            config['cnn_features'] * 4, out_features=config['cnn_features']))
        self.time_distributed_2 = TimeDistributed(nn.Linear(in_features=
            config['cnn_features'], out_features=config['cnn_features'] // 2))

    def forward(self, conv_embed):
        x = torch.transpose(conv_embed, 2, 1)
        x = self.time_distributed_1(x)
        x = self.time_distributed_2(x)
        return x


class TimeDistributed(nn.Module):

    def __init__(self, layer, activation='relu'):
        super().__init__()
        self.layer = layer
        self.activation = self.select_activation(activation)

    def forward(self, x):
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.layer(x_reshaped)
        y = self.activation(y)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y

    def select_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        raise KeyError


class ChatSpaceModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed = CharEmbedding(config)
        self.conv = CharConvolution(config)
        self.lstm = CharLSTM(config)
        self.projection = Projection(config)
        self.fnn = SequentialFNN(config)
        self.batch_normalization = nn.BatchNorm1d(4 * config['cnn_features'])
        self.layer_normalization = nn.LayerNorm(config['cnn_features'])

    def forward(self, input_seq, length) ->torch.Tensor:
        x = self.embed.forward(input_seq)
        x = self.conv.forward(x)
        x = self.batch_normalization.forward(x)
        x = self.fnn.forward(x)
        x = self.lstm.forward(x, length)
        x = self.layer_normalization(x)
        x = self.projection.forward(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pingpong_ai_chatspace(_paritybench_base):
    pass
    def test_000(self):
        self._check(CharEmbedding(*[], **{'config': _mock_config(vocab_size=4, embedding_dim=4)}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(Projection(*[], **{'config': _mock_config(cnn_features=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(SequentialFNN(*[], **{'config': _mock_config(cnn_features=4)}), [torch.rand([16, 16, 4])], {})

    def test_003(self):
        self._check(TimeDistributed(*[], **{'layer': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

