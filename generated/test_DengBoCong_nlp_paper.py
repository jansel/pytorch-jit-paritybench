import sys
_module = sys.modules[__name__]
del sys
batch_normalization = _module
bpe = _module
conv2d_cosnorm = _module
data_enhancement = _module
group_normalization = _module
data_utils = _module
pre_treat = _module
utils = _module
get_config = _module
logME = _module
chatter = _module
seq2seq = _module
seq2seq_chatter = _module
DAM = _module
InferSent = _module
gpt2 = _module
informer = _module
nbt = _module
smn = _module
common = _module
kb = _module
model = _module
tracker = _module
task_chatter = _module
transformer = _module
attention = _module
en_text_to_phoneme = _module
preprocess_tfrecord = _module
search_kits = _module
setup = _module

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


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import copy


import time


import torch.nn as nn


from typing import Tuple


import torch.nn.functional as F


import random


import torch.optim as optim


class Encoder(nn.Module):
    """
    seq2seq的encoder，主要就是使用Embedding和GRU对输入进行编码
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, dec_units: int, dropout: float):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=enc_units, bidirectional=True)
        self.fc = nn.Linear(enc_units * 2, dec_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) ->Tuple[torch.Tensor]:
        inputs = self.embedding(inputs)
        dropout = self.dropout(inputs)
        outputs, state = self.gru(dropout)
        state = torch.tanh(self.fc(torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)))
        return outputs, state


class BahdanauAttention(nn.Module):

    def __init__(self, units: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=2 * units, out_features=units)
        self.W2 = nn.Linear(in_features=units, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) ->Tuple[torch.Tensor]:
        values = values.permute(1, 0, 2)
        hidden_with_time_axis = torch.unsqueeze(query, 1)
        score = self.V(torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = F.softmax(score, 1)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights


class Decoder(nn.Module):
    """
    seq2seq的decoder，将初始化的inputs、隐藏层和encoder的输出作为
    输入，encoder的输入用来和隐藏层进行attention，得到的上下文
    向量和x进行整合然后丢到gru里去，最后Linear输出一下
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, dec_units: int, dropout: float, attention: nn.Module):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=enc_units * 2 + embedding_dim, hidden_size=dec_units)
        self.fc = nn.Linear(in_features=enc_units * 3 + embedding_dim, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, enc_output: torch.Tensor) ->Tuple[torch.Tensor]:
        inputs = inputs.unsqueeze(0)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        embedding = self.dropout(self.embedding(inputs))
        gru_inputs = torch.cat((embedding, torch.unsqueeze(context_vector, dim=0)), dim=-1)
        output, dec_state = self.gru(gru_inputs, hidden.unsqueeze(0))
        embedding = embedding.squeeze(0)
        output = output.squeeze(0)
        context_vector = context_vector
        output = self.fc(torch.cat((embedding, context_vector, output), dim=-1))
        return output, dec_state.squeeze(0)

