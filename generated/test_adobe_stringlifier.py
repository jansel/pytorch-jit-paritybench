import sys
_module = sys.modules[__name__]
del sys
setup = _module
stringlifier = _module
api = _module
modules = _module
stringc = _module
stringc2 = _module
training = _module

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


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch.nn as nn


import numpy as np


import random


class AwDoC(nn.Module):

    def __init__(self, config, encodings):
        super(AwDoC, self).__init__()
        self._config = config
        self._encodings = encodings
        self._char_emb = nn.Embedding(len(encodings._char2int), config.char_emb_size)
        self._rnn = nn.LSTM(config.char_emb_size, config.rnn_size, config.rnn_layers, batch_first=True)
        self._hidden = nn.Sequential(nn.Linear(config.rnn_size, config.hidden), nn.Tanh(), nn.Dropout(0.5))
        self._softmax_type = nn.Linear(config.hidden, len(encodings._type2int))
        self._softmax_subtype = nn.Linear(config.hidden, len(encodings._subtype2int))

    def _make_input(self, domain_list):
        max_seq_len = max([len(domain) for domain in domain_list])
        x = np.zeros((len(domain_list), max_seq_len))
        for iBatch in range(x.shape[0]):
            domain = domain_list[iBatch]
            n = len(domain)
            ofs_x = max_seq_len - n
            for iSeq in range(x.shape[1]):
                if iSeq < n:
                    char = domain[-iSeq - 1].lower()
                    if char in self._encodings._char2int:
                        iChar = self._encodings._char2int[char]
                    else:
                        iChar = self._encodings._char2int['<UNK>']
                    x[iBatch, iSeq + ofs_x] = iChar
        return x

    def forward(self, domain_list):
        x = torch.tensor(self._make_input(domain_list), dtype=torch.long, device=self._get_device())
        hidden = self._char_emb(x)
        hidden = torch.dropout(hidden, 0.5, self.training)
        output, _ = self._rnn(hidden)
        output = output[:, -1, :]
        hidden = self._hidden(output)
        return self._softmax_type(hidden), self._softmax_subtype(hidden)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))


class CTagger(nn.Module):

    def __init__(self, config, encodings):
        super(CTagger, self).__init__()
        self._config = config
        self._encodings = encodings
        self._char_emb = nn.Embedding(len(encodings._char2int), config.char_emb_size, padding_idx=0)
        self._case_emb = nn.Embedding(4, 16, padding_idx=0)
        self._rnn = nn.LSTM(config.char_emb_size + 16, config.rnn_size, config.rnn_layers, batch_first=True, bidirectional=True)
        self._hidden = nn.Sequential(nn.Linear(config.rnn_size * 2, config.hidden), nn.Tanh(), nn.Dropout(0.5))
        self._softmax_type = nn.Linear(config.hidden, len(encodings._label2int))

    def _make_input(self, word_list):
        max_seq_len = max([len(word) for word in word_list])
        x_char = np.zeros((len(word_list), max_seq_len))
        x_case = np.zeros((len(word_list), max_seq_len))
        for iBatch in range(x_char.shape[0]):
            word = word_list[iBatch]
            for index in range(len(word)):
                char = word[index]
                case_idx = 0
                if char.lower() == char.upper():
                    case_idx = 1
                elif char.lower() != char:
                    case_idx = 2
                else:
                    case_idx = 3
                char = char.lower()
                if char in self._encodings._char2int:
                    char_idx = self._encodings._char2int[char]
                else:
                    char_idx = 1
                x_char[iBatch, index] = char_idx
                x_case[iBatch, index] = case_idx
        return x_char, x_case

    def forward(self, string_list):
        x_char, x_case = self._make_input(string_list)
        x_char = torch.tensor(x_char, dtype=torch.long, device=self._get_device())
        x_case = torch.tensor(x_case, dtype=torch.long, device=self._get_device())
        hidden = torch.cat([self._char_emb(x_char), self._case_emb(x_case)], dim=-1)
        hidden = torch.dropout(hidden, 0.5, self.training)
        output, _ = self._rnn(hidden)
        hidden = self._hidden(output)
        return self._softmax_type(hidden)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))

