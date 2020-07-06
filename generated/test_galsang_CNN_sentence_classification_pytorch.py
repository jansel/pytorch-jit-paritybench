import sys
_module = sys.modules[__name__]
del sys
model = _module
run = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from sklearn.utils import shuffle


import numpy as np


import copy


class CNN(nn.Module):

    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        self.MODEL = kwargs['MODEL']
        self.BATCH_SIZE = kwargs['BATCH_SIZE']
        self.MAX_SENT_LEN = kwargs['MAX_SENT_LEN']
        self.WORD_DIM = kwargs['WORD_DIM']
        self.VOCAB_SIZE = kwargs['VOCAB_SIZE']
        self.CLASS_SIZE = kwargs['CLASS_SIZE']
        self.FILTERS = kwargs['FILTERS']
        self.FILTER_NUM = kwargs['FILTER_NUM']
        self.DROPOUT_PROB = kwargs['DROPOUT_PROB']
        self.IN_CHANNEL = 1
        assert len(self.FILTERS) == len(self.FILTER_NUM)
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == 'static' or self.MODEL == 'non-static' or self.MODEL == 'multichannel':
            self.WV_MATRIX = kwargs['WV_MATRIX']
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == 'static':
                self.embedding.weight.requires_grad = False
            elif self.MODEL == 'multichannel':
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2
        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == 'multichannel':
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)
        conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)
        return x

