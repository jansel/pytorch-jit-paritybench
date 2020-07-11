import sys
_module = sys.modules[__name__]
del sys
dataHandler = _module
model = _module
sample = _module
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


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import random


import torch.optim as optim


import torch.autograd as autograd


class PoetryModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        length = input.size()[0]
        embeds = self.embeddings(input).view((length, 1, -1))
        output, hidden = self.lstm(embeds, hidden)
        output = F.relu(self.linear1(output.view(length, -1)))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, length=1):
        return Variable(torch.zeros(length, 1, self.hidden_dim)), Variable(torch.zeros(length, 1, self.hidden_dim))

