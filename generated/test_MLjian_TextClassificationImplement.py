import sys
_module = sys.modules[__name__]
del sys
config = _module
data_process = _module
lstm_sum = _module
train = _module
train_word2vec = _module
pad = _module
models = _module
lstm_sum = _module
train = _module
train_cfg = _module
code = _module
lgb = _module
sklearn_config = _module
sklearn_train = _module
doc2vec = _module
ensemble = _module
ensemble_sparse = _module
feature_construct = _module
feature_select = _module
hash = _module
lda = _module
lsa = _module
nmf = _module
tf = _module
tfidf = _module
tree = _module

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


import torch.nn.functional as F


from torch.nn.utils import rnn


import pandas as pd


import time


import numpy as np


class LSTMsum(nn.Module):

    def __init__(self, emb_weights, emb_freeze, input_size, hidden_size, num_layers, l1_size, l2_size, num_classes, bidir, lstm_dropout):
        super(LSTMsum, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=emb_weights, freeze=emb_freeze)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidir, dropout=lstm_dropout)
        if bidir:
            self.l1 = nn.Linear(hidden_size * 2, l1_size)
        else:
            self.l1 = nn.Linear(hidden_size, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, num_classes)

    def forward(self, input):
        emb_out = self.embedding(input)
        lstm_out, _ = self.lstm(emb_out)
        sum_out = torch.sum(lstm_out, dim=1)
        l1_out = F.relu(self.l1(sum_out))
        l2_out = F.relu(self.l2(l1_out))
        l3_out = self.l3(l2_out)
        return l3_out

