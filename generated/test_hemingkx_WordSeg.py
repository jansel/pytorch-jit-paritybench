import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
data_process = _module
metrics = _module
model = _module
run = _module
train = _module
utils = _module
config = _module
data_loader = _module
model = _module
run = _module
train = _module
config = _module
data_loader = _module
model = _module
run = _module
train = _module
Vocabulary = _module
data_loader = _module
embedding = _module
metric = _module
model = _module
run = _module
train = _module
data_loader = _module
embedding = _module
model = _module
run = _module
train = _module
data_loader = _module
model = _module
run = _module
train = _module

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


from torch.nn.utils.rnn import pad_sequence


import logging


from sklearn.model_selection import train_test_split


from torch.utils.data import DataLoader


from torch.nn.parallel import DistributedDataParallel


from torch.utils.data.distributed import DistributedSampler


import warnings


import torch.nn as nn


from itertools import chain


from itertools import tee


from typing import Any


from typing import Sequence


from typing import Iterable


from torch import optim


from torch.optim.lr_scheduler import StepLR


from sklearn.model_selection import KFold


class BiLSTM_CRF(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, target_size, num_layers, lstm_drop_out, nn_drop_out):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.nn_drop_out = nn_drop_out
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=lstm_drop_out if num_layers > 1 else 0, bidirectional=True)
        if nn_drop_out > 0:
            self.dropout = nn.Dropout(nn_drop_out)
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        self.crf = CRF(target_size, batch_first=True)

    def forward(self, unigrams, training=True):
        uni_embeddings = self.embedding(unigrams)
        sequence_output, _ = self.bilstm(uni_embeddings)
        if training and self.nn_drop_out > 0:
            sequence_output = self.dropout(sequence_output)
        tag_scores = self.classifier(sequence_output)
        return tag_scores

    def forward_with_crf(self, unigrams, input_mask, input_tags):
        """
        函数功能：
            1. 使用BiLSTM模型计算每个字对应的4个标签的概率 self.forware
            2. 使用crf算法计算损失值 self.crf
        """
        tag_scores = self.forward(unigrams)
        loss = self.crf(tag_scores, input_tags, input_mask) * -1
        return tag_scores, loss

