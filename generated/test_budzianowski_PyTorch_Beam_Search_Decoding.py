import sys
_module = sys.modules[__name__]
del sys
decode_beam = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


class DecoderRNN(nn.Module):

    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        """
        Illustrative decoder
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True, dropout=dropout, batch_first=False)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)
        embedded = F.dropout(embedded, self.dropout_rate)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)
        return output, hidden

