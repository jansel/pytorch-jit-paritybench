import sys
_module = sys.modules[__name__]
del sys
adam_base = _module
fast_gbw = _module
gbw = _module
learning_rate = _module
setup = _module
test = _module
main = _module
model = _module
process_gbw = _module
sparse_model = _module
stream_gbw = _module
fast_gbw_utest = _module
gbw_utest = _module
log_uniform_sampler_utest = _module
sampled_softmax_utest = _module
util = _module

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


import math


import torch


from torch.optim import Optimizer


import random


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import _LRScheduler


import time


import torch.nn as nn


import torch.optim as optim


from torch.autograd import Variable


import torch.nn.functional as F


class SampledSoftmax(nn.Module):

    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()
        self.ntokens = ntokens
        self.nsampled = nsampled
        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)
        """
        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            util.initialize(self.params.weight)
        """

    def forward(self, inputs, labels):
        if self.training:
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):
        assert inputs.data.get_device() == labels.data.get_device()
        device_id = labels.data.get_device()
        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values
        sample_ids = Variable(torch.LongTensor(sample_ids))
        true_freq = Variable(torch.FloatTensor(true_freq))
        sample_freq = Variable(torch.FloatTensor(sample_freq))
        true_weights = F.embedding(labels, self.params.weight, sparse=True)
        true_bias = torch.index_select(self.params.bias, 0, labels)
        sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
        sample_bias = torch.index_select(self.params.bias, 0, sample_ids)
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = sample_weights.matmul(inputs.t()).t() + sample_bias
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e+37
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long())
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels


class RNNModel(nn.Module):
    """A recurrent module"""

    def __init__(self, ntokens, ninp, nhid, nout, nlayers, proj, dropout):
        super(RNNModel, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        if proj:
            self.proj = nn.Linear(nhid, nout)
            util.initialize(self.proj.weight)
        else:
            self.proj = None

    def forward(self, inputs, hidden):
        inputs = self.drop(inputs)
        output, hidden = self.rnn(inputs, hidden)
        if self.proj is not None:
            output = self.proj(output)
        output = self.drop(output)
        return output.view(output.size(0) * output.size(1), output.size(2)), hidden

    def init_hidden(self, bsz):
        return Variable(torch.zeros(self.nlayers, bsz, self.nhid)), Variable(torch.zeros(self.nlayers, bsz, self.nhid))

