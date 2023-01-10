import sys
_module = sys.modules[__name__]
del sys
an4 = _module
common_voice = _module
librispeech = _module
merge_manifests = _module
ted = _module
verify_manifest = _module
voxforge = _module
deepspeech_pytorch = _module
checkpoint = _module
configs = _module
inference_config = _module
lightning_config = _module
train_config = _module
data = _module
data_opts = _module
utils = _module
decoder = _module
enums = _module
inference = _module
loader = _module
data_loader = _module
data_module = _module
sparse_image_warp = _module
spec_augment = _module
model = _module
testing = _module
training = _module
utils = _module
validation = _module
noise_inject = _module
search_lm_params = _module
select_lm_params = _module
server = _module
setup = _module
test = _module
tests = _module
pretrained_smoke_test = _module
smoke_test = _module
train = _module
transcribe = _module

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


from enum import Enum


from torch import nn


from typing import List


from torch.cuda.amp import autocast


import math


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import Sampler


from torch.utils.data import DistributedSampler


from torch.utils.data import DataLoader


import torchaudio


import random


import matplotlib


import matplotlib.pyplot as plt


from typing import Union


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import CTCLoss


from abc import ABC


from abc import abstractmethod


from scipy.io.wavfile import write


import logging


class SequenceWise(nn.Module):

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):

    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask
            for i, length in enumerate(lengths):
                length = length.item()
                if mask[i].size(2) - length > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):

    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, h=None):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x, h)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x, h


class Lookahead(nn.Module):

    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = 0, self.context - 1
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1, groups=self.n_features, padding=0, bias=False)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n_features=' + str(self.n_features) + ', context=' + str(self.context) + ')'


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (InferenceBatchSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lookahead,
     lambda: ([], {'n_features': 4, 'context': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (SequenceWise,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_SeanNaren_deepspeech_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

