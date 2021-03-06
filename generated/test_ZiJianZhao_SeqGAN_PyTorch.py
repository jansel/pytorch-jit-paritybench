import sys
_module = sys.modules[__name__]
del sys
data_iter = _module
discriminator = _module
generator = _module
loss = _module
main = _module
rollout = _module
target_lstm = _module

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


import random


import math


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


import copy


class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, emb_dim)) for n, f in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1.0 - torch.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class Generator(nn.Module):
    """Generator """

    def __init__(self, num_emb, emb_dim, hidden_dim, use_cuda):
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()
        self.init_params()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        emb = self.emb(x)
        h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(emb, (h0, c0))
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))
        return pred

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=1)
        return pred, h, c

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h, c
        return h, c

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(self, batch_size, seq_len, x=None):
        res = []
        flag = False
        if x is None:
            flag = True
        if flag:
            x = Variable(torch.zeros((batch_size, 1)).long())
        if self.use_cuda:
            x = x
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            for i in range(seq_len):
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output


class NLLLoss(nn.Module):
    """Self-Defined NLLLoss Function

    Args:
        weight: Tensor (num_class, )
    """

    def __init__(self, weight):
        super(NLLLoss, self).__init__()
        self.weight = weight

    def forward(self, prob, target):
        """
        Args:
            prob: (N, C) 
            target : (N, )
        """
        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1))
        weight = weight.expand(N, C)
        if prob.is_cuda:
            weight = weight
        prob = weight * prob
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot
        loss = torch.masked_select(prob, one_hot)
        return -torch.sum(loss)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss


class TargetLSTM(nn.Module):
    """Target Lstm """

    def __init__(self, num_emb, emb_dim, hidden_dim, use_cuda):
        super(TargetLSTM, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()
        self.init_params()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        emb = self.emb(x)
        h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(emb, (h0, c0))
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))
        return pred

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=1)
        return pred, h, c

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h, c
        return h, c

    def init_params(self):
        for param in self.parameters():
            param.data.normal_(0, 1)

    def sample(self, batch_size, seq_len):
        res = []
        with torch.no_grad():
            x = Variable(torch.zeros((batch_size, 1)).long())
            if self.use_cuda:
                x = x
            h, c = self.init_hidden(batch_size)
            samples = []
            for i in range(seq_len):
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                samples.append(x)
            output = torch.cat(samples, dim=1)
            return output
        return None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GANLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64), torch.rand([4, 4, 4, 64])], {}),
     False),
    (Generator,
     lambda: ([], {'num_emb': 4, 'emb_dim': 4, 'hidden_dim': 4, 'use_cuda': False}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64)], {}),
     False),
    (TargetLSTM,
     lambda: ([], {'num_emb': 4, 'emb_dim': 4, 'hidden_dim': 4, 'use_cuda': False}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64)], {}),
     False),
]

class Test_ZiJianZhao_SeqGAN_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

