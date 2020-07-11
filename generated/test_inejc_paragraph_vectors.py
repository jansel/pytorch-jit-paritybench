import sys
_module = sys.modules[__name__]
del sys
paragraphvec = _module
data = _module
export_vectors = _module
loss = _module
models = _module
train = _module
utils = _module
setup = _module
tests = _module
test_data = _module
test_loss = _module
test_models = _module

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


import re


from math import ceil


import numpy as np


import torch


from numpy.random import choice


from torchtext.data import Field


from torchtext.data import TabularDataset


import torch.nn as nn


import time


from torch.optim import Adam


import matplotlib.pyplot as plt


class NegativeSampling(nn.Module):
    """Negative sampling loss as proposed by T. Mikolov et al. in Distributed
    Representations of Words and Phrases and their Compositionality.
    """

    def __init__(self):
        super(NegativeSampling, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        """Computes the value of the loss function.

        Parameters
        ----------
        scores: autograd.Variable of size (batch_size, num_noise_words + 1)
            Sparse unnormalized log probabilities. The first element in each
            row is the ground truth score (i.e. the target), other elements
            are scores of samples from the noise distribution.
        """
        k = scores.size()[1] - 1
        return -torch.sum(self._log_sigmoid(scores[:, (0)]) + torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k) / scores.size()[0]


class DM(nn.Module):
    """Distributed Memory version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """

    def __init__(self, vec_dim, num_docs, num_words):
        super(DM, self).__init__()
        self._D = nn.Parameter(torch.randn(num_docs, vec_dim), requires_grad=True)
        self._W = nn.Parameter(torch.randn(num_words, vec_dim), requires_grad=True)
        self._O = nn.Parameter(torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        context_ids: torch.Tensor of size (batch_size, num_context_words)
            Vocabulary indices of context words.

        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        x = torch.add(self._D[(doc_ids), :], torch.sum(self._W[(context_ids), :], dim=1))
        return torch.bmm(x.unsqueeze(1), self._O[:, (target_noise_ids)].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[(index), :].data.tolist()


class DBOW(nn.Module):
    """Distributed Bag of Words version of Paragraph Vectors.

    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_docs: int
        Number of documents in a dataset.

    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """

    def __init__(self, vec_dim, num_docs, num_words):
        super(DBOW, self).__init__()
        self._D = nn.Parameter(torch.randn(num_docs, vec_dim), requires_grad=True)
        self._O = nn.Parameter(torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, doc_ids, target_noise_ids):
        """Sparse computation of scores (unnormalized log probabilities)
        that should be passed to the negative sampling loss.

        Parameters
        ----------
        doc_ids: torch.Tensor of size (batch_size,)
            Document indices of paragraphs.

        target_noise_ids: torch.Tensor of size (batch_size, num_noise_words + 1)
            Vocabulary indices of target and noise words. The first element in
            each row is the ground truth index (i.e. the target), other
            elements are indices of samples from the noise distribution.

        Returns
        -------
            autograd.Variable of size (batch_size, num_noise_words + 1)
        """
        return torch.bmm(self._D[(doc_ids), :].unsqueeze(1), self._O[:, (target_noise_ids)].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self, index):
        return self._D[(index), :].data.tolist()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NegativeSampling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_inejc_paragraph_vectors(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

