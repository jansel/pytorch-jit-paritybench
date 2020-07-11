import sys
_module = sys.modules[__name__]
del sys
cdssm = _module

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


import torchvision.datasets as dsets


import torchvision.transforms as transforms


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


FILTER_LENGTH = 1


J = 4


K = 300


L = 128


TOTAL_LETTER_GRAMS = int(3 * 10000.0)


WINDOW_SIZE = 3


WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class CDSSM(nn.Module):

    def __init__(self):
        super(CDSSM, self).__init__()
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.doc_sem = nn.Linear(K, L)
        self.learn_gamma = nn.Conv1d(1, 1, 1)

    def forward(self, q, pos, negs):
        q = q.transpose(1, 2)
        q_c = F.tanh(self.query_conv(q))
        q_k = kmax_pooling(q_c, 2, 1)
        q_k = q_k.transpose(1, 2)
        q_s = F.tanh(self.query_sem(q_k))
        q_s = q_s.resize(L)
        pos = pos.transpose(1, 2)
        pos_c = F.tanh(self.doc_conv(pos))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1, 2)
        pos_s = F.tanh(self.doc_sem(pos_k))
        pos_s = pos_s.resize(L)
        negs = [neg.transpose(1, 2) for neg in negs]
        neg_cs = [F.tanh(self.doc_conv(neg)) for neg in negs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1, 2) for neg_k in neg_ks]
        neg_ss = [F.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [neg_s.resize(L) for neg_s in neg_ss]
        dots = [q_s.dot(pos_s)]
        dots = dots + [q_s.dot(neg_s) for neg_s in neg_ss]
        dots = torch.stack(dots)
        with_gamma = self.learn_gamma(dots.resize(J + 1, 1, 1))
        return with_gamma

