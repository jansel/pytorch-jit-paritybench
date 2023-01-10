import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class RGCN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        self.conv1 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

