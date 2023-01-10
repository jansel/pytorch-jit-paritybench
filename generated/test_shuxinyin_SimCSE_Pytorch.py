import sys
_module = sys.modules[__name__]
del sys
ESimCSE_Model = _module
ESimCSE_dataloader = _module
ESimCSE_train = _module
ESimCSE = _module
SimCSE = _module
dataloader = _module
model = _module
train = _module
master = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


from scipy.stats import spearmanr


class ESimcseModel(nn.Module):

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(ESimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]
        if self.pooling == 'pooler':
            return out.pooler_output
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)


class MomentumEncoder(ESimcseModel):
    """ MomentumEncoder """

    def __init__(self, pretrained_model, pooling):
        super(MomentumEncoder, self).__init__(pretrained_model, pooling)


class MultiNegativeRankingLoss(nn.Module):

    def __init__(self):
        super(MultiNegativeRankingLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def multi_negative_ranking_loss(self, embed_src, embed_pos, embed_neg, scale=20.0):
        """
        scale is a temperature parameter
        """
        if embed_neg is not None:
            embed_pos = torch.cat([embed_pos, embed_neg], dim=0)
        scores = self.cos_sim(embed_src, embed_pos) * scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)

    def cos_sim(self, a, b):
        """ the function is same with torch.nn.F.cosine_similarity but processed the problem of tensor dimension
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if len(b.shape) == 1:
            b = b.unsqueeze(0)
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout=0.3):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]
        if self.pooling == 'pooler':
            return out.pooler_output
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

