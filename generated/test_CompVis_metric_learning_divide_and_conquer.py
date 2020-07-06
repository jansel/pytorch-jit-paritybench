import sys
_module = sys.modules[__name__]
del sys
browse_results = _module
experiment = _module
lib = _module
clustering = _module
data = _module
loader = _module
reassignment = _module
sampler = _module
utils = _module
set = _module
base = _module
inshop = _module
sop = _module
transform = _module
vid = _module
evaluation = _module
normalized_mutual_information = _module
recall = _module
faissext = _module
loss = _module
margin_loss = _module
sampler = _module
model = _module
similarity = _module
utils = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import torch


import logging


import numpy as np


import sklearn.cluster


from scipy.optimize import linear_sum_assignment


from torch.utils.data.sampler import BatchSampler


import random


import torchvision


from torchvision import transforms


import sklearn.metrics.cluster


import sklearn.metrics.pairwise


import torch.nn.functional as F


import torch.nn as nn


from math import ceil


from torch.nn import Linear


from torch.nn import Dropout


from torch.nn import AvgPool2d


from torch.nn import MaxPool2d


from torch.nn.init import xavier_normal_


import sklearn


import time


import collections


import warnings


def pdist(A, squared=False, eps=0.0001):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.zeros_like(input.data).scatter(dim, index, 1.0)


class Sampler(nn.Module):
    """
    Sample for each anchor negative examples
        are K closest points on the distance >= cutoff

    Inputs:
        - **data**: input tensor with shape (batch_size, embed_dim).
        Here we assume the consecutive batch_k examples are of the same class.
        For example, if batch_k = 5, the first 5 examples belong to the same class,
        6th-10th examples belong to another class, etc.

    Outputs:
        - a_indices: indices of anchors.
        - x[a_indices]: sampled anchor embeddings.
        - x[p_indices]: sampled positive embeddings.
        - x[n_indices]: sampled negative embeddings.
        - x: embeddings of the input batch.
    """

    def __init__(self, cutoff=0.5, infinity=1000000.0, eps=1e-06):
        super(Sampler, self).__init__()
        self.cutoff = cutoff
        self.infinity = infinity
        self.eps = eps

    def forward(self, x, labels):
        """
        x: input tensor of shape (batch_size, embed_dim)
        labels: tensor of class labels of shape (batch_size,)
        """
        d = pdist(x)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.eye(len(d)).type_as(d)
        num_neg = int(pos.data.sum()) // len(pos)
        neg = topk_mask(d + self.infinity * ((pos > 0) + (d < self.cutoff)).type_as(d), dim=1, largest=False, K=num_neg)
        a_indices = []
        p_indices = []
        n_indices = []
        for i in range(len(d)):
            a_indices.extend([i] * num_neg)
            p_indices.extend(np.atleast_1d(pos[i].nonzero().squeeze().cpu().numpy()))
            n_indices.extend(np.atleast_1d(neg[i].nonzero().squeeze().cpu().numpy()))
            if len(a_indices) != len(p_indices) or len(a_indices) != len(n_indices):
                logging.warning('Probably too many positives, because of lacking classes in' + ' the current cluster.' + ' n_anchors={}, n_pos={}, n_neg= {}'.format(*map(len, [a_indices, p_indices, n_indices])))
                min_len = min(map(len, [a_indices, p_indices, n_indices]))
                a_indices = a_indices[:min_len]
                p_indices = p_indices[:min_len]
                n_indices = n_indices[:min_len]
        assert len(a_indices) == len(p_indices) == len(n_indices), '{}, {}, {}'.format(*map(len, [a_indices, p_indices, n_indices]))
        return a_indices, x[a_indices], x[p_indices], x[n_indices]


class MarginLoss(torch.nn.Module):
    """Margin based loss.

    Parameters
    ----------
    nb_classes: int
        Number of classes in the train dataset.
        Used to initialize class-specific boundaries beta.
    margin : float
        Margin between positive and negative pairs.
    nu : float
        Regularization parameter for beta.
    class_specific_beta : bool
        Are class-specific boundaries beind used?

    Inputs:
        - anchors: sampled anchor embeddings.
        - positives: sampled positive embeddings.
        - negatives: sampled negative embeddings.
        - anchor_classes: labels of anchors. Used to get class-specific beta.

    Outputs:
        Loss value.
    """

    def __init__(self, nb_classes, beta=1.2, margin=0.2, nu=0.0, class_specific_beta=False, **kwargs):
        super(MarginLoss, self).__init__()
        self.nb_classes = nb_classes
        self.class_specific_beta = class_specific_beta
        if class_specific_beta:
            assert nb_classes is not None
            beta = torch.ones(nb_classes, dtype=torch.float32) * beta
        else:
            beta = torch.tensor([beta], dtype=torch.float32)
        self.beta = torch.nn.Parameter(beta)
        self.margin = margin
        self.nu = nu
        self.sampler = Sampler()

    def forward(self, E, T):
        anchor_idx, anchors, positives, negatives = self.sampler(E, T)
        anchor_classes = T[anchor_idx]
        if anchor_classes is not None:
            if self.class_specific_beta:
                beta = self.beta[anchor_classes]
            else:
                beta = self.beta
            beta_regularization_loss = torch.norm(beta, p=1) * self.nu
        else:
            beta = self.beta
            beta_regularization_loss = 0.0
        try:
            d_ap = ((positives - anchors) ** 2).sum(dim=1) + 1e-08
        except Exception as e:
            None
            None
            raise e
        d_ap = torch.sqrt(d_ap)
        d_an = ((negatives - anchors) ** 2).sum(dim=1) + 1e-08
        d_an = torch.sqrt(d_an)
        pos_loss = F.relu(d_ap - beta + self.margin)
        neg_loss = F.relu(beta - d_an + self.margin)
        pair_cnt = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).type_as(pos_loss)
        loss = torch.sum(pos_loss + neg_loss)
        if pair_cnt > 0.0:
            loss = (loss + beta_regularization_loss) / pair_cnt
        return loss

