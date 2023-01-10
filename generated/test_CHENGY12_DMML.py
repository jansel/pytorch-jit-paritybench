import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
common = _module
duke = _module
market1501 = _module
random_erasing = _module
sampler = _module
eval = _module
loss = _module
common = _module
contrastive = _module
dmml = _module
lifted = _module
npair = _module
triplet = _module
model = _module
train = _module
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


from torchvision import transforms


from torch.utils.data import dataloader


import numpy as np


import random


import torch.utils.data as Data


from torchvision.datasets.folder import default_loader


from torch.utils.data import sampler


from collections import defaultdict


import torch


from torch.nn import CrossEntropyLoss


import torch.nn as nn


import numbers


from torch.nn import functional as F


from torchvision import models


import torch.optim as optim


from torch.nn import DataParallel


import time


def euclidean_dist(x, y, squared=True):
    """
    Compute (Squared) Euclidean distance between two tensors.

    Args:
        x: input tensor with size N x D.
        y: input tensor with size M x D.

        return: distance matrix with size N x M.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    if squared:
        return dist
    else:
        return torch.sqrt(dist + 1e-12)


def get_mask(label, mask_type='positive'):
    """
    Generate positive and negative masks for contrastive and triplet loss.
    """
    device = label.device
    identity = torch.eye(label.shape[0]).byte()
    not_identity = ~identity
    not_identity = not_identity
    if mask_type == 'positive':
        mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    elif mask_type == 'negative':
        mask = torch.ne(label.unsqueeze(1), label.unsqueeze(0))
    mask = mask.byte()
    mask = mask & not_identity
    return mask


class ContrastiveLoss(nn.Module):
    """
    Batch hard contrastive loss.
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for contrastive loss.')
        self.margin = margin

    def forward(self, feature, label):
        distance = euclidean_dist(feature, feature, squared=False)
        positive_mask = get_mask(label, 'positive')
        hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]
        p_loss = hardest_positive.mean()
        negative_mask = get_mask(label, 'negative')
        max_distance = distance.max(dim=1)[0]
        not_negative_mask = ~negative_mask
        negative_distance = distance + max_distance * not_negative_mask.float()
        hardest_negative = negative_distance.min(dim=1)[0]
        n_loss = (self.margin - hardest_negative).clamp(min=0).mean()
        con_loss = p_loss + n_loss
        return con_loss


def cosine_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = -torch.mul(x, y).sum(2)
    return dist


class DMMLLoss(nn.Module):
    """
    DMML loss with center support distance and hard mining distance.

    Args:
        num_support: the number of support samples per class.
        distance_mode: 'center_support' or 'hard_mining'.
    """

    def __init__(self, num_support, distance_mode='hard_mining', margin=0.4, gid=None):
        super().__init__()
        if not distance_mode in ['center_support', 'hard_mining']:
            raise Exception('Invalid distance mode for DMML loss.')
        if not isinstance(margin, numbers.Real):
            raise Exception('Invalid margin parameter for DMML loss.')
        self.num_support = num_support
        self.distance_mode = distance_mode
        self.margin = margin
        self.gid = gid

    def forward(self, feature, label):
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)
        if self.gid is not None:
            feature, label, classes = feature, label, classes
        num_classes = len(classes)
        num_query = label.eq(classes[0]).sum() - self.num_support
        support_inds_list = list(map(lambda c: label.eq(c).nonzero()[:self.num_support].squeeze(1), classes))
        query_inds = torch.stack(list(map(lambda c: label.eq(c).nonzero()[self.num_support:], classes))).view(-1)
        query_samples = feature[query_inds]
        if self.distance_mode == 'center_support':
            center_points = torch.stack([torch.mean(feature[support_inds], dim=0) for support_inds in support_inds_list])
            dists = euclidean_dist(query_samples, center_points)
        elif self.distance_mode == 'hard_mining':
            dists = []
            max_self_dists = []
            for i, support_inds in enumerate(support_inds_list):
                dist_all = cosine_dist(query_samples, feature[support_inds])
                max_dist, _ = torch.max(dist_all[i * num_query:(i + 1) * num_query], dim=1)
                min_dist, _ = torch.min(dist_all, dim=1)
                dists.append(min_dist)
                max_self_dists.append(max_dist)
            dists = torch.stack(dists).t()
            for i in range(num_classes):
                dists[i * num_query:(i + 1) * num_query, i] = max_self_dists[i]
        log_prob = F.log_softmax(-dists, dim=1).view(num_classes, num_query, -1)
        target_inds = torch.arange(0, num_classes)
        if self.gid is not None:
            target_inds = target_inds
        target_inds = target_inds.view(num_classes, 1, 1).expand(num_classes, num_query, 1).long()
        dmml_loss = -log_prob.gather(2, target_inds).squeeze().view(-1).mean()
        batch_size = feature.size(0)
        l2_loss = torch.sum(feature ** 2) / batch_size
        dmml_loss += 0.002 * 0.25 * l2_loss
        return dmml_loss


class LiftedLoss(nn.Module):
    """
    Lifted loss.
    """

    def __init__(self, margin=0.4, gid=None):
        super(LiftedLoss, self).__init__()
        self.margin = margin
        self.gid = gid

    def forward(self, features, labels):
        batch_size = labels.size(0)
        positive_mask = labels.view(1, -1) == labels.view(-1, 1)
        negative_mask = ~positive_mask
        dists = euclidean_dist(features, features, squared=False)
        dists_repeated_x = dists.repeat(1, batch_size).view(-1, batch_size)
        negative_mask_repeated_x = negative_mask.repeat(1, batch_size).view(-1, batch_size).float()
        dists_repeated_y = dists.transpose(1, 0).repeat(batch_size, 1)
        negative_mask_repeated_y = negative_mask.transpose(1, 0).repeat(batch_size, 1).float()
        positive_dists = dists.view(-1, 1)
        J_matrix = torch.log(torch.sum(torch.exp(self.margin - dists_repeated_x) * negative_mask_repeated_x, 1, keepdim=True) + torch.sum(torch.exp(self.margin - dists_repeated_y) * negative_mask_repeated_y, 1, keepdim=True)) + positive_dists
        J_matrix_valid = torch.masked_select(J_matrix, positive_mask.view(-1, 1))
        if self.gid is not None:
            J_matrix_valid = torch.max(torch.zeros(J_matrix_valid.size()), J_matrix_valid)
        else:
            J_matrix_valid = torch.max(torch.zeros(J_matrix_valid.size()), J_matrix_valid)
        lifted_loss_matrix = J_matrix_valid * J_matrix_valid
        lifted_loss = torch.sum(lifted_loss_matrix) / (2 * positive_mask.sum().item())
        return lifted_loss


class NpairLoss(nn.Module):
    """
    Multi-class N-pair loss.

    Args:
        reg_lambda: L2 norm regularization for embedding vectors.
    """

    def __init__(self, reg_lambda=0.002, gid=None):
        super(NpairLoss, self).__init__()
        self.reg_lambda = reg_lambda
        self.gid = gid

    def forward(self, feature, label):
        feature = feature.cpu()
        label = label.cpu()
        classes = torch.unique(label)
        if self.gid is not None:
            feature, label, classes = feature, label, classes
        anchor_inds = torch.stack(list(map(lambda c: label.eq(c).nonzero()[0].squeeze(0), classes)))
        positive_inds = torch.stack(list(map(lambda c: label.eq(c).nonzero()[1].squeeze(0), classes)))
        anchor = feature[anchor_inds]
        positive = feature[positive_inds]
        batch_size = anchor.size(0)
        classes = classes.view(classes.size(0), 1)
        target = (classes == classes.t()).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()
        similarity = torch.matmul(anchor, positive.t())
        ce_loss = torch.mean(torch.sum(-target * F.log_softmax(similarity, -1), -1))
        l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size
        npair_loss = ce_loss + l2_loss * self.reg_lambda * 0.25
        return npair_loss


class TripletLoss(nn.Module):
    """
    Batch hard triplet loss.
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        if not (isinstance(margin, numbers.Real) or margin == 'soft'):
            raise Exception('Invalid margin parameter for triplet loss.')
        self.margin = margin

    def forward(self, feature, label):
        distance = euclidean_dist(feature, feature, squared=False)
        positive_mask = get_mask(label, 'positive')
        hardest_positive = (distance * positive_mask.float()).max(dim=1)[0]
        negative_mask = get_mask(label, 'negative')
        max_distance = distance.max(dim=1)[0]
        not_negative_mask = ~negative_mask.data
        negative_distance = distance + max_distance * not_negative_mask.float()
        hardest_negative = negative_distance.min(dim=1)[0]
        diff = hardest_positive - hardest_negative
        if isinstance(self.margin, numbers.Real):
            tri_loss = (self.margin + diff).clamp(min=0).mean()
        else:
            tri_loss = F.softplus(diff).mean()
        return tri_loss


class resnet_model(nn.Module):

    def __init__(self, num_classes=None, include_top=False, remove_downsample=False):
        super(resnet_model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.include_top = include_top
        if remove_downsample:
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].conv2.stride = 1
        if self.include_top:
            self.fc = nn.Linear(2048, num_classes)
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        if not self.include_top:
            return feat
        else:
            logits = self.fc(feat)
            return feat, logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ContrastiveLoss,
     lambda: ([], {'margin': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TripletLoss,
     lambda: ([], {'margin': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (resnet_model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_CHENGY12_DMML(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

