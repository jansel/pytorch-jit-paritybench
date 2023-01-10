import sys
_module = sys.modules[__name__]
del sys
GCL = _module
augmentors = _module
augmentor = _module
edge_adding = _module
edge_attr_masking = _module
edge_removing = _module
feature_dropout = _module
feature_masking = _module
functional = _module
identity = _module
markov_diffusion = _module
node_dropping = _module
node_shuffling = _module
ppr_diffusion = _module
rw_sampling = _module
eval = _module
eval = _module
logistic_regression = _module
random_forest = _module
svm = _module
losses = _module
barlow_twins = _module
bootstrap = _module
infonce = _module
jsd = _module
losses = _module
triplet = _module
vicreg = _module
models = _module
contrast_model = _module
samplers = _module
utils = _module
conf = _module
run_livereload = _module
BGRL_G2L = _module
BGRL_L2L = _module
DGI_inductive = _module
DGI_transductive = _module
GBT = _module
GRACE = _module
GRACE_SupCon = _module
GraphCL = _module
InfoGraph = _module
MVGRL_graph = _module
MVGRL_node = _module
setup = _module

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


from abc import ABC


from abc import abstractmethod


from typing import Optional


from typing import Tuple


from typing import NamedTuple


from typing import List


import torch.nn.functional as F


from torch.distributions import Uniform


from torch.distributions import Beta


from torch.distributions.bernoulli import Bernoulli


import numpy as np


from sklearn.metrics import f1_score


from sklearn.model_selection import PredefinedSplit


from sklearn.model_selection import GridSearchCV


from torch import nn


from torch.optim import Adam


from typing import *


import random


import copy


class LogisticRegression(nn.Module):

    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class HardMixingLoss(torch.nn.Module):

    def __init__(self, projection):
        super(HardMixingLoss, self).__init__()
        self.projection = projection

    @staticmethod
    def tensor_similarity(z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return torch.bmm(z2, z1.unsqueeze(dim=-1)).squeeze()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, threshold=0.1, s=150, mixup=0.2, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        num_samples = z1.shape[0]
        device = z1.device
        threshold = int(num_samples * threshold)
        refl1 = _similarity(z1, z1).diag()
        refl2 = _similarity(z2, z2).diag()
        pos_similarity = f(_similarity(z1, z2))
        neg_similarity1 = torch.cat([_similarity(z1, z1), _similarity(z1, z2)], dim=1)
        neg_similarity2 = torch.cat([_similarity(z2, z1), _similarity(z2, z2)], dim=1)
        neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
        neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
        neg_similarity1 = f(neg_similarity1)
        neg_similarity2 = f(neg_similarity2)
        z_pool = torch.cat([z1, z2], dim=0)
        hard_samples1 = z_pool[indices1[:, :threshold]]
        hard_samples2 = z_pool[indices2[:, :threshold]]
        hard_sample_idx1 = torch.randint(hard_samples1.shape[1], size=[num_samples, 2 * s])
        hard_sample_idx2 = torch.randint(hard_samples2.shape[1], size=[num_samples, 2 * s])
        hard_sample_draw1 = hard_samples1[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx1]
        hard_sample_draw2 = hard_samples2[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx2]
        hard_sample_mixing1 = mixup * hard_sample_draw1[:, :s, :] + (1 - mixup) * hard_sample_draw1[:, s:, :]
        hard_sample_mixing2 = mixup * hard_sample_draw2[:, :s, :] + (1 - mixup) * hard_sample_draw2[:, s:, :]
        h_m1 = self.projection(hard_sample_mixing1)
        h_m2 = self.projection(hard_sample_mixing2)
        neg_m1 = f(self.tensor_similarity(z1, h_m1)).sum(dim=1)
        neg_m2 = f(self.tensor_similarity(z2, h_m2)).sum(dim=1)
        pos = pos_similarity.diag()
        neg1 = neg_similarity1.sum(dim=1)
        neg2 = neg_similarity2.sum(dim=1)
        loss1 = -torch.log(pos / (neg1 + neg_m1 - refl1))
        loss2 = -torch.log(pos / (neg2 + neg_m2 - refl2))
        loss = (loss1 + loss2) * 0.5
        loss = loss.mean()
        return loss


class RingLoss(torch.nn.Module):

    def __init__(self):
        super(RingLoss, self).__init__()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, y: torch.Tensor, tau, threshold=0.1, *args, **kwargs):
        f = lambda x: torch.exp(x / tau)
        num_samples = h1.shape[0]
        device = h1.device
        threshold = int(num_samples * threshold)
        false_neg_mask = torch.zeros((num_samples, 2 * num_samples), dtype=torch.int)
        for i in range(num_samples):
            false_neg_mask[i] = (y == y[i]).repeat(2)
        pos_sim = f(_similarity(h1, h2))
        neg_sim1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)
        neg_sim2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
        neg_sim1, indices1 = torch.sort(neg_sim1, descending=True)
        neg_sim2, indices2 = torch.sort(neg_sim2, descending=True)
        y_repeated = y.repeat(2)
        false_neg_cnt = torch.zeros(num_samples)
        for i in range(num_samples):
            false_neg_cnt[i] = (y_repeated[indices1[i, threshold:-threshold]] == y[i]).sum()
        neg_sim1 = f(neg_sim1[:, threshold:-threshold])
        neg_sim2 = f(neg_sim2[:, threshold:-threshold])
        pos = pos_sim.diag()
        neg1 = neg_sim1.sum(dim=1)
        neg2 = neg_sim2.sum(dim=1)
        loss1 = -torch.log(pos / neg1)
        loss2 = -torch.log(pos / neg2)
        loss = (loss1 + loss2) * 0.5
        loss = loss.mean()
        return loss


class Loss(ABC):

    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) ->torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) ->torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1.0 - pos_mask
    return pos_mask, neg_mask


class Sampler(ABC):

    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

    @abstractmethod
    def sample(self, anchor, sample, *args, **kwargs):
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)
        return anchor, new_sample, new_pos_mask, new_neg_mask


class CrossScaleSampler(Sampler):

    def __init__(self, *args, **kwargs):
        super(CrossScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, batch=None, neg_sample=None, use_gpu=True, *args, **kwargs):
        num_graphs = anchor.shape[0]
        num_nodes = sample.shape[0]
        device = sample.device
        if neg_sample is not None:
            assert num_graphs == 1
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)
            sample = torch.cat([sample, neg_sample], dim=0)
        else:
            assert batch is not None
            if use_gpu:
                ones = torch.eye(num_nodes, dtype=torch.float32, device=device)
                pos_mask = scatter(ones, batch, dim=0, reduce='sum')
            else:
                pos_mask = torch.zeros((num_graphs, num_nodes), dtype=torch.float32)
                for node_idx, graph_idx in enumerate(batch):
                    pos_mask[graph_idx][node_idx] = 1.0
        neg_mask = 1.0 - pos_mask
        return anchor, sample, pos_mask, neg_mask


class SameScaleSampler(Sampler):

    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1.0 - pos_mask
        return anchor, sample, pos_mask, neg_mask


def get_sampler(mode: str, intraview_negs: bool) ->Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    elif mode == 'G2L':
        return CrossScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')


class SingleBranchContrast(torch.nn.Module):

    def __init__(self, loss: Loss, mode: str, intraview_negs: bool=False, **kwargs):
        super(SingleBranchContrast, self).__init__()
        assert mode == 'G2L'
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h, g, batch=None, hn=None, extra_pos_mask=None, extra_neg_mask=None):
        if batch is None:
            assert hn is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, neg_sample=hn)
        else:
            assert batch is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(anchor=g, sample=h, batch=batch)
        pos_mask, neg_mask = add_extra_mask(pos_mask, neg_mask, extra_pos_mask, extra_neg_mask)
        loss = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
        return loss


class DualBranchContrast(torch.nn.Module):

    def __init__(self, loss: Loss, mode: str, intraview_negs: bool=False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None, extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        elif batch is None or batch.max().item() + 1 <= 1:
            assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
        else:
            assert all(v is not None for v in [h1, h2, g1, g2, batch])
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)
        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)
        return (l1 + l2) * 0.5


class BootstrapContrast(torch.nn.Module):

    def __init__(self, loss, mode='L2L'):
        super(BootstrapContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=False)

    def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None, g1_pred=None, g2_pred=None, g1_target=None, g2_target=None, batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=h1_target, sample=h2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=h2_target, sample=h1_pred)
        elif self.mode == 'G2G':
            assert all(v is not None for v in [g1_pred, g2_pred, g1_target, g2_target])
            anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=g2_pred)
            anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=g1_pred)
        else:
            assert all(v is not None for v in [h1_pred, h2_pred, g1_target, g2_target])
            if batch is None or batch.max().item() + 1 <= 1:
                pos_mask1 = pos_mask2 = torch.ones([1, h1_pred.shape[0]], device=h1_pred.device)
                anchor1, sample1 = g1_target, h2_pred
                anchor2, sample2 = g2_target, h1_pred
            else:
                anchor1, sample1, pos_mask1, _ = self.sampler(anchor=g1_target, sample=h2_pred, batch=batch)
                anchor2, sample2, pos_mask2, _ = self.sampler(anchor=g2_target, sample=h1_pred, batch=batch)
        pos_mask1, _ = add_extra_mask(pos_mask1, extra_pos_mask=extra_pos_mask)
        pos_mask2, _ = add_extra_mask(pos_mask2, extra_pos_mask=extra_pos_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2)
        return (l1 + l2) * 0.5


class WithinEmbedContrast(torch.nn.Module):

    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        return (l1 + l2) * 0.5


class Normalize(torch.nn.Module):

    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):

    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


class FC(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU())
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FC,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogisticRegression,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_PyGCL_PyGCL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

