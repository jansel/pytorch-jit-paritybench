import sys
_module = sys.modules[__name__]
del sys
baseline = _module
aro = _module
chinese_whispers = _module
sklearn_cluster = _module
dsgcn = _module
cfg_test_det_fashion_20_prpsls = _module
cfg_test_det_fashion_2_prpsls = _module
cfg_test_det_fashion_8_prpsls = _module
cfg_test_det_ms1m_20_prpsls = _module
cfg_test_det_ms1m_2_prpsls = _module
cfg_test_det_ms1m_5_prpsls = _module
cfg_test_det_ms1m_8_prpsls = _module
cfg_test_det_ytb_4_prpsls = _module
cfg_test_seg_ms1m_20_prpsls = _module
cfg_test_seg_ms1m_2_prpsls = _module
cfg_test_seg_ms1m_5_prpsls = _module
cfg_test_seg_ms1m_8_prpsls = _module
cfg_train_det_fashion_4_prpsls = _module
cfg_train_det_fashion_84_prpsls = _module
cfg_train_det_fashion_8_prpsls = _module
cfg_train_det_ms1m_4_prpsls = _module
cfg_train_det_ms1m_84_prpsls = _module
cfg_train_det_ms1m_8_prpsls = _module
cfg_train_seg_ms1m_4_prpsls = _module
cfg_train_seg_ms1m_84_prpsls = _module
cfg_train_seg_ms1m_8_prpsls = _module
datasets = _module
build_dataloader = _module
cluster_dataset = _module
cluster_det_processor = _module
cluster_processor = _module
cluster_seg_processor = _module
sampler = _module
main = _module
models = _module
dsgcn = _module
runner = _module
test_cluster_det = _module
test_cluster_seg = _module
train = _module
train_cluster_det = _module
train_cluster_seg = _module
evaluation = _module
evaluate = _module
metrics = _module
lgcn = _module
cfg_test_lgcn_fashion = _module
cfg_test_lgcn_ms1m = _module
cfg_train_lgcn_fashion = _module
cfg_train_lgcn_ms1m = _module
build_dataloader = _module
main = _module
lgcn = _module
online_evaluation = _module
test_lgcn = _module
train_lgcn = _module
post_process = _module
deoverlap = _module
nms = _module
proposals = _module
generate_basic_proposals = _module
generate_iter_proposals = _module
generate_proposals = _module
graph = _module
stat_cluster = _module
super_vertex = _module
analyze_proposals = _module
baseline_cluster = _module
download_data = _module
dsgcn_upper_bound = _module
test_knn = _module
utils = _module
adjacency = _module
dataset = _module
dist = _module
draw = _module
faiss_gpu = _module
faiss_search = _module
knn = _module
logger = _module
misc = _module
misc_cluster = _module
vegcn = _module
confidence = _module
cfg_test_gcne_fashion = _module
cfg_test_gcne_ms1m = _module
cfg_test_gcnv_fashion = _module
cfg_test_gcnv_ms1m = _module
cfg_train_gcne_fashion = _module
cfg_train_gcne_ms1m = _module
cfg_train_gcnv_fashion = _module
cfg_train_gcnv_ms1m = _module
gcn_e_dataset = _module
gcn_v_dataset = _module
deduce = _module
extract = _module
main = _module
gcn_e = _module
gcn_v = _module
utils = _module
test_gcn_e = _module
test_gcn_v = _module
train_gcn_e = _module
train_gcn_v = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn.functional as F


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


import math


from torch.utils.data.sampler import Sampler


from torch.utils.data.distributed import DistributedSampler as _DistributedSampler


import torch.nn as nn


from torch.nn.parameter import Parameter


from collections import OrderedDict


from torch.nn import init


import scipy.sparse as sp


import torch.distributed as dist


import torch.multiprocessing as mp


import time


import random


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, agg, dropout=0):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()
        self.dropout = dropout

    def forward(self, features, A):
        feat_dim = features.shape[-1]
        assert feat_dim == self.in_dim
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=-1)
        if features.dim() == 2:
            op = 'nd,df->nf'
        elif features.dim() == 3:
            op = 'bnd,df->bnf'
        else:
            raise RuntimeError('the dimension of features should be 2 or 3')
        out = torch.einsum(op, (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training=self.training)
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.gc = GraphConv(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, adj, D=None):
        x = self.gc(x, adj, D)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class GNN(nn.Module):
    """ input is (bs, N, D), for featureless, D=1
        dev output is (bs, num_classes)
        seg output is (bs, N, num_classes)
    """

    def __init__(self, planes, feature_dim, featureless, num_classes=1, dropout=0.0, reduce_method='max', stage='det', use_random_seed=False, **kwargs):
        assert feature_dim > 0
        assert dropout >= 0 and dropout < 1
        super(GNN, self).__init__()
        if featureless:
            self.inplanes = 1
        else:
            self.inplanes = feature_dim
        self.num_classes = num_classes
        self.reduce_method = reduce_method
        self.stage = stage
        if self.stage == 'det':
            self.loss = torch.nn.MSELoss()
        elif self.stage == 'seg':
            self.num_classes = 2
            if use_random_seed:
                self.inplanes += 1
            self.loss = torch.nn.NLLLoss(ignore_index=-100)
        else:
            raise KeyError('Unknown stage: {}'.format(stage))

    def pool(self, x):
        if self.reduce_method == 'sum':
            return torch.sum(x, dim=1)
        elif self.reduce_method == 'mean':
            return torch.mean(x, dim=1)
        elif self.reduce_method == 'max':
            return torch.max(x, dim=1)[0]
        elif self.reduce_method == 'no_pool':
            return x
        else:
            raise KeyError('Unkown reduce method', self.reduce_method)

    def forward(self, data, return_loss=False):
        x = self.extract(data[0], data[1])
        if return_loss:
            label = data[2]
            if self.stage == 'det':
                loss = self.loss(x.view(-1), label)
            elif self.stage == 'seg':
                loss = self.loss(x, label)
            return x, loss
        else:
            return x


class GCN(GNN):

    def __init__(self, planes, feature_dim, featureless, num_classes=1, dropout=0.0, reduce_method='max', stage='det', **kwargs):
        super().__init__(planes, feature_dim, featureless, num_classes, dropout, reduce_method, stage, **kwargs)
        self.layers = self._make_layer(BasicBlock, planes, dropout)
        self.classifier = nn.Linear(self.inplanes, self.num_classes)

    def _make_layer(self, block, planes, dropout=0.0):
        layers = nn.ModuleList([])
        for i, plane in enumerate(planes):
            layers.append(block(self.inplanes, plane, dropout))
            self.inplanes = plane
        return layers

    def extract(self, x, adj):
        bs = x.size(0)
        adj.detach_()
        D = adj.sum(dim=2, keepdim=True)
        D.detach_()
        assert (D > 0).all(), 'D should larger than 0, otherwise gradient will be NaN.'
        for layer in self.layers:
            x = layer(x, adj, D)
        x = self.pool(x)
        x = x.view(-1, self.inplanes)
        x = self.classifier(x)
        if self.reduce_method == 'no_pool':
            if self.num_classes > 1:
                x = x.view(bs, -1, self.num_classes)
                x = torch.transpose(x, 1, 2).contiguous()
                x = F.log_softmax(x, dim=1)
            else:
                x = x.view(bs, -1)
        return x


class SGC(GNN):

    def __init__(self, planes, feature_dim, featureless, num_classes=1, dropout=0.0, reduce_method='max', stage='det', **kwargs):
        super().__init__(planes, feature_dim, featureless, num_classes, dropout, reduce_method, stage, **kwargs)
        assert stage == 'det'
        self.degree = len(planes)
        self.classifier = nn.Linear(self.inplanes, num_classes)

    def extract(self, x, adj):
        adj.detach_()
        D = adj.sum(dim=2, keepdim=True)
        D.detach_()
        assert (D > 0).all(), 'D should larger than 0, otherwise gradient will be NaN.'
        for _ in range(self.degree):
            if x.dim() == 3:
                x = torch.bmm(adj, x) / D
            elif x.dim() == 2:
                x = torch.spmm(adj, x) / D
        x = self.pool(x)
        x = self.classifier(x)
        return x


class MeanAggregator(nn.Module):

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        if features.dim() == 2:
            x = torch.spmm(A, features)
        elif features.dim() == 3:
            x = torch.bmm(A, features)
        else:
            raise RuntimeError('the dimension of features should be 2 or 3')
        return x


class lgcn(nn.Module):

    def __init__(self, feature_dim):
        super(lgcn, self).__init__()
        self.bn0 = nn.BatchNorm1d(feature_dim, affine=False)
        self.conv1 = GraphConv(feature_dim, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256, MeanAggregator)
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.PReLU(256), nn.Linear(256, 2))
        self.loss = nn.CrossEntropyLoss()

    def extract(self, x, A, one_hop_idxs):
        B, N, D = x.shape
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k1 = one_hop_idxs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B, k1, dout)
        for b in range(B):
            edge_feat[(b), :, :] = x[b, one_hop_idxs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)
        return pred

    def forward(self, data, return_loss=False):
        x, A, one_hop_idxs, labels = data
        x = self.extract(x, A, one_hop_idxs)
        if return_loss:
            loss = self.loss(x, labels.view(-1))
            return x, loss
        else:
            return x


class GCN_E(nn.Module):

    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_E, self).__init__()
        nhid_half = int(nhid / 2)
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.conv2 = GraphConv(nhid, nhid, MeanAggregator, dropout)
        self.conv3 = GraphConv(nhid, nhid_half, MeanAggregator, dropout)
        self.conv4 = GraphConv(nhid_half, nhid_half, MeanAggregator, dropout)
        self.nclass = nclass
        self.classifier = nn.Sequential(nn.Linear(nhid_half, nhid_half), nn.PReLU(nhid_half), nn.Linear(nhid_half, self.nclass))
        if nclass == 1:
            self.loss = nn.MSELoss()
        elif nclass == 2:
            self.loss = nn.NLLLoss()
        else:
            raise ValueError('nclass should be 1 or 2')

    def forward(self, data, return_loss=False):
        x, adj = data[0], data[1]
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        x = self.conv3(x, adj)
        x = self.conv4(x, adj)
        x = x.view(-1, x.shape[-1])
        pred = self.classifier(x)
        pred = F.log_softmax(pred, dim=-1)
        if return_loss:
            label = data[2].view(-1)
            loss = self.loss(pred, label)
            return pred, loss
        return pred


class GCN_V(nn.Module):

    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_V, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.nclass = nclass
        self.classifier = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid), nn.Linear(nhid, self.nclass))
        self.loss = torch.nn.MSELoss()

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, adj = data[0], data[1]
        x = self.conv1(x, adj)
        pred = self.classifier(x).view(-1)
        if output_feat:
            return pred, x
        if return_loss:
            label = data[2]
            loss = self.loss(pred, label)
            return pred, loss
        return pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GCN_V,
     lambda: ([], {'feature_dim': 4, 'nhid': 4, 'nclass': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MeanAggregator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_yl_1993_learn_to_cluster(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

