import sys
_module = sys.modules[__name__]
del sys
acnn_train = _module
config = _module
pyt_acnn = _module
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


from torch.utils.data import DataLoader


from torch.utils.data import dataset


import numpy as np


import torch.nn.functional as F


from torch import nn


import logging


import torch


from collections import Counter


class ACNN(nn.Module):

    def __init__(self, opt, embedding):
        super(ACNN, self).__init__()
        self.opt = opt
        self.dw = embedding.shape[1]
        self.vac_len = embedding.shape[0]
        self.d = self.dw + 2 * self.opt.DP
        self.p = (self.opt.K - 1) // 2
        self.x_embedding = nn.Embedding(self.vac_len, self.dw)
        self.x_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist_embedding = nn.Embedding(self.opt.NP, self.opt.DP)
        self.rel_weight = nn.Parameter(torch.randn(self.opt.NR, self.opt.DC))
        self.dropout = nn.Dropout(self.opt.KP)
        self.conv = nn.Conv2d(1, self.opt.DC, (self.opt.K, self.d), (1, self.d), (self.p, 0), bias=True)
        self.U = nn.Parameter(torch.randn(self.opt.DC, self.opt.NR))
        self.max_pool = nn.MaxPool1d(self.opt.N, stride=1)

    def input_attention(self, input_tuple, is_training=True):
        x, e1, e1d2, e2, e2d1, zd, d1, d2 = input_tuple
        x_emb = self.x_embedding(x)
        e1_emb = self.x_embedding(e1)
        e2_emb = self.x_embedding(e2)
        dist1_emb = self.dist_embedding(d1)
        dist2_emb = self.dist_embedding(d2)
        x_cat = torch.cat((x_emb, dist1_emb, dist2_emb), 2)
        ine1_aw = F.softmax(torch.bmm(x_emb, e1_emb.transpose(2, 1)), 1)
        ine2_aw = F.softmax(torch.bmm(x_emb, e2_emb.transpose(2, 1)), 1)
        in_aw = (ine1_aw + ine2_aw) / 2
        R = torch.mul(x_cat, in_aw)
        return R

    def attentive_pooling(self, R_star):
        RU = torch.matmul(R_star.transpose(2, 1), self.U)
        G = torch.matmul(RU, self.rel_weight)
        AP = F.softmax(G, dim=1)
        RA = torch.mul(R_star, AP.transpose(2, 1))
        wo = self.max_pool(RA).squeeze(-1)
        return wo

    def forward(self, input_tuple, is_training=True):
        R = self.input_attention(input_tuple, is_training)
        R_star = self.conv(R.unsqueeze(1)).squeeze(-1)
        R_star = torch.tanh(R_star)
        wo = self.attentive_pooling(R_star)
        return wo, self.rel_weight


class DistanceLoss(nn.Module):

    def __init__(self, nr, margin=1):
        super(DistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rel_weight, in_y, all_y):
        wo_norm = F.normalize(wo)
        wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, in_y.size()[-1], 1)
        rel_emb = torch.mm(in_y, rel_weight)
        ay_emb = torch.mm(all_y, rel_weight)
        gt_dist = torch.norm(wo_norm - rel_emb, 2, 1)
        all_dist = torch.norm(wo_norm_tile - ay_emb, 2, 2)
        masking_y = torch.mul(in_y, 10000)
        _t_dist = torch.min(torch.add(all_dist, masking_y), 1)[0]
        loss = torch.mean(self.margin + gt_dist - _t_dist)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DistanceLoss,
     lambda: ([], {'nr': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_lawlietAi_pytorch_acnn_model(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

