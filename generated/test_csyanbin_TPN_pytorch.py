import sys
_module = sys.modules[__name__]
del sys
dataset_mini = _module
dataset_tiered = _module
models = _module
test = _module
train = _module

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


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


from torch.optim.lr_scheduler import StepLR


import math


import scipy as sp


import scipy.stats


class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""

    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        h_dim, z_dim = args['h_dim'], args['z_dim']
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, padding=1))
        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)
        self.m0 = nn.MaxPool2d(2)
        self.m1 = nn.MaxPool2d(2, padding=1)

    def forward(self, x, rn):
        x = x.view(-1, 64, 5, 5)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = out.view(out.size(0), -1)
        return out


class Prototypical(nn.Module):
    """Main Module for prototypical networlks"""

    def __init__(self, args):
        super(Prototypical, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
        self.args = args
        self.encoder = CNNEncoder(args)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)
        inp = torch.cat((support, query), 0)
        emb = self.encoder(inp)
        emb_s, emb_q = torch.split(emb, [num_classes * num_support, num_classes * num_queries], 0)
        emb_s = emb_s.view(num_classes, num_support, 1600).mean(1)
        emb_q = emb_q.view(-1, 1600)
        emb_s = torch.unsqueeze(emb_s, 0)
        emb_q = torch.unsqueeze(emb_q, 1)
        dist = ((emb_q - emb_s) ** 2).mean(2)
        ce = nn.CrossEntropyLoss()
        loss = ce(-dist, torch.argmax(q_labels, 1))
        pred = torch.argmax(-dist, 1)
        gt = torch.argmax(q_labels, 1)
        correct = (pred == gt).sum()
        total = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)
        return loss, acc


class LabelPropagation(nn.Module):
    """Label Propagation"""

    def __init__(self, args):
        super(LabelPropagation, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']
        self.args = args
        self.encoder = CNNEncoder(args)
        self.relation = RelationNetwork()
        if args['rn'] == 300:
            self.alpha = torch.tensor([args['alpha']], requires_grad=False)
        elif args['rn'] == 30:
            self.alpha = nn.Parameter(torch.tensor([args['alpha']]), requires_grad=True)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        eps = np.finfo(float).eps
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)
        inp = torch.cat((support, query), 0)
        emb_all = self.encoder(inp).view(-1, 1600)
        N, d = emb_all.shape[0], emb_all.shape[1]
        if self.args['rn'] in [30, 300]:
            self.sigma = self.relation(emb_all, self.args['rn'])
            emb_all = emb_all / (self.sigma + eps)
            emb1 = torch.unsqueeze(emb_all, 1)
            emb2 = torch.unsqueeze(emb_all, 0)
            W = ((emb1 - emb2) ** 2).mean(2)
            W = torch.exp(-W / 2)
        if self.args['k'] > 0:
            topk, indices = torch.topk(W, self.args['k'])
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = (mask + torch.t(mask) > 0).type(torch.float32)
            W = W * mask
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2
        ys = s_labels
        yu = torch.zeros(num_classes * num_queries, num_classes)
        y = torch.cat((ys, yu), 0)
        F = torch.matmul(torch.inverse(torch.eye(N) - self.alpha * S + eps), y)
        Fq = F[num_classes * num_support:, :]
        ce = nn.CrossEntropyLoss()
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F, gt)
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)
        return loss, acc


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNNEncoder,
     lambda: ([], {'args': _mock_config(h_dim=4, z_dim=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (RelationNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 5, 5]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_csyanbin_TPN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

