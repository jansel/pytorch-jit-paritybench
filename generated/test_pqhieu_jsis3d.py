import sys
_module = sys.modules[__name__]
del sys
eval = _module
loaders = _module
s3dis = _module
losses = _module
discriminative = _module
nll = _module
models = _module
mtpnet = _module
pointnet = _module
plot = _module
pred = _module
collect_annotations = _module
estimate_mean_size = _module
estimate_median_freq = _module
prepare_h5 = _module
train = _module
utils = _module
merge = _module

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


import numpy as np


import torch.utils.data as data


import torch


import torch.nn as nn


import torch.nn.functional as F


from sklearn.cluster import MeanShift


import torch.optim as optim


from collections import defaultdict


class DiscriminativeLoss(nn.Module):

    def __init__(self, delta_d, delta_v, alpha=1.0, beta=1.0, gamma=0.001, reduction='mean'):
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_d = delta_d
        self.delta_v = delta_v

    def forward(self, embedded, masks, size):
        centroids = self._centroids(embedded, masks, size)
        L_v = self._variance(embedded, masks, centroids, size)
        L_d = self._distance(centroids, size)
        L_r = self._regularization(centroids, size)
        loss = self.alpha * L_v + self.beta * L_d + self.gamma * L_r
        return loss

    def _centroids(self, embedded, masks, size):
        batch_size = embedded.size(0)
        embedding_size = embedded.size(2)
        K = masks.size(2)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        masks = masks.unsqueeze(3)
        x = x * masks
        centroids = []
        for i in range(batch_size):
            n = size[i]
            mu = x[(i), :, :n].sum(0) / masks[(i), :, :n].sum(0)
            if K > n:
                m = int(K - n)
                filled = torch.zeros(m, embedding_size)
                filled = filled
                mu = torch.cat([mu, filled], dim=0)
            centroids.append(mu)
        centroids = torch.stack(centroids)
        return centroids

    def _variance(self, embedded, masks, centroids, size):
        batch_size = embedded.size(0)
        num_points = embedded.size(1)
        embedding_size = embedded.size(2)
        K = masks.size(2)
        mu = centroids.unsqueeze(1).expand(-1, num_points, -1, -1)
        x = embedded.unsqueeze(2).expand(-1, -1, K, -1)
        var = torch.norm(x - mu, 2, dim=3)
        var = torch.clamp(var - self.delta_v, min=0.0) ** 2
        var = var * masks
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            loss += torch.sum(var[(i), :, :n]) / torch.sum(masks[(i), :, :n])
        loss /= batch_size
        return loss

    def _distance(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            if n <= 1:
                continue
            mu = centroids[(i), :n, :]
            mu_a = mu.unsqueeze(1).expand(-1, n, -1)
            mu_b = mu_a.permute(1, 0, 2)
            diff = mu_a - mu_b
            norm = torch.norm(diff, 2, dim=2)
            margin = 2 * self.delta_d * (1.0 - torch.eye(n))
            margin = margin
            distance = torch.sum(torch.clamp(margin - norm, min=0.0) ** 2)
            distance /= float(n * (n - 1))
            loss += distance
        loss /= batch_size
        return loss

    def _regularization(self, centroids, size):
        batch_size = centroids.size(0)
        loss = 0.0
        for i in range(batch_size):
            n = size[i]
            mu = centroids[(i), :n, :]
            norm = torch.norm(mu, 2, dim=1)
            loss += torch.mean(norm)
        loss /= batch_size
        return loss


class NLLLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.nll = nn.NLLLoss(weight, reduction=reduction)

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.view(batch_size * num_points, -1)
        y = y.view(batch_size * num_points)
        loss = self.nll(x, y)
        return loss


class STN3D(nn.Module):

    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(nn.Conv1d(input_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, input_channels * input_channels))

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class PointNet(nn.Module):

    def __init__(self, input_channels):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(nn.Conv1d(input_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(), nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU())

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        T = self.stn1(x)
        x = torch.bmm(T, x)
        x = self.mlp1(x)
        T = self.stn2(x)
        f = torch.bmm(T, x)
        x = self.mlp2(f)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, f], 1)
        x = self.mlp3(x)
        return x


class MTPNet(nn.Module):

    def __init__(self, input_channels, num_classes, embedding_size):
        super(MTPNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.input_channels = input_channels
        self.net = PointNet(self.input_channels)
        self.fc1 = nn.Conv1d(128, self.num_classes, 1)
        self.fc2 = nn.Conv1d(128, self.embedding_size, 1)

    def forward(self, x):
        x = self.net(x)
        logits = self.fc1(x)
        logits = logits.transpose(2, 1)
        logits = torch.log_softmax(logits, dim=-1)
        embedded = self.fc2(x)
        embedded = embedded.transpose(2, 1)
        return logits, embedded


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MTPNet,
     lambda: ([], {'input_channels': 4, 'num_classes': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PointNet,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (STN3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64])], {}),
     True),
]

class Test_pqhieu_jsis3d(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

