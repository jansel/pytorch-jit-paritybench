import sys
_module = sys.modules[__name__]
del sys
fewshots = _module
SetupEpisode = _module
data = _module
base = _module
cache = _module
load = _module
queries = _module
setup = _module
utils = _module
engine = _module
labels_lrd2_bin = _module
labels_lrd2_multi = _module
labels_r2d2 = _module
models = _module
adjust = _module
factory = _module
load_model = _module
lrd2 = _module
r2d2 = _module
utils = _module
log = _module
model = _module
norm = _module
vis = _module
eval = _module
run_eval = _module
run_train = _module
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


import numpy as np


from functools import partial


from torch.autograd import Variable


import torchvision


from torchvision import transforms


from torch import nn


import torch.nn as nn


from torch import transpose as t


from torch import inverse as inv


from torch import mm


import math


import logging


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


class AdjustLayer(nn.Module):

    def __init__(self, init_scale=0.0001, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale]))
        self.bias = nn.Parameter(torch.FloatTensor([init_bias]))
        self.base = base

    def forward(self, x):
        if self.base == 1:
            return x * self.scale + self.bias
        else:
            return x * self.base ** self.scale + self.base ** self.bias - 1


class LambdaLayer(nn.Module):

    def __init__(self, learn_lambda=False, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda])
        self.base = base
        if learn_lambda:
            self.l = nn.Parameter(self.l)
        else:
            self.l = Variable(self.l)

    def forward(self, x):
        if self.base == 1:
            return x * self.l
        else:
            return x * self.base ** self.l


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def _norm(num_channels, bn_momentum, groupnorm=False):
    if groupnorm:
        return norm.GroupNorm(num_channels)
    else:
        return nn.BatchNorm2d(num_channels, momentum=bn_momentum)


class RRFeatures(nn.Module):

    def __init__(self, x_dim, parameters, lrelu_slope, drop, groupnorm, bn_momentum):
        super(RRFeatures, self).__init__()
        self.features1 = nn.Sequential(nn.Conv2d(x_dim, parameters[0], 3, padding=1), _norm(parameters[0], bn_momentum, groupnorm=groupnorm), nn.MaxPool2d(2, stride=2), nn.LeakyReLU(lrelu_slope))
        self.features2 = nn.Sequential(nn.Conv2d(parameters[0], parameters[1], 3, padding=1), _norm(parameters[1], bn_momentum, groupnorm=groupnorm), nn.MaxPool2d(2, stride=2), nn.LeakyReLU(lrelu_slope))
        self.features3 = nn.Sequential(nn.Conv2d(parameters[1], parameters[2], 3, padding=1), _norm(parameters[2], bn_momentum, groupnorm=groupnorm), nn.MaxPool2d(2, stride=2), nn.LeakyReLU(lrelu_slope), nn.Dropout(drop))
        self.features4 = nn.Sequential(nn.Conv2d(parameters[2], parameters[3], 3, padding=1), _norm(parameters[3], bn_momentum, groupnorm=groupnorm), nn.MaxPool2d(2, stride=1), nn.LeakyReLU(lrelu_slope), nn.Dropout(drop))
        self.pool3 = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x3 = self.pool3(x)
        x3 = x3.view(x3.size(0), -1)
        x = self.features4(x)
        x4 = x.view(x.size(0), -1)
        x = torch.cat((x3, x4), 1)
        return x


def roll(x, shift):
    return torch.cat((x[-shift:], x[:-shift]))


def shuffle_queries_bin(x, n_way, n_shot, n_query, n_augment, y_outer, y_outer_2d):
    ind_xs = torch.linspace(0, n_way * n_shot * n_augment - 1, steps=n_way * n_shot * n_augment).long()
    ind_xs = Variable(ind_xs)
    perm_xq = torch.randperm(n_way * n_query).long()
    perm_xq = Variable(perm_xq)
    permute = torch.cat([ind_xs, perm_xq + len(ind_xs)])
    return x[permute, :, :, :], y_outer[perm_xq], y_outer_2d[perm_xq]


def shuffle_queries_multi(x, n_way, n_shot, n_query, n_augment, y_binary, y):
    ind_xs = torch.linspace(0, n_way * n_shot * n_augment - 1, steps=n_way * n_shot * n_augment).long()
    ind_xs = Variable(ind_xs)
    perm_xq = torch.randperm(n_way * n_query).long()
    perm_xq = Variable(perm_xq)
    permute = torch.cat([ind_xs, perm_xq + len(ind_xs)])
    x = x[permute, :, :, :]
    y_binary = y_binary[perm_xq, :]
    y = y[perm_xq]
    return x, y_binary, y


class LRD2(nn.Module):

    def __init__(self, encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, irls_iterations, linsys):
        super(LRD2, self).__init__()
        self.encoder = encoder
        self.debug = debug
        self.lambda_ = LambdaLayer(learn_lambda, init_lambda, lambda_base)
        self.L = nn.CrossEntropyLoss()
        self.L_bin = nn.BCEWithLogitsLoss()
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)
        self.output_dim = out_dim
        self.n_augment = n_augment
        assert irls_iterations > 0
        self.iterations = irls_iterations
        self.linsys = linsys

    def loss(self, sample):
        xs, xq = Variable(sample['xs']), Variable(sample['xq'])
        assert xs.size(0) == xq.size(0)
        n_way, n_shot, n_query = xs.size(0), xs.size(1), xq.size(1)
        x = torch.cat([xs.view(n_way * n_shot * self.n_augment, *xs.size()[2:]), xq.view(n_way * n_query, *xq.size()[2:])], 0)
        if n_way > 2:
            y_inner_binary = labels_lrd2_multi.make_float_label(n_shot, n_way * n_shot * self.n_augment)
            y_outer_binary = labels_r2d2.make_float_label(n_way, n_query)
            y_outer = labels_r2d2.make_long_label(n_way, n_query)
            x, y_outer_binary, y_outer = shuffle_queries_multi(x, n_way, n_shot, n_query, self.n_augment, y_outer_binary, y_outer)
            zs, zq = self.encode(x, n_way, n_shot)
            scores = Variable(torch.FloatTensor(n_query * n_way, n_way).zero_())
            for i in range(n_way):
                w0 = Variable(torch.FloatTensor(n_way * n_shot * self.n_augment).zero_())
                wb = self.ir_logistic(zs, w0, y_inner_binary)
                y_hat = mm(zq, wb)
                scores[:, i] = y_hat
                y_inner_binary = roll(y_inner_binary, n_shot)
            _, ind_prediction = torch.max(scores, 1)
            _, ind_gt = torch.max(y_outer_binary, 1)
            loss_val = self.L(scores, y_outer)
            acc_val = torch.eq(ind_prediction, ind_gt).float().mean()
            return loss_val, {'loss': loss_val.data[0], 'acc': acc_val.data[0]}
        else:
            y_inner_binary = labels_lrd2_bin.make_float_label(n_way, n_shot * self.n_augment)
            y_outer = labels_lrd2_bin.make_byte_label(n_way, n_query)
            y_outer_2d = labels_lrd2_bin.make_float_label(n_way, n_query).unsqueeze(1)
            x, y_outer, y_outer_2d = shuffle_queries_bin(x, n_way, n_shot, n_query, self.n_augment, y_outer, y_outer_2d)
            zs, zq = self.encode(x, n_way, n_shot)
            w0 = Variable(torch.FloatTensor(n_way * n_shot * self.n_augment).zero_())
            wb = self.ir_logistic(zs, w0, y_inner_binary)
            y_hat = mm(zq, wb)
            ind_prediction = (torch.sigmoid(y_hat) >= 0.5).squeeze(1)
            loss_val = self.L_bin(y_hat, y_outer_2d)
            acc_val = torch.eq(ind_prediction, y_outer).float().mean()
            return loss_val, {'loss': loss_val.data[0], 'acc': acc_val.data[0]}

    def encode(self, X, n_way, n_shot):
        z = self.encoder.forward(X)
        zs = z[:n_way * n_shot * self.n_augment]
        zq = z[n_way * n_shot * self.n_augment:]
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)), 1))
        zs = torch.cat((zs, ones), 1)
        ones = Variable(torch.unsqueeze(torch.ones(zq.size(0)), 1))
        zq = torch.cat((zq, ones), 1)
        return zs, zq

    def ir_logistic(self, X, w0, y_inner):
        eta = w0
        mu = torch.sigmoid(eta)
        s = mu * (1 - mu)
        z = eta + (y_inner - mu) / s
        S = torch.diag(s)
        w_ = mm(t(X, 0, 1), inv(mm(X, t(X, 0, 1)) + self.lambda_(inv(S))))
        z_ = t(z.unsqueeze(0), 0, 1)
        w = mm(w_, z_)
        for i in range(self.iterations - 1):
            eta = w0 + mm(X, w).squeeze(1)
            mu = torch.sigmoid(eta)
            s = mu * (1 - mu)
            z = eta + (y_inner - mu) / s
            S = torch.diag(s)
            z_ = t(z.unsqueeze(0), 0, 1)
            if not self.linsys:
                w_ = mm(t(X, 0, 1), inv(mm(X, t(X, 0, 1)) + self.lambda_(inv(S))))
                w = mm(w_, z_)
            else:
                A = mm(X, t(X, 0, 1)) + self.lambda_(inv(S))
                w_, _ = gesv(z_, A)
                w = mm(t(X, 0, 1), w_)
        return w


def to_variable(x):
    if torch.cuda.is_available():
        x = x
    return Variable(x)


def make_float_label(n_way, n_samples):
    label = torch.FloatTensor(n_way * n_samples, n_way).zero_()
    for i in range(n_way):
        label[n_samples * i:n_samples * (i + 1), i] = 1
    return to_variable(label)


def make_long_label(n_way, n_samples):
    label = torch.LongTensor(n_way * n_samples).zero_()
    for i in range(n_way * n_samples):
        label[i] = i // n_samples
    return to_variable(label)


def t_(x):
    return t(x, 0, 1)


class RRNet(nn.Module):

    def __init__(self, encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, linsys):
        super(RRNet, self).__init__()
        self.encoder = encoder
        self.debug = debug
        self.lambda_rr = LambdaLayer(learn_lambda, init_lambda, lambda_base)
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)
        self.output_dim = out_dim
        self.n_augment = n_augment
        self.linsys = linsys

    def loss(self, sample):
        xs, xq = Variable(sample['xs']), Variable(sample['xq'])
        assert xs.size(0) == xq.size(0)
        n_way, n_shot, n_query = xs.size(0), xs.size(1), xq.size(1)
        if n_way * n_shot * self.n_augment > self.output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(self.output_dim + 1))
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(n_way * n_shot * self.n_augment))
        y_inner = make_float_label(n_way, n_shot * self.n_augment) / np.sqrt(n_way * n_shot * self.n_augment)
        y_outer_binary = make_float_label(n_way, n_query)
        y_outer = make_long_label(n_way, n_query)
        x = torch.cat([xs.view(n_way * n_shot * self.n_augment, *xs.size()[2:]), xq.view(n_way * n_query, *xq.size()[2:])], 0)
        x, y_outer_binary, y_outer = shuffle_queries_multi(x, n_way, n_shot, n_query, self.n_augment, y_outer_binary, y_outer)
        z = self.encoder.forward(x)
        zs = z[:n_way * n_shot * self.n_augment]
        zq = z[n_way * n_shot * self.n_augment:]
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)), 1))
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)
        else:
            wb = self.rr_standard(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)
        w = wb.narrow(dimension=0, start=0, length=self.output_dim)
        b = wb.narrow(dimension=0, start=self.output_dim, length=1)
        out = mm(zq, w) + b
        y_hat = self.adjust(out)
        _, ind_prediction = torch.max(y_hat, 1)
        _, ind_gt = torch.max(y_outer_binary, 1)
        loss_val = self.L(y_hat, y_outer)
        acc_val = torch.eq(ind_prediction, ind_gt).float().mean()
        return loss_val, {'loss': loss_val.data[0], 'acc': acc_val.data[0]}

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)
        if not linsys:
            w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            w, _ = gesv(v, A)
        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)
        if not linsys:
            w = mm(mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = gesv(v, A)
            w = mm(t_(x), w_)
        return w


class GroupNorm(nn.Module):

    def __init__(self, num_features, num_groups=32, eps=1e-05):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0
        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdjustLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bertinetto_r2d2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

