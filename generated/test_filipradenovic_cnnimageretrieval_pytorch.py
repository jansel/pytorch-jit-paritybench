import sys
_module = sys.modules[__name__]
del sys
cirtorch = _module
datasets = _module
datahelpers = _module
genericdataset = _module
testdataset = _module
traindataset = _module
examples = _module
test = _module
test_e2e = _module
train = _module
layers = _module
functional = _module
loss = _module
normalization = _module
pooling = _module
networks = _module
imageretrievalnet = _module
utils = _module
download = _module
download_win = _module
evaluate = _module
general = _module
whiten = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import math


import numpy as np


import torch


import torch.nn as nn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.models as models


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import torch.utils.model_zoo as model_zoo


import torchvision


class ContrastiveLoss(nn.Module):
    """CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-06):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        return LF.contrastive_loss(x, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return LF.triplet_loss(x, label, margin=self.margin)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class L2N(nn.Module):

    def __init__(self, eps=1e-06):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class PowerLaw(nn.Module):

    def __init__(self, eps=1e-06):
        super(PowerLaw, self).__init__()
        self.eps = eps

    def forward(self, x):
        return LF.powerlaw(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


class MAC(nn.Module):

    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return LF.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return LF.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-06):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GeMmp(nn.Module):

    def __init__(self, p=3, mp=1, eps=1e-06):
        super(GeMmp, self).__init__()
        self.p = Parameter(torch.ones(mp) * p)
        self.mp = mp
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '[{}]'.format(self.mp) + ', ' + 'eps=' + str(self.eps) + ')'


class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-06):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class Rpool(nn.Module):

    def __init__(self, rpool, whiten=None, L=3, eps=1e-06):
        super(Rpool, self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def forward(self, x, aggregate=True):
        o = LF.roipool(x, self.rpool, self.L, self.eps)
        s = o.size()
        o = o.view(s[0] * s[1], s[2], s[3], s[4])
        o = self.norm(o)
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))
        o = o.view(s[0], s[1], s[2], s[3], s[4])
        if aggregate:
            o = self.norm(o.sum(1, keepdim=False))
        return o

    def __repr__(self):
        return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'


class ImageRetrievalNet(nn.Module):

    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        o = self.features(x)
        if self.lwhiten is not None:
            s = o.size()
            o = o.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0], s[2], s[3], self.lwhiten.out_features).permute(0, 3, 1, 2)
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)
        if self.whiten is not None:
            o = self.norm(self.whiten(o))
        return o.permute(1, 0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr

