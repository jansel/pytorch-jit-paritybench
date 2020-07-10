import sys
_module = sys.modules[__name__]
del sys
monoloco = _module
eval = _module
eval_kitti = _module
generate_kitti = _module
geom_baseline = _module
reid_baseline = _module
stereo_baselines = _module
network = _module
architectures = _module
losses = _module
net = _module
pifpaf = _module
process = _module
predict = _module
prep = _module
preprocess_ki = _module
preprocess_nu = _module
transforms = _module
run = _module
train = _module
datasets = _module
hyp_tuning = _module
trainer = _module
utils = _module
camera = _module
iou = _module
kitti = _module
logs = _module
misc = _module
nuscenes = _module
visuals = _module
figures = _module
printer = _module
webcam = _module
setup = _module
test_package = _module
test_utils = _module
test_visuals = _module

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


from collections import defaultdict


import numpy as np


import torch


import torch.backends.cudnn as cudnn


from torch import nn


import torch.nn.functional as F


import torchvision


import torchvision.transforms as T


import torch.nn as nn


import math


import logging


from torch.utils.data import Dataset


import time


import random


import copy


import warnings


from torch.utils.data import DataLoader


from torch.optim import lr_scheduler


class ResNet50(nn.Module):

    def __init__(self, num_classes, loss):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)
        if self.loss == {'xent'}:
            return y
        return y, f


class Linear(nn.Module):

    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        out = x + y
        return out


class LinearModel(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super(LinearModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


class CustomL1Loss(torch.nn.Module):
    """
    L1 loss with more weight to errors at a shorter distance
    It inherits from nn.module so it supports backward
    """

    def __init__(self, dic_norm, device, beta=1):
        super(CustomL1Loss, self).__init__()
        self.dic_norm = dic_norm
        self.device = device
        self.beta = beta

    @staticmethod
    def compute_weights(xx, beta=1):
        """
        Return the appropriate weight depending on the distance and the hyperparameter chosen
        alpha = 1 refers to the curve of A Photogrammetric Approach for Real-time...
        It is made for unnormalized outputs (to be more understandable)
        From 70 meters on every value is weighted the same (0.1**beta)
        Alpha is optional value from Focal loss. Yet to be analyzed
        """
        alpha = 1
        ww = np.maximum(0.1, 1 - xx / 78) ** beta
        return alpha * ww

    def print_loss(self):
        xx = np.linspace(0, 80, 100)
        y1 = self.compute_weights(xx, beta=1)
        y2 = self.compute_weights(xx, beta=2)
        y3 = self.compute_weights(xx, beta=3)
        plt.plot(xx, y1)
        plt.plot(xx, y2)
        plt.plot(xx, y3)
        plt.xlabel('Distance [m]')
        plt.ylabel('Loss function Weight')
        plt.legend(('Beta = 1', 'Beta = 2', 'Beta = 3'))
        plt.show()

    def forward(self, output, target):
        unnormalized_output = output.cpu().detach().numpy() * self.dic_norm['std']['Y'] + self.dic_norm['mean']['Y']
        weights_np = self.compute_weights(unnormalized_output, self.beta)
        weights = torch.from_numpy(weights_np).float()
        losses = torch.abs(output - target) * weights
        loss = losses.mean()
        return loss


class LaplacianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """

    def __init__(self, size_average=True, reduce=True, evaluate=False):
        super(LaplacianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate

    def laplacian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py

        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]
        norm = 1 - mu / xx
        term_a = torch.abs(norm) * torch.exp(-si)
        term_b = si
        norm_bi = np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(torch.exp(si).cpu().detach().numpy())
        if self.evaluate:
            return norm_bi
        return term_a + term_b

    def forward(self, outputs, targets):
        values = self.laplacian_1d(outputs, targets)
        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


class GaussianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """

    def __init__(self, device, size_average=True, reduce=True, evaluate=False):
        super(GaussianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate
        self.device = device

    def gaussian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]
        min_si = torch.ones(si.size()) * 0.1
        si = torch.max(min_si, si)
        norm = xx - mu
        term_a = (norm / si) ** 2 / 2
        term_b = torch.log(si * math.sqrt(2 * math.pi))
        norm_si = np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(si.cpu().detach().numpy())
        if self.evaluate:
            return norm_si
        return term_a + term_b

    def forward(self, outputs, targets):
        values = self.gaussian_1d(outputs, targets)
        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GaussianLoss,
     lambda: ([], {'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LaplacianLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'linear_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LinearModel,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (ResNet50,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_vita_epfl_monoloco(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

