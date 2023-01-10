import sys
_module = sys.modules[__name__]
del sys
config = _module
train_cls_config = _module
train_reg_config = _module
dataset = _module
auglib = _module
database = _module
verifybase = _module
model = _module
dul_loss = _module
dul_reg = _module
dul_resnet = _module
faster1v1 = _module
fc_layer = _module
train_dul_cls = _module
train_dul_reg = _module

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


import random


import numpy as np


from torch.utils import data


import pandas as pd


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


from sklearn import metrics


import time


import torchvision


import torch.optim as optim


from torch.utils.data import DataLoader


class ClsLoss(nn.Module):
    """ Classic loss function for face recognition """

    def __init__(self, args):
        super(ClsLoss, self).__init__()
        self.args = args

    def forward(self, predy, target, mu=None, logvar=None):
        loss = None
        if self.args.loss_mode == 'focal_loss':
            logp = F.cross_entropy(predy, target, reduce=False)
            prob = torch.exp(-logp)
            loss = ((1 - prob) ** self.args.loss_power * logp).mean()
        elif self.args.loss_mode == 'hardmining':
            batchsize = predy.shape[0]
            logp = F.cross_entropy(predy, label, reduce=False)
            inv_index = torch.argsort(-logp)
            num_hard = int(self.args.hard_ratio * batch_size)
            hard_idx = ind_sorted[:num_hard]
            loss = torch.sum(F.cross_entropy(pred[hard_idx], label[hard_idx]))
        else:
            loss = F.cross_entropy(predy, target)
        if mu is not None and logvar is not None:
            kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            kl_loss = kl_loss.sum(dim=1).mean()
            loss = loss + self.args.kl_lambda * kl_loss
        return loss


class RegLoss(nn.Module):

    def __init__(self, feat_dim=512, classnum=85742):
        super(RegLoss, self).__init__()
        self.feat_dim = feat_dim
        self.classnum = classnum
        self.centers = torch.Tensor(classnum, feat_dim)

    def fetch_center_from_fc_layer(self, fc_state_dict):
        weights_key = 'module.weight' if 'module.weight' in fc_state_dict.keys() else 'weight'
        try:
            weights = fc_state_dict[weights_key]
        except Exception as e:
            None
        else:
            assert weights.size() == torch.Size([self.classnum, self.feat_dim]), 'weights.size can not match with (classnum, feat_dim)'
            self.centers = weights
            None

    def forward(self, mu, logvar, labels):
        fit_loss = (self.fc_weights[labels] - mu).pow(2) / (1e-10 + torch.exp(logvar))
        reg_loss = (fit_loss + logvar) / 2.0
        reg_loss = torch.sum(reg_loss, dim=1).mean()
        return reg_loss


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SEBlock(nn.Module):
    """ Squeeze and Excitation Module """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False), nn.PReLU(), nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.se_layer(x)


class IR_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IR_BasicBlock, self).__init__()
        self.ir_basic = nn.Sequential(nn.BatchNorm2d(inplanes, eps=2e-05), nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(planes, eps=2e-05), nn.PReLU(), nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(planes, eps=2e-05))
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.prelu = nn.PReLU()
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.ir_basic(x)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return out + residual


class IR_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IR_Bottleneck, self).__init__()
        self.ir_bottle = nn.Sequential(nn.BatchNorm2d(inplanes, eps=2e-05), nn.Conv2d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm2d(planes, eps=2e-05), nn.PReLU(), nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(planes, eps=2e-05), nn.PReLU(), nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False), nn.BatchNorm2d(planes * self.expansion, eps=2e-05))
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.prelu = nn.PReLU()
        if self.use_se:
            self.se = SEBlock(planes * self.expansion)

    def forward(self, x):
        residual = x
        out = self.ir_bottle(x)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.prelu(out + residual)


class DULResNet(nn.Module):

    def __init__(self, block, layers, feat_dim=512, drop_ratio=0.4, use_se=True, used_as='baseline'):
        """
        used_as = baseline : just use mu_head for deterministic model
        used_as = dul_cls  : use both mu_head and logvar_head for dul_cls model
        used_as = backbone : neither mu_head nor logvar_head is used, just for feature extracting.
        """
        super(DULResNet, self).__init__()
        self.inplanes = 64
        self.use_se = use_se
        self.used_as = used_as
        self.layer0 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(self.inplanes), nn.PReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.mu_head = nn.Sequential(nn.BatchNorm2d(512 * block.expansion, eps=2e-05, affine=False), nn.Dropout(p=drop_ratio), Flatten(), nn.Linear(512 * block.expansion * 7 * 7, feat_dim), nn.BatchNorm1d(feat_dim, eps=2e-05))
        if used_as == 'dul_cls':
            self.logvar_head = nn.Sequential(nn.BatchNorm2d(512 * block.expansion, eps=2e-05, affine=False), nn.Dropout(p=drop_ratio), Flatten(), nn.Linear(512 * block.expansion * 7 * 7, feat_dim), nn.BatchNorm1d(feat_dim, eps=2e-05))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.used_as == 'backbone':
            mu = x
            logvar = None,
            embedding = None
        elif self.used_as == 'baseline':
            mu = None
            logvar = None,
            embedding = self.mu_head(x)
        else:
            mu = self.mu_head(x)
            logvar = self.logvar_head(x)
            embedding = self._reparameterize(mu, logvar)
        return mu, logvar, embedding


class RegHead(nn.Module):
    expansion = 1
    resnet_out_channels = 512

    def __init__(self, feat_dim=512, drop_ratio=0.4):
        super(RegHead, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1) * 0.0001)
        self.beta = nn.Parameter(torch.ones(1) * -7)
        self.mu_head = nn.Sequential(nn.BatchNorm2d(self.resnet_out_channels, eps=2e-05, affine=False), nn.Dropout(p=drop_ratio), Flatten(), nn.Linear(self.resnet_out_channels * self.expansion * 7 * 7, feat_dim), nn.BatchNorm1d(feat_dim, eps=2e-05))
        self.logvar_head = nn.Sequential(nn.BatchNorm2d(self.resnet_out_channels * self.expansion, eps=2e-05, affine=False), nn.Dropout(p=drop_ratio), Flatten(), nn.Linear(self.resnet_out_channels * self.expansion * 7 * 7, feat_dim, bias=False), nn.BatchNorm1d(feat_dim, eps=0.001), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim, eps=0.001, affine=False))

    def forward(self, layer4_out):
        mu = self.mu_head(layer4_out)
        logvar = self.logvar_head(layer4_out)
        logvar = self.gamma * logvar + self.beta
        return mu, logvar


class FullyConnectedLayer(nn.Module):
    """ fully connected layer lib for face-recognition """

    def __init__(self, args):
        super(FullyConnectedLayer, self).__init__()
        self.args = args
        self.weight = nn.Parameter(torch.Tensor(args.classnum, args.in_feats))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(args.margin)
        self.sin_m = math.sin(args.margin)
        self.remain = math.sin(math.pi - self.args.margin) * self.args.margin
        self.thresh = math.cos(math.pi - self.args.margin)
        self.register_buffer('factor_t', torch.zeros(1))
        self.iter = 0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.mlambda = [lambda x: x ** 0, lambda x: x ** 1, lambda x: 2 * x ** 2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x ** 4 - 8 * x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]

    def forward(self, x, label):
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1, 1)
        cosin_simi = cos_theta[torch.arange(0, label.size(0)), label].view(-1, 1)
        if self.args.fc_mode == 'softmax':
            score = cosin_simi
        elif self.args.fc_mode == 'sphereface':
            self.iter += 1
            self.lamb = max(self.lambda_min, self.base * (1 + self.alpha * self.iter) ** (-1 * self.power))
            cos_theta_m = self.mlambda[int(self.args.margin)](cosin_simi)
            theta = cosin_simi.data.acos()
            k = (self.args.margin * theta / math.pi).floor()
            phi_theta = (-1.0) ** k * cos_theta_m - 2 * k
            score = (self.lamb * cosin_simi + phi_theta) / (1 + self.lamb)
        elif self.args.fc_mode == 'cosface':
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cosin_simi - self.args.margin, cosin_simi)
            else:
                score = cosin_simi - self.args.margin
        elif self.args.fc_mode == 'arcface':
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
            else:
                score = torch.where(cosin_simi > self.thresh, cos_theta_m, cosin_simi - self.remain)
        elif self.args.fc_mode == 'mvcos':
            mask = cos_theta > cosin_simi - self.args.margin
            hard_vector = cos_theta[mask]
            if self.args.hard_mode == 'adaptive':
                cos_theta[mask] = (self.args.t + 1.0) * hard_vector + self.args.t
            else:
                cos_theta[mask] = hard_vector + self.args.t
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cosin_simi - self.args.margin, cosin_simi)
            else:
                score = cosin_simi - self.args.margin
        elif self.args.fc_mode == 'mvarc':
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            if self.args.hard_mode == 'adaptive':
                cos_theta[mask] = (self.args.t + 1.0) * hard_vector + self.args.t
            else:
                cos_theta[mask] = hard_vector + self.args.t
            if self.args.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
            else:
                score = cos_theta_m
        elif self.args.fc_mode == 'curface':
            with torch.no_grad():
                origin_cos = cos_theta
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi - self.mm)
            hard_sample = cos_theta[mask]
            with torch.no_grad():
                self.factor_t = cos_theta_m.mean() * 0.01 + 0.99 * self.factor_t
            cos_theta[mask] = hard_sample * (self.factor_t + hard_sample)
        else:
            raise Exception('unknown fc type!')
        cos_theta.scatter_(1, label.data.view(-1, 1), score)
        cos_theta *= self.args.scale
        return cos_theta


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClsLoss,
     lambda: ([], {'args': _mock_config(loss_mode=MSELoss())}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Ontheway361_dul_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

