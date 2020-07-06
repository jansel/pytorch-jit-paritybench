import sys
_module = sys.modules[__name__]
del sys
evaluate_imagenet = _module
evaluate_timing = _module
models = _module
selecsls = _module
util = _module
imagenet_data_loader = _module

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


import logging


import time


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import math


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


class SelecSLSBlock(nn.Module):

    def __init__(self, inp, skip, k, oup, isFirst, stride):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]
        self.conv1 = nn.Sequential(nn.Conv2d(inp, k, 3, stride, 1, groups=1, bias=False, dilation=1), nn.BatchNorm2d(k), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(k, k, 1, 1, 0, groups=1, bias=False, dilation=1), nn.BatchNorm2d(k), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(k, k // 2, 3, 1, 1, groups=1, bias=False, dilation=1), nn.BatchNorm2d(k // 2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(k // 2, k, 1, 1, 0, groups=1, bias=False, dilation=1), nn.BatchNorm2d(k), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(k, k // 2, 3, 1, 1, groups=1, bias=False, dilation=1), nn.BatchNorm2d(k // 2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(2 * k + (0 if isFirst else skip), oup, 1, 1, 0, groups=1, bias=False, dilation=1), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) in [1, 2]
        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.isFirst:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


class SelecSLSBlockFused(nn.Module):

    def __init__(self, inp, skip, a, b, c, d, e, oup, isFirst, stride):
        super(SelecSLSBlockFused, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]
        self.conv1 = nn.Sequential(nn.Conv2d(inp, a, 3, stride, 1, groups=1, bias=True, dilation=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(a, b, 1, 1, 0, groups=1, bias=True, dilation=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(b, c, 3, 1, 1, groups=1, bias=True, dilation=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(c, d, 1, 1, 0, groups=1, bias=True, dilation=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(d, e, 3, 1, 1, groups=1, bias=True, dilation=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(a + c + e + (0 if isFirst else skip), oup, 1, 1, 0, groups=1, bias=True, dilation=1), nn.ReLU(inplace=True))

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) in [1, 2]
        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.isFirst:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]


def bn_fuse(c, b):
    """ BN fusion code adapted from my Caffe BN fusion code and code from @MIPT-Oulu. This function assumes everything is on the cpu"""
    with torch.no_grad():
        eps = b.eps
        mu = b.running_mean
        var = b.running_var
        gamma = b.weight
        if 'bias' in b.state_dict():
            beta = b.bias
        else:
            beta = torch.zeros(gamma.size(0)).float()
        W = c.weight
        if 'bias' in c.state_dict():
            bias = c.bias
        else:
            bias = torch.zeros(W.size(0)).float()
        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)
        W.mul_(A)
        bias.add_(b)
    return W.clone().detach(), bias.clone().detach()


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class Net(nn.Module):

    def __init__(self, nClasses=1000, config='SelecSLS60'):
        super(Net, self).__init__()
        self.stem = conv_bn(3, 32, 2)
        self.features = []
        if config == 'SelecSLS42':
            None
            self.selecSLS_config = [[32, 0, 64, 64, True, 2], [64, 64, 64, 128, False, 1], [128, 0, 144, 144, True, 2], [144, 144, 144, 288, False, 1], [288, 0, 304, 304, True, 2], [304, 304, 304, 480, False, 1]]
            self.head = nn.Sequential(conv_bn(480, 960, 2), conv_bn(960, 1024, 1), conv_bn(1024, 1024, 2), conv_1x1_bn(1024, 1280))
            self.num_features = 1280
        elif config == 'SelecSLS42_B':
            None
            self.selecSLS_config = [[32, 0, 64, 64, True, 2], [64, 64, 64, 128, False, 1], [128, 0, 144, 144, True, 2], [144, 144, 144, 288, False, 1], [288, 0, 304, 304, True, 2], [304, 304, 304, 480, False, 1]]
            self.head = nn.Sequential(conv_bn(480, 960, 2), conv_bn(960, 1024, 1), conv_bn(1024, 1280, 2), conv_1x1_bn(1280, 1024))
            self.num_features = 1024
        elif config == 'SelecSLS60':
            None
            self.selecSLS_config = [[32, 0, 64, 64, True, 2], [64, 64, 64, 128, False, 1], [128, 0, 128, 128, True, 2], [128, 128, 128, 128, False, 1], [128, 128, 128, 288, False, 1], [288, 0, 288, 288, True, 2], [288, 288, 288, 288, False, 1], [288, 288, 288, 288, False, 1], [288, 288, 288, 416, False, 1]]
            self.head = nn.Sequential(conv_bn(416, 756, 2), conv_bn(756, 1024, 1), conv_bn(1024, 1024, 2), conv_1x1_bn(1024, 1280))
            self.num_features = 1280
        elif config == 'SelecSLS60_B':
            None
            self.selecSLS_config = [[32, 0, 64, 64, True, 2], [64, 64, 64, 128, False, 1], [128, 0, 128, 128, True, 2], [128, 128, 128, 128, False, 1], [128, 128, 128, 288, False, 1], [288, 0, 288, 288, True, 2], [288, 288, 288, 288, False, 1], [288, 288, 288, 288, False, 1], [288, 288, 288, 416, False, 1]]
            self.head = nn.Sequential(conv_bn(416, 756, 2), conv_bn(756, 1024, 1), conv_bn(1024, 1280, 2), conv_1x1_bn(1280, 1024))
            self.num_features = 1024
        elif config == 'SelecSLS84':
            None
            self.selecSLS_config = [[32, 0, 64, 64, True, 2], [64, 64, 64, 144, False, 1], [144, 0, 144, 144, True, 2], [144, 144, 144, 144, False, 1], [144, 144, 144, 144, False, 1], [144, 144, 144, 144, False, 1], [144, 144, 144, 304, False, 1], [304, 0, 304, 304, True, 2], [304, 304, 304, 304, False, 1], [304, 304, 304, 304, False, 1], [304, 304, 304, 304, False, 1], [304, 304, 304, 304, False, 1], [304, 304, 304, 512, False, 1]]
            self.head = nn.Sequential(conv_bn(512, 960, 2), conv_bn(960, 1024, 1), conv_bn(1024, 1024, 2), conv_1x1_bn(1024, 1280))
            self.num_features = 1280
        elif config == 'SelecSLS102':
            None
            self.selecSLS_config = [[32, 0, 64, 64, True, 2], [64, 64, 64, 64, False, 1], [64, 64, 64, 64, False, 1], [64, 64, 64, 128, False, 1], [128, 0, 128, 128, True, 2], [128, 128, 128, 128, False, 1], [128, 128, 128, 128, False, 1], [128, 128, 128, 128, False, 1], [128, 128, 128, 288, False, 1], [288, 0, 288, 288, True, 2], [288, 288, 288, 288, False, 1], [288, 288, 288, 288, False, 1], [288, 288, 288, 288, False, 1], [288, 288, 288, 288, False, 1], [288, 288, 288, 288, False, 1], [288, 288, 288, 480, False, 1]]
            self.head = nn.Sequential(conv_bn(480, 960, 2), conv_bn(960, 1024, 1), conv_bn(1024, 1024, 2), conv_1x1_bn(1024, 1280))
            self.num_features = 1280
        else:
            raise ValueError('Invalid net configuration ' + config + ' !!!')
        for inp, skip, k, oup, isFirst, stride in self.selecSLS_config:
            self.features.append(SelecSLSBlock(inp, skip, k, oup, isFirst, stride))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Linear(self.num_features, nClasses))

    def forward(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def prune_and_fuse(self, gamma_thresh, verbose=False):
        """ Function that iterates over the modules in the model and prunes different parts by name. Sparsity emerges implicitly due to the use of
        adaptive gradient descent approaches such as Adam, in conjunction with L2 or WD regularization on the parameters. The filters
        that are implicitly zeroed out can be explicitly pruned without any impact on the model accuracy (and might even improve in some cases).
        """
        device = next(self.parameters()).device
        self
        with torch.no_grad():
            new_stem = []
            input_validity = torch.ones(3)
            for i in range(0, len(self.stem), 3):
                input_size = sum(input_validity.int()).item()
                out_validity = abs(self.stem[i + 1].weight) > gamma_thresh
                out_size = sum(out_validity.int()).item()
                W, b = bn_fuse(self.stem[i], self.stem[i + 1])
                new_stem.append(nn.Conv2d(input_size, out_size, kernel_size=self.stem[i].kernel_size, stride=self.stem[i].stride, padding=self.stem[i].padding))
                new_stem.append(nn.ReLU(inplace=True))
                new_stem[-2].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(input_validity).squeeze()), 0, torch.nonzero(out_validity).squeeze()))
                new_stem[-2].bias.copy_(b[out_validity])
                input_validity = out_validity.clone().detach()
                if verbose:
                    None
            self.stem = nn.Sequential(*new_stem)
            new_features = []
            skip_validity = 0
            for i in range(len(self.features)):
                inp = int(sum(input_validity.int()).item())
                if self.features[i].isFirst:
                    skip = 0
                a_validity = abs(self.features[i].conv1[1].weight) > gamma_thresh
                b_validity = abs(self.features[i].conv2[1].weight) > gamma_thresh
                c_validity = abs(self.features[i].conv3[1].weight) > gamma_thresh
                d_validity = abs(self.features[i].conv4[1].weight) > gamma_thresh
                e_validity = abs(self.features[i].conv5[1].weight) > gamma_thresh
                out_validity = abs(self.features[i].conv6[1].weight) > gamma_thresh
                new_features.append(SelecSLSBlockFused(inp, skip, int(sum(a_validity.int()).item()), int(sum(b_validity.int()).item()), int(sum(c_validity.int()).item()), int(sum(d_validity.int()).item()), int(sum(e_validity.int()).item()), int(sum(out_validity.int()).item()), self.features[i].isFirst, self.features[i].stride))
                i_validity = input_validity.clone().detach()
                o_validity = a_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv1[0], self.features[i].conv1[1])
                new_features[i].conv1[0].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv1[0].bias.copy_(bias[o_validity])
                if verbose:
                    None
                i_validity = o_validity.clone().detach()
                o_validity = b_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv2[0], self.features[i].conv2[1])
                new_features[i].conv2[0].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv2[0].bias.copy_(bias[o_validity])
                if verbose:
                    None
                i_validity = o_validity.clone().detach()
                o_validity = c_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv3[0], self.features[i].conv3[1])
                new_features[i].conv3[0].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv3[0].bias.copy_(bias[o_validity])
                if verbose:
                    None
                i_validity = o_validity.clone().detach()
                o_validity = d_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv4[0], self.features[i].conv4[1])
                new_features[i].conv4[0].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv4[0].bias.copy_(bias[o_validity])
                if verbose:
                    None
                i_validity = o_validity.clone().detach()
                o_validity = e_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv5[0], self.features[i].conv5[1])
                new_features[i].conv5[0].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv5[0].bias.copy_(bias[o_validity])
                if verbose:
                    None
                i_validity = torch.cat([a_validity.clone().detach(), c_validity.clone().detach(), e_validity.clone().detach()], 0)
                if self.features[i].isFirst:
                    skip = int(sum(out_validity.int()).item())
                    skip_validity = out_validity.clone().detach()
                else:
                    i_validity = torch.cat([i_validity, skip_validity], 0)
                o_validity = out_validity.clone().detach()
                W, bias = bn_fuse(self.features[i].conv6[0], self.features[i].conv6[1])
                new_features[i].conv6[0].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(i_validity).squeeze()), 0, torch.nonzero(o_validity).squeeze()))
                new_features[i].conv6[0].bias.copy_(bias[o_validity])
                if verbose:
                    None
                input_validity = out_validity.clone().detach()
            self.features = nn.Sequential(*new_features)
            new_head = []
            for i in range(len(self.head)):
                input_size = int(sum(input_validity.int()).item())
                out_validity = abs(self.head[i][1].weight) > gamma_thresh
                out_size = int(sum(out_validity.int()).item())
                W, b = bn_fuse(self.head[i][0], self.head[i][1])
                new_head.append(nn.Conv2d(input_size, out_size, kernel_size=self.head[i][0].kernel_size, stride=self.head[i][0].stride, padding=self.head[i][0].padding))
                new_head.append(nn.ReLU(inplace=True))
                new_head[-2].weight.copy_(torch.index_select(torch.index_select(W, 1, torch.nonzero(input_validity).squeeze()), 0, torch.nonzero(out_validity).squeeze()))
                new_head[-2].bias.copy_(b[out_validity])
                input_validity = out_validity.clone().detach()
                if verbose:
                    None
            self.head = nn.Sequential(*new_head)
            new_classifier = []
            new_classifier.append(nn.Linear(int(sum(input_validity.int()).item()), self.classifier[0].weight.shape[0]))
            new_classifier[0].weight.copy_(torch.index_select(self.classifier[0].weight, 1, torch.nonzero(input_validity).squeeze()))
            new_classifier[0].bias.copy_(self.classifier[0].bias)
            self.classifier = nn.Sequential(*new_classifier)
        self


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_mehtadushy_SelecSLS_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

