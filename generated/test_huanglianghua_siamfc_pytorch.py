import sys
_module = sys.modules[__name__]
del sys
siamfc = _module
backbones = _module
datasets = _module
heads = _module
losses = _module
ops = _module
siamfc = _module
transforms = _module
demo = _module
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


import torch.nn as nn


import numpy as np


from torch.utils.data import Dataset


import torch.nn.functional as F


import torch


import torch.optim as optim


import time


from collections import namedtuple


from torch.optim.lr_scheduler import ExponentialLR


from torch.utils.data import DataLoader


import numbers


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(num_features, *args, eps=1e-06, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 2), _BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, groups=2), _BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1), _BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, groups=2), _BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, groups=2))


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 11, 2), _BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, groups=2), _BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1), _BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, groups=2), _BatchNorm2d(384), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 192, 11, 2), _BatchNorm2d(192), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(192, 512, 5, 1), _BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(512, 768, 3, 1), _BatchNorm2d(768), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(768, 768, 3, 1), _BatchNorm2d(768), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(768, 512, 3, 1), _BatchNorm2d(512))


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mask = target == 1
        neg_mask = target == 0
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(input, target, weight, reduction='sum')


def log_minus_sigmoid(x):
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + 0.5 * torch.clamp(x, min=0, max=0)


def log_sigmoid(x):
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + 0.5 * torch.clamp(x, min=0, max=0)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)
        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)
        loss = -(target * pos_weight * pos_log_sig + (1 - target) * neg_weight * neg_log_sig)
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        loss /= avg_weight.mean()
        return loss.mean()


class GHMCLoss(nn.Module):

    def __init__(self, bins=30, momentum=0.5):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [(t / bins) for t in range(bins + 1)]
        self.edges[-1] += 1e-06
        if momentum > 0:
            self.acc_sum = [(0.0) for _ in range(bins)]

    def forward(self, input, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)
        g = torch.abs(input.sigmoid().detach() - target)
        tot = input.numel()
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights /= weights.mean()
        loss = F.binary_cross_entropy_with_logits(input, target, weights, reduction='sum') / tot
        return loss


class OHNMLoss(nn.Module):

    def __init__(self, neg_ratio=3.0):
        super(OHNMLoss, self).__init__()
        self.neg_ratio = neg_ratio

    def forward(self, input, target):
        pos_logits = input[target > 0]
        pos_labels = target[target > 0]
        neg_logits = input[target == 0]
        neg_labels = target[target == 0]
        pos_num = pos_logits.numel()
        neg_num = int(pos_num * self.neg_ratio)
        neg_logits, neg_indices = neg_logits.topk(neg_num)
        neg_labels = neg_labels[neg_indices]
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_logits, neg_logits]), torch.cat([pos_labels, neg_labels]), reduction='mean')
        return loss


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNetV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (AlexNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (AlexNetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (BalancedLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GHMCLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiamFC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_huanglianghua_siamfc_pytorch(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

