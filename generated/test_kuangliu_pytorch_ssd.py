import sys
_module = sys.modules[__name__]
del sys
datagen = _module
encoder = _module
multibox_layer = _module
multibox_loss = _module
convert_vgg = _module
convert_voc = _module
ssd = _module
test = _module
train = _module
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


import random


import numpy as np


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import math


import itertools


import torch.nn as nn


import torch.nn.init as init


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision


import torch.optim as optim


import torch.backends.cudnn as cudnn


import time


class MultiBoxLayer(nn.Module):
    num_classes = 21
    num_anchors = [4, 6, 6, 6, 4, 4]
    in_planes = [512, 1024, 512, 256, 256, 256]

    def __init__(self):
        super(MultiBoxLayer, self).__init__()
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 21, kernel_size=3, padding=1))

    def forward(self, xs):
        """
        Args:
          xs: (list) of tensor containing intermediate layer outputs.

        Returns:
          loc_preds: (tensor) predicted locations, sized [N,8732,4].
          conf_preds: (tensor) predicted class confidences, sized [N,8732,21].
        """
        y_locs = []
        y_confs = []
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)
            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, 21)
            y_confs.append(y_conf)
        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds


class MultiBoxLoss(nn.Module):
    num_classes = 21

    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def cross_entropy_loss(self, x, y):
        """Cross entropy loss w/o averaging across all samples.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) cross entroy loss, sized [N,].
        """
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1, 1))

    def test_cross_entropy_loss(self):
        a = Variable(torch.randn(10, 4))
        b = Variable(torch.ones(10).long())
        loss = self.cross_entropy_loss(a, b)
        None
        None

    def hard_negative_mining(self, conf_loss, pos):
        """Return negative indices that is 3x the number as postive indices.

        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and conf_targets, sized [N*8732,].
          pos: (tensor) positive(matched) box indices, sized [N,8732].

        Return:
          (tensor) negative indices, sized [N,8732].
        """
        batch_size, num_boxes = pos.size()
        conf_loss[pos] = 0
        conf_loss = conf_loss.view(batch_size, -1)
        _, idx = conf_loss.sort(1, descending=True)
        _, rank = idx.sort(1)
        num_pos = pos.long().sum(1)
        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)
        neg = rank < num_neg.expand_as(rank)
        return neg

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        """Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, 8732, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, 8732, 4].
          conf_preds: (tensor) predicted class confidences, sized [batch_size, 8732, num_classes].
          conf_targets: (tensor) encoded target classes, sized [batch_size, 8732].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(conf_preds, conf_targets).
        """
        batch_size, num_boxes, _ = loc_preds.size()
        pos = conf_targets > 0
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.Tensor([0]))
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)
        pos_loc_preds = loc_preds[pos_mask].view(-1, 4)
        pos_loc_targets = loc_targets[pos_mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)
        conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes), conf_targets.view(-1))
        neg = self.hard_negative_mining(conf_loss, pos)
        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
        mask = (pos_mask + neg_mask).gt(0)
        pos_and_neg = (pos + neg).gt(0)
        preds = conf_preds[mask].view(-1, self.num_classes)
        targets = conf_targets[pos_and_neg]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        loc_loss /= num_matched_boxes
        conf_loss /= num_matched_boxes
        None
        return loc_loss + conf_loss


class L2Norm2d(nn.Module):
    """L2Norm layer across all channels."""

    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        """out = scale * x / sqrt(\\sum x_i^2)"""
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)


class SSD300(nn.Module):
    input_size = 300

    def __init__(self):
        super(SSD300, self).__init__()
        self.base = self.VGG16()
        self.norm4 = L2Norm2d(20)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
        self.multibox = MultiBoxLayer()

    def forward(self, x):
        hs = []
        h = self.base(x)
        hs.append(self.norm4(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)
        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)
        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)
        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)
        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)
        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds

    def VGG16(self):
        """VGG16 layers."""
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (L2Norm2d,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kuangliu_pytorch_ssd(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

