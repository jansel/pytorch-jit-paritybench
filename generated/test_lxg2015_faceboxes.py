import sys
_module = sys.modules[__name__]
del sys
createWiderFace = _module
dataset = _module
encoderl = _module
multibox_layer = _module
multibox_loss = _module
networks = _module
predict = _module
trainvisdom = _module

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


from torch.autograd import Function


from torch.utils.data import DataLoader


from torchvision import models


class MultiBoxLayer(nn.Module):
    num_classes = 2
    num_anchors = [21, 1, 1]
    in_planes = [128, 256, 256]

    def __init__(self):
        super(MultiBoxLayer, self).__init__()
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i] * 2, kernel_size=3, padding=1))

    def forward(self, xs):
        """
		xs:list of 之前的featuremap list
		retrun: loc_preds: [N,21842,4]
				conf_preds:[N,24842,2]
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
            y_conf = y_conf.view(N, -1, 2)
            y_confs.append(y_conf)
        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds


class MultiBoxLoss(nn.Module):
    num_classes = 2

    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def cross_entropy_loss(self, x, y):
        x = x.detach()
        y = y.detach()
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1, keepdim=True)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1, 1))

    def hard_negative_mining(self, conf_loss, pos):
        """
		conf_loss [N*21482,]
		pos [N,21482]
		return negative indice
		"""
        batch_size, num_boxes = pos.size()
        conf_loss[pos.view(-1, 1)] = 0
        conf_loss = conf_loss.view(batch_size, -1)
        _, idx = conf_loss.sort(1, descending=True)
        _, rank = idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)
        neg = rank < num_neg.expand_as(rank)
        return neg

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        """
		loc_preds[batch,21842,4]
		loc_targets[batch,21842,4]
		conf_preds[batch,21842,2]
		conf_targets[batch,21842]
		"""
        batch_size, num_boxes, _ = loc_preds.size()
        pos = conf_targets > 0
        num_pos = pos.long().sum(1, keepdim=True)
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.Tensor([0]), requires_grad=True)
        pos_mask1 = pos.unsqueeze(2).expand_as(loc_preds)
        pos_loc_preds = loc_preds[pos_mask1].view(-1, 4)
        pos_loc_targets = loc_targets[pos_mask1].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)
        conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes), conf_targets.view(-1, 1))
        neg = self.hard_negative_mining(conf_loss, pos)
        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
        mask = (pos_mask + neg_mask).gt(0)
        pos_and_neg = (pos + neg).gt(0)
        preds = conf_preds[mask].view(-1, self.num_classes)
        targets = conf_targets[pos_and_neg]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        N = num_pos.data.sum()
        loc_loss /= N
        conf_loss /= N
        None
        return loc_loss + conf_loss


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride), nn.BatchNorm2d(out_channels), nn.ReLU(True))


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = conv_bn_relu(128, 32, 1)
        self.conv2 = conv_bn_relu(128, 32, 1)
        self.conv3 = conv_bn_relu(128, 24, 1)
        self.conv4 = conv_bn_relu(24, 32, 3, padding=1)
        self.conv5 = conv_bn_relu(128, 24, 1)
        self.conv6 = conv_bn_relu(24, 32, 3, padding=1)
        self.conv7 = conv_bn_relu(32, 32, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x)
        x3 = self.conv4(x3)
        x4 = self.conv5(x)
        x4 = self.conv6(x4)
        x4 = self.conv7(x4)
        output = torch.cat([x1, x2, x3, x4], 1)
        return output


class FaceBox(nn.Module):
    input_size = 1024

    def __init__(self):
        super(FaceBox, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()
        self.conv3_1 = conv_bn_relu(128, 128, 1)
        self.conv3_2 = conv_bn_relu(128, 256, 3, 2, 1)
        self.conv4_1 = conv_bn_relu(256, 128, 1)
        self.conv4_2 = conv_bn_relu(128, 256, 3, 2, 1)
        self.multilbox = MultiBoxLayer()

    def forward(self, x):
        hs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(torch.cat([x, -x], 1))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(torch.cat([x, -x], 1))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        hs.append(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        hs.append(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        hs.append(x)
        loc_preds, conf_preds = self.multilbox(hs)
        return loc_preds, conf_preds


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FaceBox,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Inception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
]

class Test_lxg2015_faceboxes(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

