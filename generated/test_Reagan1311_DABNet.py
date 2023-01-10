import sys
_module = sys.modules[__name__]
del sys
builders = _module
dataset_builder = _module
model_builder = _module
dataset = _module
camvid = _module
cityscapes = _module
eval_fps = _module
DABNet = _module
predict = _module
test = _module
train = _module
colorize_mask = _module
convert_state = _module
loss = _module
lr_scheduler = _module
metric = _module
trainID2labelID = _module
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


from torch.utils import data


import numpy as np


import random


import time


import torch


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import matplotlib.pyplot as plt


import math


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import _LRScheduler


class BNPReLU(nn.Module):

    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=0.001)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output


class DABModule(nn.Module):

    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        return output + input


class DownSamplingBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut
        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
        output = self.bn_prelu(output)
        return output


class InputInjection(nn.Module):

    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class DABNet(nn.Module):

    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(Conv(3, 32, 3, 2, padding=1, bn_acti=True), Conv(32, 32, 3, 1, padding=1, bn_acti=True), Conv(32, 32, 3, 1, padding=1, bn_acti=True))
        self.down_1 = InputInjection(1)
        self.down_2 = InputInjection(2)
        self.down_3 = InputInjection(3)
        self.bn_prelu_1 = BNPReLU(32 + 3)
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module('DAB_Module_1_' + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module('DAB_Module_2_' + str(i), DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)
        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 = self.init_conv(input)
        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))
        out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        return out


class CrossEntropyLoss2d(nn.Module):
    """
    This file defines a cross entropy loss for 2D images
    """

    def __init__(self, weight=None, ignore_label=255):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        """
        super().__init__()
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class FocalLoss2d(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -(1 - pt) ** self.gamma * self.alpha * logpt
        return loss


class ProbOhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256, down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        return self.criterion(pred, target)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNPReLU,
     lambda: ([], {'nIn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DABModule,
     lambda: ([], {'nIn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DABNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DownSamplingBlock,
     lambda: ([], {'nIn': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InputInjection,
     lambda: ([], {'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Reagan1311_DABNet(_paritybench_base):
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

