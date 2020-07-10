import sys
_module = sys.modules[__name__]
del sys
data = _module
demo = _module
deploy = _module
net = _module
train = _module
chg_model = _module
composite = _module
loss_draw = _module

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


import torch


import random


import numpy as np


from torchvision import transforms


import logging


import torch.nn as nn


import torch.nn.functional as F


import time


import math


from math import log10


import torch.optim as optim


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torchvision


import collections


class VGG16(nn.Module):

    def __init__(self, args):
        super(VGG16, self).__init__()
        self.stage = args.stage
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.deconv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=True)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2, bias=True)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2, bias=True)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2, bias=True)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=True)
        if args.stage == 2:
            for p in self.parameters():
                p.requires_grad = False
        if self.stage == 2 or self.stage == 3:
            self.refine_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x11 = F.relu(self.conv1_1(x))
        x12 = F.relu(self.conv1_2(x11))
        x1p, id1 = F.max_pool2d(x12, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x21 = F.relu(self.conv2_1(x1p))
        x22 = F.relu(self.conv2_2(x21))
        x2p, id2 = F.max_pool2d(x22, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x31 = F.relu(self.conv3_1(x2p))
        x32 = F.relu(self.conv3_2(x31))
        x33 = F.relu(self.conv3_3(x32))
        x3p, id3 = F.max_pool2d(x33, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x41 = F.relu(self.conv4_1(x3p))
        x42 = F.relu(self.conv4_2(x41))
        x43 = F.relu(self.conv4_3(x42))
        x4p, id4 = F.max_pool2d(x43, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x51 = F.relu(self.conv5_1(x4p))
        x52 = F.relu(self.conv5_2(x51))
        x53 = F.relu(self.conv5_3(x52))
        x5p, id5 = F.max_pool2d(x53, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        x61 = F.relu(self.conv6_1(x5p))
        x61d = F.relu(self.deconv6_1(x61))
        x5d = F.max_unpool2d(x61d, id5, kernel_size=2, stride=2)
        x51d = F.relu(self.deconv5_1(x5d))
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x41d = F.relu(self.deconv4_1(x4d))
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x31d = F.relu(self.deconv3_1(x3d))
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x21d = F.relu(self.deconv2_1(x2d))
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.deconv1_1(x1d))
        raw_alpha = self.deconv1(x12d)
        pred_mattes = F.sigmoid(raw_alpha)
        if self.stage <= 1:
            return pred_mattes, 0
        refine0 = torch.cat((x[:, :3, :, :], pred_mattes), 1)
        refine1 = F.relu(self.refine_conv1(refine0))
        refine2 = F.relu(self.refine_conv2(refine1))
        refine3 = F.relu(self.refine_conv3(refine2))
        pred_refine = self.refine_pred(refine3)
        pred_alpha = F.sigmoid(raw_alpha + pred_refine)
        return pred_mattes, pred_alpha

