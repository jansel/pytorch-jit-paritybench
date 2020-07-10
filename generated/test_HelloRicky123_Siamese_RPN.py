import sys
_module = sys.modules[__name__]
del sys
create_dataset_ytbid = _module
create_lmdb = _module
test_OTB = _module
train_siamrpn = _module
lib = _module
custom_transforms = _module
generate_anchors = _module
loss = _module
utils = _module
visual = _module
net = _module
config = _module
dataset = _module
network = _module
run_SiamRPN = _module
tracker = _module
train = _module

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


import numpy as np


import torch.nn


import torch.nn.functional as F


import random


import time


import functools


from torch.multiprocessing import Pool


from torch.multiprocessing import Manager


from torch.utils.data.dataset import Dataset


import torchvision.transforms as transforms


from torchvision.models import alexnet


from torch.autograd import Variable


from torch import nn


import torch.optim as optim


import torchvision


import torch.nn as nn


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


from collections import OrderedDict


class SiameseAlexNet(nn.Module):

    def __init__(self):
        super(SiameseAlexNet, self).__init__()
        self.featureExtract = nn.Sequential(nn.Conv2d(3, 96, 11, stride=2), nn.BatchNorm2d(96), nn.MaxPool2d(3, stride=2), nn.ReLU(inplace=True), nn.Conv2d(96, 256, 5), nn.BatchNorm2d(256), nn.MaxPool2d(3, stride=2), nn.ReLU(inplace=True), nn.Conv2d(256, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 384, 3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3), nn.BatchNorm2d(256))
        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        detection_feature = self.featureExtract(detection)
        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        score_filters = kernel_score.reshape(-1, 256, 4, 4)
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        reg_filters = kernel_regression.reshape(-1, 256, 4, 4)
        pred_regression = self.regress_adjust(F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)
        template_feature = self.featureExtract(template)
        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.score_filters = kernel_score.reshape(-1, 256, 4, 4)
        self.reg_filters = kernel_regression.reshape(-1, 256, 4, 4)

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1, self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1, self.score_displacement + 1))
        return pred_score, pred_regression

