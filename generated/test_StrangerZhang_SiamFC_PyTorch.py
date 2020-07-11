import sys
_module = sys.modules[__name__]
del sys
convert_pretrained_model = _module
create_dataset = _module
create_lmdb = _module
demo_siamfc = _module
run_SiamFC = _module
train_siamfc = _module
siamfc = _module
alexnet = _module
config = _module
custom_transforms = _module
dataset = _module
tracker = _module
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


import torch


import re


import numpy as np


from scipy import io as sio


import torch.nn.functional as F


import torchvision.transforms as transforms


from torchvision.models import alexnet


from torch.autograd import Variable


from torch import nn


from torch.utils.data.dataset import Dataset


import time


import warnings


import torch.optim as optim


import torchvision


import pandas as pd


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


class SiameseAlexNet(nn.Module):

    def __init__(self, gpu_id, train=True):
        super(SiameseAlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 96, 11, 2), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(96, 256, 5, 1, groups=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(256, 384, 3, 1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 384, 3, 1, groups=2), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3, 1, groups=2))
        self.corr_bias = nn.Parameter(torch.zeros(1))
        if train:
            gt, weight = self._create_gt_mask((config.train_response_sz, config.train_response_sz))
            with torch.device(gpu_id):
                self.train_gt = torch.from_numpy(gt)
                self.train_weight = torch.from_numpy(weight)
            gt, weight = self._create_gt_mask((config.response_sz, config.response_sz))
            with torch.device(gpu_id):
                self.valid_gt = torch.from_numpy(gt)
                self.valid_weight = torch.from_numpy(weight)
        self.exemplar = None
        self.gpu_id = gpu_id

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar, instance = x
        if exemplar is not None and instance is not None:
            batch_size = exemplar.shape[0]
            exemplar = self.features(exemplar)
            instance = self.features(instance)
            score_map = []
            N, C, H, W = instance.shape
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, exemplar, groups=N) * config.response_scale + self.corr_bias
            return score.transpose(0, 1)
        elif exemplar is not None and instance is None:
            self.exemplar = self.features(exemplar)
            self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)
        else:
            instance = self.features(instance)
            N, _, H, W = instance.shape
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, self.exemplar, groups=N)
            return score.transpose(0, 1)

    def loss(self, pred):
        return F.binary_cross_entropy_with_logits(pred, self.gt)

    def weighted_loss(self, pred):
        if self.training:
            return F.binary_cross_entropy_with_logits(pred, self.train_gt, self.train_weight, reduction='sum') / config.train_batch_size
        else:
            return F.binary_cross_entropy_with_logits(pred, self.valid_gt, self.valid_weight, reduction='sum') / config.train_batch_size

    def _create_gt_mask(self, shape):
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
        x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x ** 2 + y ** 2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[(np.newaxis), :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, (np.newaxis), :, :]
        return mask.astype(np.float32), weights.astype(np.float32)

