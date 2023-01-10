import sys
_module = sys.modules[__name__]
del sys
load_data = _module
process = _module
unprocess = _module
dnd_denoise = _module
metrics = _module
models = _module
generator = _module
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


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import numpy as np


import torch.nn as nn


import torch.distributions as tdist


from torchvision import transforms


from torchvision import utils


from torch.autograd import Variable


import torch.utils.data as Data


import torch.nn.functional as F


import torchvision


import time


import torch.optim as optim


from time import sleep


from torchvision import utils as vutils


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.downsampleby2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.upsampleby2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1))

    def forward(self, input_img, input_variance):
        input_1 = input_img
        input_2 = input_variance
        input_cat = torch.cat((input_1, input_2), 1)
        skips = []
        feats = self.conv1(input_cat)
        skips.append(feats)
        feats = self.downsampleby2(feats)
        feats = self.conv2(feats)
        skips.append(feats)
        feats = self.downsampleby2(feats)
        feats = self.conv3(feats)
        skips.append(feats)
        feats = self.downsampleby2(feats)
        feats = self.conv4(feats)
        skips.append(feats)
        feats = self.downsampleby2(feats)
        feats = self.conv5(feats)
        feats = self.upsampleby2(feats)
        feats = torch.cat((feats, skips.pop()), 1)
        feats = self.conv6(feats)
        feats = self.upsampleby2(feats)
        feats = torch.cat((feats, skips.pop()), 1)
        feats = self.conv7(feats)
        feats = self.upsampleby2(feats)
        feats = torch.cat((feats, skips.pop()), 1)
        feats = self.conv8(feats)
        feats = self.upsampleby2(feats)
        feats = torch.cat((feats, skips.pop()), 1)
        feats = self.conv9(feats)
        residual = self.conv10(feats)
        return input_1 + residual

