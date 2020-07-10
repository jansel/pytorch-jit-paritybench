import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
feeder = _module
feeder = _module
ntu_gendata = _module
ntu_read_skeleton = _module
tools = _module
main = _module
HCN = _module
model = _module
resource = _module
utils = _module
utils = _module

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


from torch.autograd import Variable


from torchvision import transforms


import random


import torch.nn.functional as F


import logging


import torch.optim as optim


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torchvision


from collections import OrderedDict


class HCN(nn.Module):
    """
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    """

    def __init__(self, in_channel=3, num_joint=25, num_person=2, out_channel=64, window_size=64, num_class=60):
        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=num_joint, out_channels=out_channel // 2, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1), nn.Dropout2d(p=0.5), nn.MaxPool2d(2))
        self.conv1m = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv3m = nn.Sequential(nn.Conv2d(in_channels=num_joint, out_channels=out_channel // 2, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1), nn.Dropout2d(p=0.5), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 2, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Dropout2d(p=0.5), nn.MaxPool2d(2))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 4, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Dropout2d(p=0.5), nn.MaxPool2d(2))
        self.fc7 = nn.Sequential(nn.Linear(out_channel * 4 * (window_size // 16) * (window_size // 16), 256 * 2), nn.ReLU(), nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256 * 2, num_class)
        utils.initial_model_weight(layers=list(self.children()))
        None

    def forward(self, x, target=None):
        N, C, T, V, M = x.size()
        motion = x[:, :, 1:, :, :] - x[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.upsample(motion, size=(T, V), mode='bilinear', align_corners=False).contiguous().view(N, C, M, T, V).permute(0, 1, 3, 4, 2)
        logits = []
        for i in range(self.num_person):
            out = self.conv1(x[:, :, :, :, (i)])
            out = self.conv2(out)
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)
            out = self.conv1m(motion[:, :, :, :, (i)])
            out = self.conv2m(out)
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)
            out = torch.cat((out_p, out_m), dim=1)
            out = self.conv5(out)
            out = self.conv6(out)
            logits.append(out)
        out = torch.max(logits[0], logits[1])
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)
        t = out
        assert not (t != t).any()
        assert not t.abs().sum() == 0
        return out

