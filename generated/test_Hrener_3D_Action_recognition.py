import sys
_module = sys.modules[__name__]
del sys
data_generated = _module
model1 = _module
model1train = _module
model2 = _module
model2train = _module

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


import random


import matplotlib.pyplot as plt


import numpy as np


from torch.autograd import Variable


from torch.utils.data import Dataset


import torch.nn as nn


from torch.utils.data import DataLoader


class ConVNet(nn.Module):

    def __init__(self, in_dim=25, out_dim=30):
        super(ConVNet, self).__init__()
        self.transform = nn.Linear(in_dim, out_dim, bias=False)
        self.ConVNet_up = nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3), torch.nn.MaxPool2d(stride=2, kernel_size=2), torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), torch.nn.ReLU(), torch.nn.MaxPool2d(stride=2, kernel_size=2), torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
        self.ConVNet_down = nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3), torch.nn.MaxPool2d(stride=2, kernel_size=2), torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), torch.nn.ReLU(), torch.nn.MaxPool2d(stride=2, kernel_size=2), torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
        self.fc1 = nn.Sequential(nn.Linear(2 * 128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(256, 60))

    def forward(self, frame_0, frame_1, diff_0, diff_1):
        batch_n_frames = frame_0.shape[0]
        frame_0 = frame_0.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        frame_1 = frame_1.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        diff_0 = diff_0.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        diff_1 = diff_1.permute(0, 1, 3, 2).reshape(batch_n_frames * 30 * 3, 25)
        frame_0 = self.transform(frame_0).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        frame_1 = self.transform(frame_1).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        diff_0 = self.transform(diff_0).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        diff_1 = self.transform(diff_1).reshape(batch_n_frames, 30, 3, 30).permute(0, 2, 1, 3)
        frame_0_feature_maps = self.ConVNet_up(frame_0)
        frame_1_feature_maps = self.ConVNet_up(frame_1)
        diff_0_feature_maps = self.ConVNet_down(diff_0)
        diff_1_feature_maps = self.ConVNet_down(diff_1)
        frame_feature_maps = torch.max(frame_0_feature_maps, frame_1_feature_maps)
        diff_feature_maps = torch.max(diff_0_feature_maps, diff_1_feature_maps)
        feature_maps = torch.cat((frame_feature_maps, diff_feature_maps), 1)
        feature_maps = feature_maps.view(-1, 4096)
        output = self.fc2(self.fc1(feature_maps))
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

