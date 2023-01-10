import sys
_module = sys.modules[__name__]
del sys
coco_dataset = _module
entity = _module
face_detector = _module
gen_ignore_mask = _module
hand_detector = _module
CocoPoseNet = _module
FaceNet = _module
HandNet = _module
openpose = _module
pose_detect = _module
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


import math


import random


import numpy as np


import torch


from torch.utils.data import Dataset


from scipy.ndimage.filters import gaussian_filter


import torch.nn.functional as F


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import ReLU


from torch.nn import MaxPool2d


from torch.nn import init


import time


from torch.utils.data import DataLoader


from torch.optim import Adam


from matplotlib import pyplot as plt


class VGG_Base(Module):

    def __init__(self):
        super(VGG_Base, self).__init__()
        self.conv1_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        return x


class Base_model(Module):

    def __init__(self):
        super(Base_model, self).__init__()
        self.vgg_base = VGG_Base()
        self.conv4_3_CPM = Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_4_CPM = Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x


class Stage_1(Module):

    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L1 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = Conv2d(in_channels=512, out_channels=38, kernel_size=1, stride=1, padding=0)
        self.conv1_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L2 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L2 = Conv2d(in_channels=512, out_channels=19, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()

    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x))
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv1_CPM_L2(x))
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2


class Stage_x(Module):

    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv2_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv3_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv4_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv5_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv6_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7_L1 = Conv2d(in_channels=128, out_channels=38, kernel_size=1, stride=1, padding=0)
        self.conv1_L2 = Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv2_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv3_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv4_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv5_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.conv6_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7_L2 = Conv2d(in_channels=128, out_channels=19, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()

    def forward(self, x):
        h1 = self.relu(self.conv1_L1(x))
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        h2 = self.relu(self.conv1_L2(x))
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)
        return h1, h2


class CocoPoseNet(Module):
    insize = 368

    def __init__(self, path=None):
        super(CocoPoseNet, self).__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)
        if path:
            self.base.vgg_base.load_state_dict(torch.load(path))

    def forward(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps


class FaceNet(Module):
    insize = 368

    def __init__(self):
        super(FaceNet, self).__init__()
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3_CPM = Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_1_CPM = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv6_2_CPM = Conv2d(in_channels=512, out_channels=71, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage2 = Conv2d(in_channels=199, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage2 = Conv2d(in_channels=128, out_channels=71, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage3 = Conv2d(in_channels=199, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage3 = Conv2d(in_channels=128, out_channels=71, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage4 = Conv2d(in_channels=199, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage4 = Conv2d(in_channels=128, out_channels=71, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage5 = Conv2d(in_channels=199, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage5 = Conv2d(in_channels=128, out_channels=71, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage6 = Conv2d(in_channels=199, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage6 = Conv2d(in_channels=128, out_channels=71, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)

    def __call__(self, x):
        heatmaps = []
        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.relu(self.conv4_4(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3_CPM(h))
        feature_map = h
        h = self.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage2(h))
        h = self.relu(self.Mconv2_stage2(h))
        h = self.relu(self.Mconv3_stage2(h))
        h = self.relu(self.Mconv4_stage2(h))
        h = self.relu(self.Mconv5_stage2(h))
        h = self.relu(self.Mconv6_stage2(h))
        h = self.Mconv7_stage2(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage3(h))
        h = self.relu(self.Mconv2_stage3(h))
        h = self.relu(self.Mconv3_stage3(h))
        h = self.relu(self.Mconv4_stage3(h))
        h = self.relu(self.Mconv5_stage3(h))
        h = self.relu(self.Mconv6_stage3(h))
        h = self.Mconv7_stage3(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage4(h))
        h = self.relu(self.Mconv2_stage4(h))
        h = self.relu(self.Mconv3_stage4(h))
        h = self.relu(self.Mconv4_stage4(h))
        h = self.relu(self.Mconv5_stage4(h))
        h = self.relu(self.Mconv6_stage4(h))
        h = self.Mconv7_stage4(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage5(h))
        h = self.relu(self.Mconv2_stage5(h))
        h = self.relu(self.Mconv3_stage5(h))
        h = self.relu(self.Mconv4_stage5(h))
        h = self.relu(self.Mconv5_stage5(h))
        h = self.relu(self.Mconv6_stage5(h))
        h = self.Mconv7_stage5(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage6(h))
        h = self.relu(self.Mconv2_stage6(h))
        h = self.relu(self.Mconv3_stage6(h))
        h = self.relu(self.Mconv4_stage6(h))
        h = self.relu(self.Mconv5_stage6(h))
        h = self.relu(self.Mconv6_stage6(h))
        h = self.Mconv7_stage6(h)
        heatmaps.append(h)
        return heatmaps


class HandNet(Module):
    insize = 368

    def __init__(self):
        super(HandNet, self).__init__()
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3_CPM = Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6_1_CPM = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv6_2_CPM = Conv2d(in_channels=512, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage2 = Conv2d(in_channels=150, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage2 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage2 = Conv2d(in_channels=128, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage3 = Conv2d(in_channels=150, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage3 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage3 = Conv2d(in_channels=128, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage4 = Conv2d(in_channels=150, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage4 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage4 = Conv2d(in_channels=128, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage5 = Conv2d(in_channels=150, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage5 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage5 = Conv2d(in_channels=128, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.Mconv1_stage6 = Conv2d(in_channels=150, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage6 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage6 = Conv2d(in_channels=128, out_channels=22, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)

    def __call__(self, x):
        heatmaps = []
        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.relu(self.conv4_4(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3_CPM(h))
        feature_map = h
        h = self.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage2(h))
        h = self.relu(self.Mconv2_stage2(h))
        h = self.relu(self.Mconv3_stage2(h))
        h = self.relu(self.Mconv4_stage2(h))
        h = self.relu(self.Mconv5_stage2(h))
        h = self.relu(self.Mconv6_stage2(h))
        h = self.Mconv7_stage2(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage3(h))
        h = self.relu(self.Mconv2_stage3(h))
        h = self.relu(self.Mconv3_stage3(h))
        h = self.relu(self.Mconv4_stage3(h))
        h = self.relu(self.Mconv5_stage3(h))
        h = self.relu(self.Mconv6_stage3(h))
        h = self.Mconv7_stage3(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage4(h))
        h = self.relu(self.Mconv2_stage4(h))
        h = self.relu(self.Mconv3_stage4(h))
        h = self.relu(self.Mconv4_stage4(h))
        h = self.relu(self.Mconv5_stage4(h))
        h = self.relu(self.Mconv6_stage4(h))
        h = self.Mconv7_stage4(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage5(h))
        h = self.relu(self.Mconv2_stage5(h))
        h = self.relu(self.Mconv3_stage5(h))
        h = self.relu(self.Mconv4_stage5(h))
        h = self.relu(self.Mconv5_stage5(h))
        h = self.relu(self.Mconv6_stage5(h))
        h = self.Mconv7_stage5(h)
        heatmaps.append(h)
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(self.Mconv1_stage6(h))
        h = self.relu(self.Mconv2_stage6(h))
        h = self.relu(self.Mconv3_stage6(h))
        h = self.relu(self.Mconv4_stage6(h))
        h = self.relu(self.Mconv5_stage6(h))
        h = self.relu(self.Mconv6_stage6(h))
        h = self.Mconv7_stage6(h)
        heatmaps.append(h)
        return heatmaps


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Base_model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CocoPoseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Stage_1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (Stage_x,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 185, 64, 64])], {}),
     True),
    (VGG_Base,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_TreB1eN_Pytorch0_4_1_Openpose(_paritybench_base):
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

