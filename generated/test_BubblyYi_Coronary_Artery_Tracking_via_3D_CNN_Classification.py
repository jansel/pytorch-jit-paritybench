import sys
_module = sys.modules[__name__]
del sys
centerline_train_tools = _module
centerline_train_tools = _module
centerline_trainner = _module
data_provider_argu = _module
centerline_patch_generater_no_offset = _module
centerline_patch_generater_offset = _module
creat_spacinginfo_data_tool = _module
ostiapoints_patch_generater_negative = _module
ostiapoints_patch_generater_positive = _module
seedpoints_patch_generater_negative = _module
seedpoints_patch_generater_postive = _module
utils = _module
infer_tools_tree = _module
build_vessel_tree = _module
generate_seeds_ositas = _module
metric = _module
dijstra = _module
metirc = _module
setting = _module
tree = _module
utils = _module
vessels_tree_infer = _module
models = _module
centerline_net = _module
ostiapoints_net = _module
seedspoints_net = _module
ostia_net_data_provider_aug = _module
ostia_train_tools = _module
ostia_trainner = _module
seeds_net_data_provider_aug = _module
seeds_train_tools = _module
seeds_trainner = _module

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


import matplotlib


import matplotlib.pyplot as plt


from torch.utils.data import DataLoader


from time import time


from torch.utils.data import Dataset


import pandas as pd


import numpy as np


import random


import copy


import warnings


from scipy.ndimage.interpolation import zoom


import math


import torch.nn as nn


from scipy.ndimage import map_coordinates


class conv_block(nn.Module):

    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size, dilation=dilation), nn.BatchNorm3d(chann_out), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class CenterlineNet(nn.Module):

    def __init__(self, n_classes=100):
        super(CenterlineNet, self).__init__()
        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.n_classes = n_classes
        self.layer7 = nn.Conv3d(64, n_classes + 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out


class OstiapointsNet(nn.Module):

    def __init__(self):
        super(OstiapointsNet, self).__init__()
        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.layer7 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out


class SeedspointsNet(nn.Module):

    def __init__(self):
        super(SeedspointsNet, self).__init__()
        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.layer7 = nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CenterlineNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (OstiapointsNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (SeedspointsNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (conv_block,
     lambda: ([], {'chann_in': 4, 'chann_out': 4, 'k_size': 4, 'stride': 1, 'p_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
]

class Test_BubblyYi_Coronary_Artery_Tracking_via_3D_CNN_Classification(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

