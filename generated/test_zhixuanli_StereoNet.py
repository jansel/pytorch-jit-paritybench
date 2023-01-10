import sys
_module = sys.modules[__name__]
del sys
KITTILoader = _module
KITTIloader2015 = _module
SceneFlowLoader = _module
dataloader = _module
listflowfile = _module
preprocess = _module
readpfm = _module
models = _module
stereonet = _module
sceneflow = _module
utils = _module
cost_volume = _module

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


import torch.nn.parallel


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


import torch.nn.functional as F


import time


from torch.optim import RMSprop


import math


def CostVolume(input_feature, candidate_feature, position='left', method='subtract', k=4, batch_size=4, channel=32, D=192, H=256, W=512):
    """
    Some parameters:
        position
            means whether the input feature img is left or right
        k
            the conv counts of the first stage, the feature extraction stage
    """
    origin = input_feature
    candidate = candidate_feature
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to left and pad in right"""
    if position == 'left':
        leftMinusRightMove_List = []
        for disparity in range(D // 2 ** k):
            if disparity == 0:
                if method == 'subtract':
                    """ subtract method"""
                    leftMinusRightMove = origin - candidate
                else:
                    """ concat mathod """
                    leftMinusRightMove = torch.cat((origin, candidate), 1)
                leftMinusRightMove_List.append(leftMinusRightMove)
            else:
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                zero_padding = zero_padding
                left_move = torch.cat((origin, zero_padding), 3)
                if method == 'subtract':
                    """ subtract method"""
                    leftMinusRightMove = left_move[:, :, :, :origin.shape[3]] - candidate
                else:
                    """ concat mathod """
                    leftMinusRightMove = torch.cat((left_move[:, :, :, :origin.shape[3]], candidate), 1)
                leftMinusRightMove_List.append(leftMinusRightMove)
        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)
        return cost_volume


class MetricBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out


def soft_argmin(cost_volume):
    """Remove single-dimensional entries from the shape of an array."""
    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(-cost_volume)
    d_grid = torch.arange(cost_volume.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[2], cost_volume.shape[3]))
    d_grid = d_grid
    tmp = disparity_softmax * d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)
    return arg_soft_min


class StereoNet(nn.Module):

    def __init__(self, batch_size, cost_volume_method):
        super(StereoNet, self).__init__()
        self.batch_size = batch_size
        self.cost_volume_method = cost_volume_method
        cost_volume_channel = 32
        if cost_volume_method == 'subtract':
            cost_volume_channel = 32
        elif cost_volume_method == 'concat':
            cost_volume_channel = 64
        else:
            None
        self.downsampling = nn.Sequential(nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.Conv2d(32, 32, 5, stride=2, padding=2))
        self.res = nn.Sequential(ResBlock(32, 32), ResBlock(32, 32), ResBlock(32, 32), ResBlock(32, 32), ResBlock(32, 32), ResBlock(32, 32), nn.Conv2d(32, 32, 3, 1, 1))
        """ using 3d conv to instead the Euclidean distance"""
        self.cost_volume_filter = nn.Sequential(MetricBlock(cost_volume_channel, 32), MetricBlock(32, 32), MetricBlock(32, 32), MetricBlock(32, 32), nn.Conv3d(32, 1, 3, padding=1))
        self.refine = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), ResBlock(32, 32, dilation=1), ResBlock(32, 32, dilation=2), ResBlock(32, 32, dilation=4), ResBlock(32, 32, dilation=8), ResBlock(32, 32, dilation=1), ResBlock(32, 32, dilation=1), nn.Conv2d(32, 1, 3, padding=1))

    def forward_once_1(self, x):
        output = self.downsampling(x)
        output = self.res(output)
        return output

    def forward_stage1(self, input_l, input_r):
        output_l = self.forward_once_1(input_l)
        output_r = self.forward_once_1(input_r)
        return output_l, output_r

    def forward_once_2(self, cost_volume):
        """the index cost volume's dimension is not right for conv3d here, so we change it"""
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])
        output = self.cost_volume_filter(cost_volume)
        disparity_low = output
        return disparity_low

    def forward_stage2(self, feature_l, feature_r):
        cost_v_l = CostVolume(feature_l, feature_r, 'left', method=self.cost_volume_method, k=4, batch_size=self.batch_size)
        disparity_low = self.forward_once_2(cost_v_l)
        disparity_low = torch.squeeze(disparity_low, dim=1)
        return disparity_low

    def forward_stage3(self, disparity_low, left):
        """upsample and concatenate"""
        d_high = nn.functional.interpolate(disparity_low, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_high = soft_argmin(d_high)
        d_concat = torch.cat([d_high, left], dim=1)
        d_refined = self.refine(d_concat)
        return d_refined

    def forward(self, left, right):
        left_feature, right_feature = self.forward_stage1(left, right)
        disparity_low_l = self.forward_stage2(left_feature, right_feature)
        d_initial_l = nn.functional.interpolate(disparity_low_l, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        d_initial_l = soft_argmin(d_initial_l)
        d_refined_l = self.forward_stage3(disparity_low_l, left)
        d_final_l = d_initial_l + d_refined_l
        d_final_l = nn.ReLU()(d_final_l)
        return d_final_l


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MetricBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zhixuanli_StereoNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

