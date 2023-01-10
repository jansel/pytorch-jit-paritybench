import sys
_module = sys.modules[__name__]
del sys
dataloader_utils = _module
eval = _module
STNModule = _module
SVHNet = _module
train = _module
utils = _module
vis_utils = _module

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


import numpy as np


import torch


import torchvision.datasets as dsets


import torchvision.transforms as transforms


from torch.utils.data.sampler import SubsetRandomSampler


import logging


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import torch.optim as optim


class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """

    def __init__(self, in_channels, spatial_dims, kernel_size, use_dropout=False):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(32 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        batch_images = x
        x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        None
        x = x.view(-1, 32 * 4 * 4)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)
        x = x.view(-1, 2, 3)
        None
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert affine_grid_points.size(0) == batch_images.size(0), 'The batch sizes of the input images must be same as the generated grid.'
        rois = F.grid_sample(batch_images, affine_grid_points)
        None
        return rois, affine_grid_points


class BaseSVHNet(nn.Module):
    """
    Base SVHN Net to be trained
    """

    def __init__(self, in_channels, kernel_size, num_classes=10, use_dropout=False):
        super(BaseSVHNet, self).__init__()
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.ncls = num_classes
        self.dropout = use_dropout
        self.drop_prob = 0.5
        self.stride = 1
        self.conv1 = nn.Conv2d(self._in_ch, 32, kernel_size=self._ksize, stride=self.stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(32 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, self.ncls)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        None
        x = x.view(-1, 32 * 4 * 4)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
        else:
            x = self.fc1(x)
        x = self.fc2(x)
        return x


class STNSVHNet(nn.Module):

    def __init__(self, spatial_dim, in_channels, stn_kernel_size, kernel_size, num_classes=10, use_dropout=False):
        super(STNSVHNet, self).__init__()
        self._in_ch = in_channels
        self._ksize = kernel_size
        self._sksize = stn_kernel_size
        self.ncls = num_classes
        self.dropout = use_dropout
        self.drop_prob = 0.5
        self.stride = 1
        self.spatial_dim = spatial_dim
        self.stnmod = STNModule.SpatialTransformer(self._in_ch, self.spatial_dim, self._sksize)
        self.conv1 = nn.Conv2d(self._in_ch, 32, kernel_size=self._ksize, stride=self.stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(128 * 4 * 4, 3092)
        self.fc2 = nn.Linear(3092, self.ncls)

    def forward(self, x):
        rois, affine_grid = self.stnmod(x)
        out = F.relu(self.conv1(rois))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = out.view(-1, 128 * 4 * 4)
        if self.dropout:
            out = F.dropout(self.fc1(out), p=0.5)
        else:
            out = self.fc1(out)
        out = self.fc2(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseSVHNet,
     lambda: ([], {'in_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
]

class Test_aicaffeinelife_Pytorch_STN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

