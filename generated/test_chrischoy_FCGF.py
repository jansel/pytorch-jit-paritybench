import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
config = _module
demo = _module
lib = _module
data_loaders = _module
eval = _module
metrics = _module
timer = _module
trainer = _module
transforms = _module
model = _module
common = _module
residual_block = _module
resunet = _module
simpleunet = _module
benchmark_3dmatch = _module
benchmark_util = _module
test_kitti = _module
train = _module
util = _module
file = _module
misc = _module
pointcloud = _module
trajectory = _module
transform_estimation = _module
visualization = _module

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


import logging


import random


import torch.utils.data


from scipy.linalg import expm


from scipy.linalg import norm


from scipy.spatial import cKDTree


import torch.functional as F


import torch.optim as optim


import torch.nn.functional as F


import torch.nn as nn


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, dimension=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = MEF.relu(out)
        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = 'IN'

