import sys
_module = sys.modules[__name__]
del sys
bfm_model = _module
demo_4D = _module
demo_nicp = _module
io3d = _module
landmark = _module
local_affine = _module
nicp = _module
render = _module
shader = _module
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


import numpy as np


from scipy.io import loadmat


import time


import torch.nn as nn


import torch.sparse as sp


import torchvision


import torch.nn.functional as F


import matplotlib.pyplot as plt


class LocalAffine(nn.Module):

    def __init__(self, num_points, batch_size=1, edges=None):
        """
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        """
        super(LocalAffine, self).__init__()
        self.A = nn.Parameter(torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1))
        self.b = nn.Parameter(torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batch_size, num_points, 1, 1))
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        """
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix, 
        """
        if self.edges is None:
            raise Exception('edges cannot be none when calculate stiff')
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=3)
        w1 = torch.index_select(affine_weight, dim=1, index=idx1)
        w2 = torch.index_select(affine_weight, dim=1, index=idx2)
        w_diff = (w1 - w2) ** 2
        return w_diff

    def forward(self, x, pool_num=0, return_stiff=False):
        """
            x should have shape of B * N * 3
        """
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        if return_stiff:
            stiffness = self.stiffness()
            return out_x, stiffness
        else:
            return out_x

