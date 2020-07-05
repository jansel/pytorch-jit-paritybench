import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
pointnet_cls = _module
pointnet_cls_basic = _module
pointnet_seg = _module
transform_nets = _module
pointnet_part_seg = _module
test = _module
train = _module
provider = _module
batch_inference = _module
collect_indoor3d_data = _module
eval_iou_accuracy = _module
gen_indoor3d_h5 = _module
indoor3d_util = _module
model = _module
train_pytorch = _module
data_prep_util = _module
data_utils = _module
eulerangles = _module
model = _module
pc_util = _module
plyfile = _module
tf_util = _module
util_funcs = _module
util_layers = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import numpy as np


import random


import time


import torch


from torch import nn


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


from torch import FloatTensor


from typing import Tuple


from typing import Callable


from typing import Optional


from typing import Union


AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)


NUM_CLASS = 40


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(AbbPointCNN(32, 64, 8, 2, -1), AbbPointCNN(64, 96, 8, 4, -1), AbbPointCNN(96, 128, 12, 4, 120), AbbPointCNN(128, 160, 12, 6, 120))
        self.fcn = nn.Sequential(Dense(160, 128), Dense(128, 64, drop_rate=0.5), Dense(64, NUM_CLASS, with_bn=False, activation=None))

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            None
            k = make_dot(x[1])
            None
            k.view()
            None
            assert False
        x = self.pcnn2(x)[1]
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


def EndChannels(f, make_contiguous=False):
    """ Class decorator to apply 2D convolution along end channels. """


    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.f(x)
            x = x.permute(0, 2, 3, 1)
            return x
    return WrappedLayer()


class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
    """

    def __init__(self, N: int, dim: int, *args, **kwargs) ->None:
        """
        :param N: Batch size.
        :param D: Dimensions.
        """
        super(LayerNorm, self).__init__()
        if dim == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError('Dimensionality %i not supported' % dim)
        self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)

