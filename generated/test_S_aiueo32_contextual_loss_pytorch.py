import sys
_module = sys.modules[__name__]
del sys
contextual_loss = _module
config = _module
functional = _module
modules = _module
contextual = _module
contextual_bilateral = _module
vgg = _module
setup = _module
tests = _module
test_contextual = _module
test_contextual_bilateral = _module

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


import torch.nn.functional as F


import torch.nn as nn


from collections import namedtuple


import torchvision.models.vgg as vgg


LOSS_TYPES = ['cosine', 'l1', 'l2']


class VGG19(nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)
        return out


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, band_width: float=0.5, loss_type: str='cosine', use_vgg: bool=False, vgg_layer: str='relu3_4'):
        super(ContextualLoss, self).__init__()
        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
        self.band_width = band_width
        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(name='vgg_mean', tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
            self.register_buffer(name='vgg_std', tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, 'VGG model takes 3 chennel images.'
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)
        return F.contextual_loss(x, y, self.band_width)


class ContextualBilateralLoss(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.

    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, weight_sp: float=0.1, band_width: float=0.5, loss_type: str='cosine', use_vgg: bool=False, vgg_layer: str='relu3_4'):
        super(ContextualBilateralLoss, self).__init__()
        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
        self.band_width = band_width
        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(name='vgg_mean', tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
            self.register_buffer(name='vgg_std', tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, 'VGG model takes 3 chennel images.'
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)
        return F.contextual_bilateral_loss(x, y, self.band_width)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (VGG19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_S_aiueo32_contextual_loss_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

