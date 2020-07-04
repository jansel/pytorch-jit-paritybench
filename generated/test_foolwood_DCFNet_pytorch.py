import sys
_module = sys.modules[__name__]
del sys
DCFNet = _module
gen_otb2013 = _module
eval_otb = _module
net = _module
tune_otb = _module
util = _module
dataset = _module
crop_image = _module
gen_snippet = _module
parse_vid = _module
net = _module
train_DCFNet = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch


import numpy as np


from torch.utils.data import dataloader


import torch.backends.cudnn as cudnn


import time


class DCFNetFeature(nn.Module):

    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.
            LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1))

    def forward(self, x):
        return self.feature(x)


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNet(nn.Module):

    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.model_alphaf = []
        self.model_xf = []
        self.config = config

    def forward(self, x):
        x = self.feature(x) * self.config.cos_window
        xf = torch.rfft(x, signal_ndim=2)
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim
            =True)
        response = torch.irfft(complex_mul(kxzf, self.model_alphaf),
            signal_ndim=2)
        return response

    def update(self, z, lr=1.0):
        z = self.feature(z) * self.config.cos_window
        zf = torch.rfft(z, signal_ndim=2)
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1,
            keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)
        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr
                ) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in list(checkpoint.keys()):
            state_dict = checkpoint['state_dict']
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in list(state_dict.items()):
                    name = k[7:]
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)


class DCFNetFeature(nn.Module):

    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=
            True), nn.Conv2d(32, 32, 3), nn.LocalResponseNorm(size=5, alpha
            =0.0001, beta=0.75, k=1))

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):

    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.yf = config.yf.clone()
        self.lambda0 = config.lambda0

    def forward(self, z, x):
        z = self.feature(z)
        x = self.feature(x)
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1,
            keepdim=True)
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
        alphaf = self.yf / (kzzf + self.lambda0)
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        return response


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_foolwood_DCFNet_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DCFNetFeature(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

