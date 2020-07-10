import sys
_module = sys.modules[__name__]
del sys
UDT = _module
gen_otb2013 = _module
eval_otb = _module
net = _module
tune_otb = _module
util = _module
dataset = _module
crop_image = _module
parse_vid = _module
net = _module
train_UDT = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time as time


import torch.nn as nn


import torch.utils.data as data


from torch.utils.data import dataloader


import torch.backends.cudnn as cudnn


import time


class DCFNetFeature(nn.Module):

    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3), nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1))

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
        self.yf = config.yf.clone()
        self.lambda0 = config.lambda0

    def forward(self, z, x, label):
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
        alphaf = label / (kzzf + self.lambda0)
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        return response


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DCFNetFeature,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_594422814_UDT_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

