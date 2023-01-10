import sys
_module = sys.modules[__name__]
del sys
lr_sh = _module
main = _module
model = _module
train = _module
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


from torch.optim.optimizer import Optimizer


import torch


from torch import optim


from torch import nn


from collections import OrderedDict


import scipy.ndimage as nd


import scipy.io as io


import matplotlib


import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec


import numpy as np


from torch.utils import data


from torch.autograd import Variable


class _G(torch.nn.Module):

    def __init__(self, args):
        super(_G, self).__init__()
        self.args = args
        self.cube_len = args.cube_len
        padd = 0, 0, 0
        if self.cube_len == 32:
            padd = 1, 1, 1
        self.layer1 = torch.nn.Sequential(torch.nn.ConvTranspose3d(self.args.z_size, self.cube_len * 8, kernel_size=4, stride=2, bias=args.bias, padding=padd), torch.nn.BatchNorm3d(self.cube_len * 8), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.ConvTranspose3d(self.cube_len * 8, self.cube_len * 4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len * 4), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.ConvTranspose3d(self.cube_len * 4, self.cube_len * 2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len * 2), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.ConvTranspose3d(self.cube_len * 2, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len), torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.Sigmoid())

    def forward(self, x):
        out = x.view(-1, self.args.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class _D(torch.nn.Module):

    def __init__(self, args):
        super(_D, self).__init__()
        self.args = args
        self.cube_len = args.cube_len
        padd = 0, 0, 0
        if self.cube_len == 32:
            padd = 1, 1, 1
        self.layer1 = torch.nn.Sequential(torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len), torch.nn.LeakyReLU(self.args.leak_value))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv3d(self.cube_len, self.cube_len * 2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len * 2), torch.nn.LeakyReLU(self.args.leak_value))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv3d(self.cube_len * 2, self.cube_len * 4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len * 4), torch.nn.LeakyReLU(self.args.leak_value))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv3d(self.cube_len * 4, self.cube_len * 8, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)), torch.nn.BatchNorm3d(self.cube_len * 8), torch.nn.LeakyReLU(self.args.leak_value))
        self.layer5 = torch.nn.Sequential(torch.nn.Conv3d(self.cube_len * 8, 1, kernel_size=4, stride=2, bias=args.bias, padding=padd), torch.nn.Sigmoid())

    def forward(self, x):
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (_G,
     lambda: ([], {'args': _mock_config(cube_len=4, z_size=4, bias=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_rimchang_3DGAN_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

