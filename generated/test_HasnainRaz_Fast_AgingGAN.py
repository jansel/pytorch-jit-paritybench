import sys
_module = sys.modules[__name__]
del sys
dataset = _module
gan_module = _module
infer = _module
main = _module
models = _module
preprocessing = _module
preprocess_cacd = _module
preprocess_utk = _module
timing = _module

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


from torch.utils.data import Dataset


import itertools


import torch


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.utils import make_grid


import random


import matplotlib.pyplot as plt


import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.BatchNorm2d(in_features), nn.ReLU(), nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.BatchNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):

    def __init__(self, ngf, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.BatchNorm2d(ngf), nn.ReLU()]
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.BatchNorm2d(out_features), nn.ReLU()]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(out_features), nn.ReLU()]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1), nn.InstanceNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1), nn.InstanceNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'ndf': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Generator,
     lambda: ([], {'ngf': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_HasnainRaz_Fast_AgingGAN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

