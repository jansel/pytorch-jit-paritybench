import sys
_module = sys.modules[__name__]
del sys
latent_interp = _module
model = _module
network = _module
predict = _module
pygame_interp_demo = _module
transfer_weights = _module
utils = _module

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


from torch.autograd import Variable


from torch.utils.data.dataloader import DataLoader


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


from scipy import ndimage


from torch.utils.data import Dataset


class PixelNormLayer(nn.Module):

    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-08)


class WScaleLayer(nn.Module):

    def __init__(self, size):
        super(WScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.randn([1]))
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(x_size[0], self.size, x_size[2], x_size[3])
        return x


class NormConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class NormUpscaleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.features = nn.Sequential(NormConvBlock(512, 512, kernel_size=4, padding=3), NormConvBlock(512, 512, kernel_size=3, padding=1), NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1), NormConvBlock(512, 512, kernel_size=3, padding=1), NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1), NormConvBlock(512, 512, kernel_size=3, padding=1), NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1), NormConvBlock(512, 512, kernel_size=3, padding=1), NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1), NormConvBlock(256, 256, kernel_size=3, padding=1), NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1), NormConvBlock(128, 128, kernel_size=3, padding=1), NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1), NormConvBlock(64, 64, kernel_size=3, padding=1), NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1), NormConvBlock(32, 32, kernel_size=3, padding=1), NormUpscaleConvBlock(32, 16, kernel_size=3, padding=1), NormConvBlock(16, 16, kernel_size=3, padding=1))
        self.output = nn.Sequential(OrderedDict([('norm', PixelNormLayer()), ('conv', nn.Conv2d(16, 3, kernel_size=1, padding=0, bias=False)), ('wscale', WScaleLayer(3))]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (NormConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormUpscaleConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNormLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WScaleLayer,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ptrblck_prog_gans_pytorch_inference(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

