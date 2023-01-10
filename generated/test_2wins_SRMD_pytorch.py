import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
logger = _module
main = _module
model = _module
solver = _module
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


from torch.utils import data


from torchvision import transforms


import numpy as np


import torch.nn as nn


from torchvision.utils import save_image


from torchvision.utils import make_grid


from scipy import signal


from scipy.ndimage import convolve


class SRMD(nn.Module):

    def __init__(self, num_blocks=11, num_channels=18, conv_dim=128, scale_factor=1):
        super(SRMD, self).__init__()
        self.num_channels = num_channels
        self.conv_dim = conv_dim
        self.sf = scale_factor
        self.nonlinear_mapping = self.make_layers(num_blocks)
        self.conv_last = nn.Sequential(nn.Conv2d(self.conv_dim, 3 * self.sf ** 2, kernel_size=3, padding=1), nn.PixelShuffle(self.sf), nn.Sigmoid())

    def forward(self, x):
        b_size = x.shape[0]
        h, w = list(x.shape[2:])
        x = self.nonlinear_mapping(x)
        x = self.conv_last(x)
        return x

    def make_layers(self, num_blocks):
        layers = []
        in_channels = self.num_channels
        for i in range(num_blocks):
            conv2d = nn.Conv2d(in_channels, self.conv_dim, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(self.conv_dim), nn.ReLU(inplace=True)]
            in_channels = self.conv_dim
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SRMD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
]

class Test_2wins_SRMD_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

