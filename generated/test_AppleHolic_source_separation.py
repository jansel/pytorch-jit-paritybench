import sys
_module = sys.modules[__name__]
del sys
setup = _module
source_separation = _module
dataset = _module
hyperopt_run = _module
models = _module
modules = _module
settings = _module
synthesize = _module
train = _module
train_jointly = _module
trainer = _module

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


import torch


import torch.nn as nn


from typing import Tuple


from typing import Dict


from typing import Any


from torch.optim.lr_scheduler import MultiStepLR


import torch.nn.functional as F


import numpy as np


from torch.nn.init import calculate_gain


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class ComplexConvBlock(nn.Module):
    """
    Convolution block
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size:
        int, padding: int=0, layers: int=4, bn_func=nn.BatchNorm1d,
        act_func=nn.LeakyReLU, skip_res: bool=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.skip_res = skip_res
        for idx in range(layers):
            in_ = in_channels if idx == 0 else out_channels
            self.blocks.append(nn.Sequential(*[bn_func(in_), act_func(),
                ComplexConv1d(in_, out_channels, kernel_size, padding=
                padding)]))

    def forward(self, x: torch.tensor) ->torch.tensor:
        temp = x
        for idx, block in enumerate(self.blocks):
            x = block(x)
        if temp.size() != x.size() or self.skip_res:
            return x
        else:
            return x + temp


class _ComplexConvNd(nn.Module):
    """
    Implement Complex Convolution
    A: real weight
    B: img weight
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, transposed, output_padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.transposed = transposed
        self.A = self.make_weight(in_channels, out_channels, kernel_size)
        self.B = self.make_weight(in_channels, out_channels, kernel_size)
        self.reset_parameters()

    def make_weight(self, in_ch, out_ch, kernel_size):
        if self.transposed:
            tensor = nn.Parameter(torch.Tensor(in_ch, out_ch // 2, *
                kernel_size))
        else:
            tensor = nn.Parameter(torch.Tensor(out_ch, in_ch // 2, *
                kernel_size))
        return tensor

    def reset_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.A)
        gain = calculate_gain('leaky_relu', 0)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std
        with torch.no_grad():
            self.A.uniform_(-bound * (1 / np.pi ** 2), bound * (1 / np.pi ** 2)
                )
            self.B.uniform_(-1 / np.pi, 1 / np.pi)


class ComplexActLayer(nn.Module):
    """
    Activation differently 'real' part and 'img' part
    In implemented DCUnet on this repository, Real part is activated to log space.
    And Phase(img) part, it is distributed in [-pi, pi]...
    """

    def forward(self, x):
        real, img = x.chunk(2, 1)
        return torch.cat([F.leaky_relu_(real), torch.tanh(img) * np.pi], dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_AppleHolic_source_separation(_paritybench_base):
    pass
    def test_000(self):
        self._check(ComplexActLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

