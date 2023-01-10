import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
test_checkpoint = _module
test_layers = _module
test_logger = _module
test_losses = _module
torchkit = _module
checkpoint = _module
experiment = _module
layers = _module
logger = _module
losses = _module
utils = _module
config = _module
dataset = _module
git = _module
io = _module
module_freezing = _module
module_stats = _module
multithreading = _module
pdb_fallback = _module
seed = _module
timer = _module
version = _module
viz = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.testing import assert_allclose


import numpy as np


import logging


from typing import Any


from typing import List


from typing import Optional


from typing import Union


from typing import Type


from typing import cast


import torchvision


from torch.utils.tensorboard import SummaryWriter


from typing import Iterable


from typing import Iterator


import random


from typing import Tuple


from torchvision import transforms as T


class Flatten(nn.Module):
    """Flattens convolutional feature maps for fully-connected layers.

    This is a convenience module meant to be plugged into a
    `torch.nn.Sequential` model.

    Example usage::

        import torch.nn as nn
        from torchkit import layers

        # Assume an input of shape (3, 28, 28).
        net = nn.Sequential(
            layers.conv2d(3, 8, kernel_size=3),
            nn.ReLU(),
            layers.conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            layers.Flatten(),
            nn.Linear(28*28*16, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) ->Tensor:
        return x.view(x.shape[0], -1)


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in `1`_.

    Concretely, the spatial softmax of each feature map is used to compute a
    weighted mean of the pixel locations, effectively performing a soft arg-max
    over the feature dimension.

    .. _1: https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize: bool=False):
        """Constructor.

        Args:
            normalize: Whether to use normalized image coordinates, i.e.
                coordinates in the range `[-1, 1]`.
        """
        super().__init__()
        self.normalize = normalize

    def _coord_grid(self, h: int, w: int, device: torch.device) ->Tensor:
        if self.normalize:
            return torch.stack(torch.meshgrid(torch.linspace(-1, 1, w, device=device), torch.linspace(-1, 1, h, device=device)))
        return torch.stack(torch.meshgrid(torch.arange(0, w, device=device), torch.arange(0, h, device=device)))

    def forward(self, x: Tensor) ->Tensor:
        assert x.ndim == 4, 'Expecting a tensor of shape (B, C, H, W).'
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)
        xc, yc = self._coord_grid(h, w, x.device)
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class _GlobalMaxPool(nn.Module):
    """Global max pooling layer."""

    def __init__(self, dim):
        super().__init__()
        if dim == 1:
            self._pool = F.max_pool1d
        elif dim == 2:
            self._pool = F.max_pool2d
        elif dim == 3:
            self._pool = F.max_pool3d
        else:
            raise ValueError('{}D is not supported.')

    def forward(self, x: Tensor) ->Tensor:
        out = self._pool(x, kernel_size=x.size()[2:])
        for _ in range(len(out.shape[2:])):
            out.squeeze_(dim=-1)
        return out


class GlobalMaxPool1d(_GlobalMaxPool):
    """Global max pooling operation for temporal or 1D data."""

    def __init__(self):
        super().__init__(dim=1)


class GlobalMaxPool2d(_GlobalMaxPool):
    """Global max pooling operation for spatial or 2D data."""

    def __init__(self):
        super().__init__(dim=2)


class GlobalMaxPool3d(_GlobalMaxPool):
    """Global max pooling operation for 3D data."""

    def __init__(self):
        super().__init__(dim=3)


class _GlobalAvgPool(nn.Module):
    """Global average pooling layer."""

    def __init__(self, dim):
        super().__init__()
        if dim == 1:
            self._pool = F.avg_pool1d
        elif dim == 2:
            self._pool = F.avg_pool2d
        elif dim == 3:
            self._pool = F.avg_pool3d
        else:
            raise ValueError('{}D is not supported.')

    def forward(self, x: Tensor) ->Tensor:
        out = self._pool(x, kernel_size=x.size()[2:])
        for _ in range(len(out.shape[2:])):
            out.squeeze_(dim=-1)
        return out


class GlobalAvgPool1d(_GlobalAvgPool):
    """Global average pooling operation for temporal or 1D data."""

    def __init__(self):
        super().__init__(dim=1)


class GlobalAvgPool2d(_GlobalAvgPool):
    """Global average pooling operation for spatial or 2D data."""

    def __init__(self):
        super().__init__(dim=2)


class GlobalAvgPool3d(_GlobalAvgPool):
    """Global average pooling operation for 3D data."""

    def __init__(self):
        super().__init__(dim=3)


class CausalConv1d(nn.Conv1d):
    """A causal a.k.a. masked 1D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, dilation: int=1, bias: bool=True):
        """Constructor.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The filter size.
            stride: The filter stride.
            dilation: The filter dilation factor.
            bias: Whether to add the bias term or not.
        """
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, bias=bias)

    def forward(self, x: Tensor) ->Tensor:
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, :-self.__padding]
        return res


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CausalConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalMaxPool1d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GlobalMaxPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialSoftArgmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kevinzakka_torchkit(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

