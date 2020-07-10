import sys
_module = sys.modules[__name__]
del sys
evaluate_packed = _module
main = _module
wrn_mcdonnell = _module

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


from torch.utils.data import DataLoader


from torch.optim import SGD


from torch.optim.lr_scheduler import CosineAnnealingLR


import torch.utils.data


from torch.nn.functional import cross_entropy


from torch.nn import DataParallel


from torch.backends import cudnn


import torchvision.transforms as T


import torchvision.datasets as datasets


from collections import OrderedDict


import math


from torch import nn


import torch.nn.functional as F


class ForwardSign(torch.autograd.Function):
    """Fake sign op for 1-bit weights.

    See eq. (1) in https://arxiv.org/abs/1802.08530

    Does He-init like forward, and nothing on backward.
    """

    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2.0 / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    @staticmethod
    def backward(ctx, g):
        return g


class ModuleBinarizable(nn.Module):

    def __init__(self, binarize=False):
        super().__init__()
        self.binarize = binarize

    def _get_weight(self, name):
        w = getattr(self, name)
        return ForwardSign.apply(w) if self.binarize else w

    def forward(self):
        pass


def init_weight(*args):
    return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))


class Block(ModuleBinarizable):
    """Pre-activated ResNet block.
    """

    def __init__(self, width, binarize=False):
        super().__init__(binarize)
        self.bn0 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv0', init_weight(width, width, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self._get_weight('conv0'), padding=1)
        h = F.conv2d(F.relu(self.bn1(h)), self._get_weight('conv1'), padding=1)
        return x + h


class DownsampleBlock(ModuleBinarizable):
    """Downsample block.

    Does F.avg_pool2d + torch.cat instead of strided conv.
    """

    def __init__(self, width, binarize=False):
        super().__init__(binarize)
        self.bn0 = nn.BatchNorm2d(width // 2, affine=False)
        self.register_parameter('conv0', init_weight(width, width // 2, 3, 3))
        self.bn1 = nn.BatchNorm2d(width, affine=False)
        self.register_parameter('conv1', init_weight(width, width, 3, 3))

    def forward(self, x):
        h = F.conv2d(F.relu(self.bn0(x)), self._get_weight('conv0'), padding=1, stride=2)
        h = F.conv2d(F.relu(self.bn1(h)), self._get_weight('conv1'), padding=1)
        x_d = F.avg_pool2d(x, kernel_size=3, padding=1, stride=2)
        x_d = torch.cat([x_d, torch.zeros_like(x_d)], dim=1)
        return x_d + h


class WRN_McDonnell(ModuleBinarizable):
    """Implementation of modified Wide Residual Network.

    Differences with pre-activated ResNet and Wide ResNet:
        * BatchNorm has no affine weight and bias parameters
        * First layer has 16 * width channels
        * Last fc layer is removed in favor of 1x1 conv + F.avg_pool2d
        * Downsample is done by F.avg_pool2d + torch.cat instead of strided conv

    First and last convolutional layers are kept in float32.
    """

    def __init__(self, depth, width, num_classes, binarize=False):
        super().__init__()
        self.binarize = binarize
        widths = [int(v * width) for v in (16, 32, 64)]
        n = (depth - 2) // 6
        self.register_parameter('conv0', init_weight(widths[0], 3, 3, 3))
        self.group0 = self._make_block(widths[0], n)
        self.group1 = self._make_block(widths[1], n, downsample=True)
        self.group2 = self._make_block(widths[2], n, downsample=True)
        self.bn = nn.BatchNorm2d(widths[2], affine=False)
        self.register_parameter('conv_last', init_weight(num_classes, widths[2], 1, 1))
        self.bn_last = nn.BatchNorm2d(num_classes)

    def _make_block(self, width, n, downsample=False):

        def select_block(j):
            if downsample and j == 0:
                return DownsampleBlock(width, self.binarize)
            return Block(width, self.binarize)
        return nn.Sequential(OrderedDict(('block%d' % i, select_block(i)) for i in range(n)))

    def forward(self, x):
        h = F.conv2d(x, self.conv0, padding=1)
        h = self.group0(h)
        h = self.group1(h)
        h = self.group2(h)
        h = F.relu(self.bn(h))
        h = F.conv2d(h, self.conv_last)
        h = self.bn_last(h)
        return F.avg_pool2d(h, kernel_size=h.shape[-2:]).view(h.shape[0], -1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'width': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ModuleBinarizable,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
]

class Test_szagoruyko_binary_wide_resnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

