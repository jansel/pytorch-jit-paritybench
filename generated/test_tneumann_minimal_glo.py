import sys
_module = sys.modules[__name__]
del sys
glo = _module

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


import numpy as np


import torch


from torch import nn


import torch.nn.functional as fnn


from torch.autograd import Variable


from torch.optim import SGD


from torchvision.datasets import LSUN


from torchvision import transforms


from torch.utils.data import Dataset


from torchvision.utils import make_grid


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError('kernel size must be uneven')
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)
    pyr.append(current)
    return pyr


class LapLoss(nn.Module):

    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma, n_channels=input.shape[1], cuda=input.is_cuda)
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


class Generator(nn.Module):

    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(nn.ConvTranspose2d(code_dim, nf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(nf * 8), nn.ReLU(True), nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.ReLU(True), nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 2), nn.ReLU(True), nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(True), nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Generator,
     lambda: ([], {'code_dim': 4}),
     lambda: ([torch.rand([4, 4, 1, 1])], {}),
     True),
    (LapLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 512, 512]), torch.rand([4, 4, 512, 512])], {}),
     False),
]

class Test_tneumann_minimal_glo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

