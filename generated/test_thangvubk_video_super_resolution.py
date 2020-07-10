import sys
_module = sys.modules[__name__]
del sys
SR_datasets = _module
colorize = _module
loss = _module
model = _module
pytorch_ssim = _module
solver = _module
test = _module
train = _module

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


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.autograd import Variable


from math import exp


import time


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


from torch.utils.data import DataLoader


class MSE_and_SSIM_loss(nn.Module):

    def __init__(self, alpha=0.9):
        super(MSE_and_SSIM_loss, self).__init__()
        self.MSE = nn.MSELoss()
        self.SSIM = pytorch_ssim.SSIM()
        self.alpha = alpha

    def forward(self, img1, img2):
        loss = self.alpha * self.MSE(img1, img2) + (1 - self.alpha) * (1 - self.SSIM(img1, img2))
        return loss


class VSRCNN(nn.Module):
    """
    Model for SRCNN

    LR -> Conv1 -> Relu -> Conv2 -> Relu -> Conv3 -> HR

    Args:
        - C1, C2, C3: num output channels for Conv1, Conv2, and Conv3
        - F1, F2, F3: filter size
    """

    def __init__(self, C1=64, C2=32, C3=1, F1=9, F2=1, F3=5):
        super(VSRCNN, self).__init__()
        self.name = 'VSRCNN'
        self.conv1 = nn.Conv2d(1, C1, F1, padding=4, bias=False)
        self.conv2 = nn.Conv2d(C1, C2, F2)
        self.conv3 = nn.Conv2d(C2, C3, F3, padding=2, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Conv_ReLU_Block(nn.Module):

    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VRES(nn.Module):

    def __init__(self):
        super(VRES, self).__init__()
        self.name = 'VRES'
        self.conv_first = nn.Conv2d(5, 64, 3, padding=1, bias=False)
        self.conv_next = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        center = 2
        res = x[:, (center), :, :]
        res = res.unsqueeze(1)
        out = self.relu(self.conv_first(x))
        out = self.residual_layer(out)
        out = self.conv_last(out)
        out = torch.add(out, res)
        return out


class MFCNN(nn.Module):

    def __init__(self):
        super(MFCNN, self).__init__()
        self.name = 'MFCNN'
        self.conv1 = nn.Conv2d(5, 32, 9, padding=4, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(16, 1, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class VRES10(VRES):

    def __init__(self):
        super(VRES10, self).__init__()
        self.name = 'VRES10'
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 8)


class VRES5(VRES):

    def __init__(self):
        super(VRES5, self).__init__()
        self.name = 'VRES5'
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 3)


class VRES15(VRES):

    def __init__(self):
        super(VRES15, self).__init__()
        self.name = 'VRES15'
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 13)


class VRES7(VRES):

    def __init__(self):
        super(VRES7, self).__init__()
        self.name = 'VRES7'
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 5)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv_ReLU_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (MFCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
    (MSE_and_SSIM_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (VRES,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
    (VRES10,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
    (VRES15,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
    (VRES5,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
    (VRES7,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 5, 64, 64])], {}),
     True),
    (VSRCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_thangvubk_video_super_resolution(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

