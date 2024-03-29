import sys
_module = sys.modules[__name__]
del sys
dataset = _module
loader = _module
model = _module
cbdnet = _module
predict = _module
train = _module
utils = _module
common = _module
ISP_implement = _module
generate_dataset = _module
Demosaicing_malvar2004 = _module
modules = _module
masks = _module

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


import random


import torch


import numpy as np


from torch.utils.data import Dataset


import torch.nn as nn


import torch.nn.functional as F


import time


import scipy.io


import math


class single_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):

    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = x2 + x1
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fcn(x)


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.inc = nn.Sequential(single_conv(6, 64), single_conv(64, 64))
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(single_conv(64, 128), single_conv(128, 128), single_conv(128, 128))
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(single_conv(128, 256), single_conv(256, 256), single_conv(256, 256), single_conv(256, 256), single_conv(256, 256), single_conv(256, 256))
        self.up1 = up(256)
        self.conv3 = nn.Sequential(single_conv(128, 128), single_conv(128, 128), single_conv(128, 128))
        self.up2 = up(128)
        self.conv4 = nn.Sequential(single_conv(64, 64), single_conv(64, 64))
        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)
        down1 = self.down1(inx)
        conv1 = self.conv1(down1)
        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)
        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)
        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)
        out = self.outc(conv4)
        return out


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()

    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out


class fixed_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)
        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        h_tv = torch.pow(est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1], 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w
        loss = l2_loss + 0.5 * asym_loss + 0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FCN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Network,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     True),
    (fixed_loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (outconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (single_conv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (up,
     lambda: ([], {'in_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])], {}),
     True),
]

class Test_IDKiro_CBDNet_pytorch(_paritybench_base):
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

