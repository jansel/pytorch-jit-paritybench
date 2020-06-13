import sys
_module = sys.modules[__name__]
del sys
datasets = _module
noise2noise = _module
render = _module
test = _module
train = _module
unet = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


import random


from string import ascii_letters


import torch.nn as nn


from torch.optim import Adam


from torch.optim import lr_scheduler


from math import log10


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""
        super(HDRLoss, self).__init__()
        self._eps = eps

    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""
        loss = (denoised - target) ** 2 / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""
        super(UNet, self).__init__()
        self._block1 = nn.Sequential(nn.Conv2d(in_channels, 48, 3, stride=1,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding
            =1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self._block2 = nn.Sequential(nn.Conv2d(48, 48, 3, stride=1, padding
            =1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self._block3 = nn.Sequential(nn.Conv2d(48, 48, 3, stride=1, padding
            =1), nn.ReLU(inplace=True), nn.ConvTranspose2d(48, 48, 3,
            stride=2, padding=1, output_padding=1))
        self._block4 = nn.Sequential(nn.Conv2d(96, 96, 3, stride=1, padding
            =1), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, stride=1,
            padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(96, 96, 3,
            stride=2, padding=1, output_padding=1))
        self._block5 = nn.Sequential(nn.Conv2d(144, 96, 3, stride=1,
            padding=1), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, stride=
            1, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(96, 96,
            3, stride=2, padding=1, output_padding=1))
        self._block6 = nn.Sequential(nn.Conv2d(96 + in_channels, 64, 3,
            stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 32, 
            3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32,
            out_channels, 3, stride=1, padding=1), nn.LeakyReLU(0.1))
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block6(concat1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_joeylitalien_noise2noise_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(HDRLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(UNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

