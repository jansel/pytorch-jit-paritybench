import sys
_module = sys.modules[__name__]
del sys
main = _module
dcgan = _module
gan = _module
wgan_clipping = _module
wgan_gradient_penalty = _module
config = _module
data_loader = _module
fashion_mnist = _module
feature_extraction_test = _module
inception_score = _module
tensorboard_logger = _module

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


import torch.nn as nn


from torch.autograd import Variable


import time as t


from itertools import chain


import time


import torch.optim as optim


from torch import autograd


from torch import nn


from torch.nn import functional as F


import torch.utils.data


import numpy as np


from scipy.stats import entropy


class Generator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.ConvTranspose2d(in_channels=100,
            out_channels=1024, kernel_size=4, stride=1, padding=0), nn.
            BatchNorm2d(num_features=1024), nn.ReLU(True), nn.
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size
            =4, stride=2, padding=1), nn.BatchNorm2d(num_features=512), nn.
            ReLU(True), nn.ConvTranspose2d(in_channels=512, out_channels=
            256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
            num_features=256), nn.ReLU(True), nn.ConvTranspose2d(
            in_channels=256, out_channels=channels, kernel_size=4, stride=2,
            padding=1))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.Conv2d(in_channels=channels,
            out_channels=256, kernel_size=4, stride=2, padding=1), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=256,
            out_channels=512, kernel_size=4, stride=2, padding=1), nn.
            BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=4, stride=2,
            padding=1), nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(nn.Conv2d(in_channels=1024,
            out_channels=1, kernel_size=4, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class Generator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.ConvTranspose2d(in_channels=100,
            out_channels=1024, kernel_size=4, stride=1, padding=0), nn.
            BatchNorm2d(num_features=1024), nn.ReLU(True), nn.
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size
            =4, stride=2, padding=1), nn.BatchNorm2d(num_features=512), nn.
            ReLU(True), nn.ConvTranspose2d(in_channels=512, out_channels=
            256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
            num_features=256), nn.ReLU(True), nn.ConvTranspose2d(
            in_channels=256, out_channels=channels, kernel_size=4, stride=2,
            padding=1))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.Conv2d(in_channels=channels,
            out_channels=256, kernel_size=4, stride=2, padding=1), nn.
            BatchNorm2d(num_features=256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
            stride=2, padding=1), nn.BatchNorm2d(num_features=512), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=512,
            out_channels=1024, kernel_size=4, stride=2, padding=1), nn.
            BatchNorm2d(num_features=1024), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(nn.Conv2d(in_channels=1024,
            out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class Generator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.ConvTranspose2d(in_channels=100,
            out_channels=1024, kernel_size=4, stride=1, padding=0), nn.
            BatchNorm2d(num_features=1024), nn.ReLU(True), nn.
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size
            =4, stride=2, padding=1), nn.BatchNorm2d(num_features=512), nn.
            ReLU(True), nn.ConvTranspose2d(in_channels=512, out_channels=
            256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
            num_features=256), nn.ReLU(True), nn.ConvTranspose2d(
            in_channels=256, out_channels=channels, kernel_size=4, stride=2,
            padding=1))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(nn.Conv2d(in_channels=channels,
            out_channels=256, kernel_size=4, stride=2, padding=1), nn.
            InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(in_channels=256, out_channels=512, kernel_size
            =4, stride=2, padding=1), nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=512,
            out_channels=1024, kernel_size=4, stride=2, padding=1), nn.
            InstanceNorm2d(1024, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(nn.Conv2d(in_channels=1024,
            out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Zeleni9_pytorch_wgan(_paritybench_base):
    pass
    def test_000(self):
        self._check(Discriminator(*[], **{'channels': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_001(self):
        self._check(Generator(*[], **{'channels': 4}), [torch.rand([4, 100, 4, 4])], {})

