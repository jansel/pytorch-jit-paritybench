import sys
_module = sys.modules[__name__]
del sys
wae_gan = _module
wae_mmd = _module

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


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.autograd import Variable


from torchvision.datasets import MNIST


from torchvision.transforms import transforms


from torchvision.utils import save_image


from torch.optim.lr_scheduler import StepLR


import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.main = nn.Sequential(nn.Conv2d(self.n_channel, self.dim_h, 4, 
            2, 1, bias=False), nn.ReLU(True), nn.Conv2d(self.dim_h, self.
            dim_h * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True), nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 
            1, bias=False), nn.BatchNorm2d(self.dim_h * 4), nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8), nn.ReLU(True))
        self.fc = nn.Linear(self.dim_h * 2 ** 3, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.proj = nn.Sequential(nn.Linear(self.n_z, self.dim_h * 8 * 7 * 
            7), nn.ReLU())
        self.main = nn.Sequential(nn.ConvTranspose2d(self.dim_h * 8, self.
            dim_h * 4, 4), nn.BatchNorm2d(self.dim_h * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4), nn.
            BatchNorm2d(self.dim_h * 2), nn.ReLU(True), nn.ConvTranspose2d(
            self.dim_h * 2, 1, 4, stride=2), nn.Sigmoid())

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.main = nn.Sequential(nn.Linear(self.n_z, self.dim_h * 4), nn.
            ReLU(True), nn.Linear(self.dim_h * 4, self.dim_h * 4), nn.ReLU(
            True), nn.Linear(self.dim_h * 4, self.dim_h * 4), nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4), nn.ReLU(True), nn.
            Linear(self.dim_h * 4, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.main(x)
        return x


class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.main = nn.Sequential(nn.Conv2d(self.n_channel, self.dim_h, 4, 
            2, 1, bias=False), nn.ReLU(True), nn.Conv2d(self.dim_h, self.
            dim_h * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True), nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 
            1, bias=False), nn.BatchNorm2d(self.dim_h * 4), nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8), nn.ReLU(True))
        self.fc = nn.Linear(self.dim_h * 2 ** 3, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.proj = nn.Sequential(nn.Linear(self.n_z, self.dim_h * 8 * 7 * 
            7), nn.ReLU())
        self.main = nn.Sequential(nn.ConvTranspose2d(self.dim_h * 8, self.
            dim_h * 4, 4), nn.BatchNorm2d(self.dim_h * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4), nn.
            BatchNorm2d(self.dim_h * 2), nn.ReLU(True), nn.ConvTranspose2d(
            self.dim_h * 2, 1, 4, stride=2), nn.Sigmoid())

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_schelotto_Wasserstein_AutoEncoders(_paritybench_base):
    pass
    def test_000(self):
        self._check(Decoder(*[], **{'args': _mock_config(n_channel=4, dim_h=4, n_z=4)}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Discriminator(*[], **{'args': _mock_config(n_channel=4, dim_h=4, n_z=4)}), [torch.rand([4, 4, 4, 4])], {})

