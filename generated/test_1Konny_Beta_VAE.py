import sys
_module = sys.modules[__name__]
del sys
dataset = _module
main = _module
model = _module
solver = _module
utils = _module

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


from torch.utils.data import DataLoader


from torchvision.datasets import ImageFolder


from torchvision import transforms


import torch.nn as nn


import torch.nn.init as init


from torch.autograd import Variable


import warnings


import torch.optim as optim


import torch.nn.functional as F


from torchvision.utils import make_grid


from torchvision.utils import save_image


class View(nn.Module):

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(nn.Conv2d(nc, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True), nn.Conv2d(64, 256, 4, 1), nn.ReLU(True), View((-1, 256 * 1 * 1)), nn.Linear(256, z_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(z_dim, 256), View((-1, 256, 1, 1)), nn.ReLU(True), nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(32, nc, 4, 2, 1))
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.encoder = nn.Sequential(nn.Conv2d(nc, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), View((-1, 32 * 4 * 4)), nn.Linear(32 * 4 * 4, 256), nn.ReLU(True), nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, z_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU(True), nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 32 * 4 * 4), nn.ReLU(True), View((-1, 32, 4, 4)), nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(32, nc, 4, 2, 1))
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BetaVAE_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (BetaVAE_H,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (View,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4])], {}),
     True),
]

class Test_1Konny_Beta_VAE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

