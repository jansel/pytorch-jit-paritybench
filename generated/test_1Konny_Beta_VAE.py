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
        self.encoder = nn.Sequential(nn.Conv2d(nc, 32, 4, 2, 1), nn.ReLU(
            True), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32,
            64, 4, 2, 1), nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1), nn.
            ReLU(True), nn.Conv2d(64, 256, 4, 1), nn.ReLU(True), View((-1, 
            256 * 1 * 1)), nn.Linear(256, z_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(z_dim, 256), View((-1, 256, 
            1, 1)), nn.ReLU(True), nn.ConvTranspose2d(256, 64, 4), nn.ReLU(
            True), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, nc, 4, 2, 1))
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_1Konny_Beta_VAE(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BetaVAE_H(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(View(*[], **{'size': 4}), [torch.rand([4])], {})

