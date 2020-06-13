import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
main = _module
model = _module
solver = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


import scipy.io


import numpy as np


from torch.autograd import Variable


from torch import optim


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias
        =False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G12(nn.Module):
    """Generator for transfering from mnist to svhn"""

    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
        self.conv1 = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.tanh(self.deconv2(out))
        return out


class G21(nn.Module):
    """Generator for transfering from svhn to mnist"""

    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.tanh(self.deconv2(out))
        return out


class D1(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = self.fc(out).squeeze()
        return out


class D2(nn.Module):
    """Discriminator for svhn."""

    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim * 4, n_out, 4, 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = self.fc(out).squeeze()
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yunjey_mnist_svhn_transfer(_paritybench_base):
    pass
    def test_000(self):
        self._check(G12(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_001(self):
        self._check(G21(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_002(self):
        self._check(D1(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_003(self):
        self._check(D2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

