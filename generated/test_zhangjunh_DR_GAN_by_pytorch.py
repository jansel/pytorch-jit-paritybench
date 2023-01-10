import sys
_module = sys.modules[__name__]
del sys
data = _module
data_loader = _module
data_processing = _module
dataset = _module
idloader = _module
exam = _module
Component = _module
DRGAN = _module
model = _module
base_model = _module
model_Loader = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
utils = _module

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


import collections


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torchvision import transforms


import re


import numpy as np


import torch.utils.data as data


import torchvision.transforms as transforms


import torch


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.init as init


import torch.optim as optim


import torch.cuda


class conv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 3, 96, 96))

    >>> net = conv_unit(3, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 96, 96])

    >>> net = conv_unit(3, 16, pooling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 48, 48])
    """

    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()
        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x


class Fconv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 64, 48, 48))

    >>> net = Fconv_unit(64, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 48, 48])

    >>> net = Fconv_unit(64, 16, unsampling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 96, 96])
    """

    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()
        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]
        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x


class Decoder(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_z (int): The dimensions of the noise

    >>> Dec = Decoder()
    >>> input = Variable(torch.randn(4, 372))
    >>> output = Dec(input)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """

    def __init__(self, N_p=2, N_z=50):
        super(Decoder, self).__init__()
        Fconv_layers = [Fconv_unit(320, 160), Fconv_unit(160, 256), Fconv_unit(256, 256, unsampling=True), Fconv_unit(256, 128), Fconv_unit(128, 192), Fconv_unit(192, 192, unsampling=True), Fconv_unit(192, 96), Fconv_unit(96, 128), Fconv_unit(128, 128, unsampling=True), Fconv_unit(128, 64), Fconv_unit(64, 64), Fconv_unit(64, 64, unsampling=True), Fconv_unit(64, 32), Fconv_unit(32, 3)]
        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        self.fc = nn.Linear(320 + N_p + N_z, 320 * 6 * 6)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 320, 6, 6)
        x = self.Fconv_layers(x)
        return x


class Multi_Encoder(nn.Module):
    """
    The multi version of the Encoder.

    >>> Enc = Multi_Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """

    def __init__(self):
        super(Multi_Encoder, self).__init__()
        conv_layers = [conv_unit(3, 32), conv_unit(32, 64), conv_unit(64, 64, pooling=True), conv_unit(64, 64), conv_unit(64, 128), conv_unit(128, 128, pooling=True), conv_unit(128, 96), conv_unit(96, 192), conv_unit(192, 192, pooling=True), conv_unit(192, 128), conv_unit(128, 256), conv_unit(256, 256, pooling=True), conv_unit(256, 160), conv_unit(160, 321), nn.AvgPool2d(kernel_size=6)]
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 321)
        t = x[:, :320]
        w = x[:, 320]
        batchsize = len(w)
        r = Variable(torch.zeros(t.size())).type_as(t)
        for i in range(batchsize):
            r[i] = t[i] * w[i]
        r = torch.sum(r, 0, keepdim=True).div(torch.sum(w))
        return torch.cat((t, r.type_as(t)), 0)


class Encoder(nn.Module):
    """
    The single version of the Encoder.

    >>> Enc = Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """

    def __init__(self):
        super(Encoder, self).__init__()
        conv_layers = [conv_unit(3, 32), conv_unit(32, 64), conv_unit(64, 64, pooling=True), conv_unit(64, 64), conv_unit(64, 128), conv_unit(128, 128, pooling=True), conv_unit(128, 96), conv_unit(96, 192), conv_unit(192, 192, pooling=True), conv_unit(192, 128), conv_unit(128, 256), conv_unit(256, 256, pooling=True), conv_unit(256, 160), conv_unit(160, 320), nn.AvgPool2d(kernel_size=6)]
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 320)
        return x


class Generator(nn.Module):
    """
    >>> G = Generator()

    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> pose = Variable(torch.randn(4, 2))
    >>> noise = Variable(torch.randn(4, 50))

    >>> output = G(input, pose, noise)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """

    def __init__(self, N_p=2, N_z=50, single=False):
        super(Generator, self).__init__()
        if single:
            self.enc = Encoder()
        else:
            self.enc = Multi_Encoder()
        self.dec = Decoder(N_p, N_z)

    def forward(self, input, pose, noise):
        x = self.enc(input)
        x = torch.cat((x, pose, noise), 1)
        x = self.dec(x)
        return x


class Discriminator(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_d (int): The sum of the identities

    >>> D = Discriminator()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = D(input)
    >>> output.size()
    torch.Size([4, 503])
    """

    def __init__(self, N_p=2, N_d=500):
        super(Discriminator, self).__init__()
        conv_layers = [conv_unit(3, 32), conv_unit(32, 64), conv_unit(64, 64, pooling=True), conv_unit(64, 64), conv_unit(64, 128), conv_unit(128, 128, pooling=True), conv_unit(128, 96), conv_unit(96, 192), conv_unit(192, 192, pooling=True), conv_unit(192, 128), conv_unit(128, 256), conv_unit(256, 256, pooling=True), conv_unit(256, 160), conv_unit(160, 320), nn.AvgPool2d(kernel_size=6)]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(320, N_d + N_p + 1)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 320)
        x = self.fc(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (Fconv_unit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Multi_Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (conv_unit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zhangjunh_DR_GAN_by_pytorch(_paritybench_base):
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

