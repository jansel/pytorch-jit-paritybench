import sys
_module = sys.modules[__name__]
del sys
PCGModel = _module
custom_scheduler = _module
data = _module
densify = _module
util = _module
evaluate = _module
evaluate_dist = _module
findLR_stg1 = _module
findLR_stg2 = _module
options = _module
convertEXR = _module
render = _module
render_fixed = _module
train_stg1 = _module
train_stg2 = _module
trainer = _module
transform = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from torch import nn


from torch.nn import functional as F


import numpy as np


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import scipy


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import torch.nn.functional as F


import scipy.io


import torch.nn as nn


import logging


from torch import optim


from torch.optim import lr_scheduler


import torchvision


def conv2d_block(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())


def linear_block(in_c, out_c):
    return nn.Sequential(nn.Linear(in_c, out_c), nn.BatchNorm1d(out_c), nn.ReLU())


class Encoder(nn.Module):
    """Encoder of Structure Generator"""

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv2d_block(3, 96)
        self.conv2 = conv2d_block(96, 128)
        self.conv3 = conv2d_block(128, 192)
        self.conv4 = conv2d_block(192, 256)
        self.fc1 = linear_block(4096, 2048)
        self.fc2 = linear_block(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x.view(-1, 4096))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def deconv2d_block(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, stride=1, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())


def pixel_bias(outViewN, outW, outH, renderDepth):
    X, Y = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
    X, Y = X.float(), Y.float()
    initTile = torch.cat([X.repeat([outViewN, 1, 1]), Y.repeat([outViewN, 1, 1]), torch.ones([outViewN, outH, outW]).float() * renderDepth, torch.zeros([outViewN, outH, outW]).float()], dim=0)
    return initTile.unsqueeze_(dim=0)


class Decoder(nn.Module):
    """Build Decoder"""

    def __init__(self, outViewN, outW, outH, renderDepth):
        super(Decoder, self).__init__()
        self.outViewN = outViewN
        self.relu = nn.ReLU()
        self.fc1 = linear_block(512, 1024)
        self.fc2 = linear_block(1024, 2048)
        self.fc3 = linear_block(2048, 4096)
        self.deconv1 = deconv2d_block(256, 192)
        self.deconv2 = deconv2d_block(192, 128)
        self.deconv3 = deconv2d_block(128, 96)
        self.deconv4 = deconv2d_block(96, 64)
        self.deconv5 = deconv2d_block(64, 48)
        self.pixel_conv = nn.Conv2d(48, outViewN * 4, 1, stride=1, bias=False)
        self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        x = self.relu(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view([-1, 256, 4, 4])
        x = self.deconv1(F.interpolate(x, scale_factor=2))
        x = self.deconv2(F.interpolate(x, scale_factor=2))
        x = self.deconv3(F.interpolate(x, scale_factor=2))
        x = self.deconv4(F.interpolate(x, scale_factor=2))
        x = self.deconv5(F.interpolate(x, scale_factor=2))
        x = self.pixel_conv(x) + self.pixel_bias
        XYZ, maskLogit = torch.split(x, [self.outViewN * 3, self.outViewN], dim=1)
        return XYZ, maskLogit


class Structure_Generator(nn.Module):
    """Structure generator components in PCG"""

    def __init__(self, encoder=None, decoder=None, outViewN=8, outW=128, outH=128, renderDepth=1.0):
        super(Structure_Generator, self).__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder()
        if decoder:
            self.decoder = decoder
        else:
            self.decoder = Decoder(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        latent = self.encoder(x)
        XYZ, maskLogit = self.decoder(latent)
        return XYZ, maskLogit


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Structure_Generator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_lkhphuc_pytorch_3d_point_cloud_generation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

