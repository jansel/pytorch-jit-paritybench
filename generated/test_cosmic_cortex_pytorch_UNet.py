import sys
_module = sys.modules[__name__]
del sys
kaggle_dsb18_preprocessing = _module
predict = _module
train = _module
unet = _module
blocks = _module
dataset = _module
metrics = _module
model = _module
unet = _module
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


import torch.optim as optim


from functools import partial


import torch.nn as nn


from torch.nn.modules.loss import _Loss


import numpy as np


import torch


from torch.utils.data import Dataset


from torchvision import transforms as T


from torchvision.transforms import functional as F


from typing import Callable


from torch.nn.functional import cross_entropy


from torch.nn.modules.loss import _WeightedLoss


from torch.autograd import Variable


from torch.utils.data import DataLoader


from time import time


import torch.nn.functional as F


class SoftDiceLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_gt):
        numerator = torch.sum(y_pred * y_gt)
        denominator = torch.sum(y_pred * y_pred + y_gt * y_gt)
        return numerator / denominator


class First2D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()
        layers = [nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))
        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder2D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, downsample_kernel=2):
        super(Encoder2D, self).__init__()
        layers = [nn.MaxPool2d(kernel_size=downsample_kernel), nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center2D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()
        layers = [nn.MaxPool2d(kernel_size=2), nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))
        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder2D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()
        layers = [nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last2D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()
        layers = [nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm2d(middle_channels), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=1), nn.Softmax(dim=1)]
        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class First3D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First3D, self).__init__()
        layers = [nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm3d(middle_channels), nn.ReLU(inplace=True), nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))
        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder3D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, downsample_kernel=2):
        super(Encoder3D, self).__init__()
        layers = [nn.MaxPool3d(kernel_size=downsample_kernel), nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm3d(middle_channels), nn.ReLU(inplace=True), nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center3D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center3D, self).__init__()
        layers = [nn.MaxPool3d(kernel_size=2), nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm3d(middle_channels), nn.ReLU(inplace=True), nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True), nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))
        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder3D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder3D, self).__init__()
        layers = [nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm3d(middle_channels), nn.ReLU(inplace=True), nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True), nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)]
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last3D(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last3D, self).__init__()
        layers = [nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm3d(middle_channels), nn.ReLU(inplace=True), nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1), nn.BatchNorm3d(middle_channels), nn.ReLU(inplace=True), nn.Conv3d(middle_channels, out_channels, kernel_size=1), nn.Softmax(dim=1)]
        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None, ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight, ignore_index=self.ignore_index)


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape

    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2]
    elif len(shp) == 5:
        pad = 0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2]
    return F.pad(this, pad)


class UNet2D(nn.Module):

    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'
        super(UNet2D, self).__init__()
        encoder_layers = []
        encoder_layers.append(First2D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder2D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1]) for i in range(len(conv_depths) - 2)])
        decoder_layers = []
        decoder_layers.extend([Decoder2D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i]) for i in reversed(range(len(conv_depths) - 2))])
        decoder_layers.append(Last2D(conv_depths[1], conv_depths[0], out_channels))
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center2D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))
        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1 - dec_layer_idx]
            x_cat = torch.cat([pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite], dim=1)
            x_dec.append(dec_layer(x_cat))
        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec


class UNet3D(nn.Module):

    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'
        super(UNet3D, self).__init__()
        encoder_layers = []
        encoder_layers.append(First3D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder3D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1]) for i in range(len(conv_depths) - 2)])
        decoder_layers = []
        decoder_layers.extend([Decoder3D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i]) for i in reversed(range(len(conv_depths) - 2))])
        decoder_layers.append(Last3D(conv_depths[1], conv_depths[0], out_channels))
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center3D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))
        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1 - dec_layer_idx]
            x_cat = torch.cat([pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite], dim=1)
            x_dec.append(dec_layer(x_cat))
        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Center2D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4, 'deconv_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Center3D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4, 'deconv_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Decoder2D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4, 'deconv_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder3D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4, 'deconv_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Encoder2D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder3D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (First2D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (First3D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Last2D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Last3D,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LogNLLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
]

class Test_cosmic_cortex_pytorch_UNet(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

