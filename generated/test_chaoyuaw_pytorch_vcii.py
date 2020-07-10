import sys
_module = sys.modules[__name__]
del sys
dataset = _module
evaluate = _module
functions = _module
sign = _module
metric = _module
modules = _module
conv_rnn = _module
sign = _module
network = _module
train = _module
train_options = _module
unet = _module
unet_parts = _module
util = _module

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


import torch


import torch.utils.data as data


import numpy as np


import random


import time


from torch.autograd import Variable


from torchvision import transforms


from torch.autograd import Function


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.utils import _pair


import torch.optim as optim


import torch.optim.lr_scheduler as LS


from collections import namedtuple


class ConvRNNCellBase(nn.Module):

    def __repr__(self):
        s = '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):

    def __init__(self, input_channels, hidden_channels, kernel_size=3, stride=1, padding=0, dilation=1, hidden_kernel_size=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.hidden_kernel_size = _pair(hidden_kernel_size)
        hidden_padding = _pair(hidden_kernel_size // 2)
        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Conv2d(in_channels=self.input_channels, out_channels=gate_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=bias)
        self.conv_hh = nn.Conv2d(in_channels=self.hidden_channels, out_channels=gate_channels, kernel_size=hidden_kernel_size, stride=1, padding=hidden_padding, dilation=1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(cy)
        return hy, cy


class Sign(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)


class EncoderCell(nn.Module):

    def __init__(self, v_compress, stack, fuse_encoder, fuse_level):
        super(EncoderCell, self).__init__()
        self.v_compress = v_compress
        self.fuse_encoder = fuse_encoder
        self.fuse_level = fuse_level
        if fuse_encoder:
            None
        self.conv = nn.Conv2d(9 if stack else 3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(128 if fuse_encoder and v_compress else 64, 256, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTMCell((384 if fuse_encoder and v_compress else 256) if self.fuse_level >= 2 else 256, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTMCell((768 if fuse_encoder and v_compress else 512) if self.fuse_level >= 3 else 512, 512, kernel_size=3, stride=2, padding=1, hidden_kernel_size=1, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, unet_output1, unet_output2):
        x = self.conv(input)
        if self.v_compress and self.fuse_encoder:
            x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        if self.v_compress and self.fuse_encoder and self.fuse_level >= 2:
            x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        if self.v_compress and self.fuse_encoder and self.fuse_level >= 3:
            x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):

    def __init__(self, bits):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, bits, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):

    def __init__(self, v_compress, shrink, bits, fuse_level):
        super(DecoderCell, self).__init__()
        self.v_compress = v_compress
        self.fuse_level = fuse_level
        None
        self.conv1 = nn.Conv2d(bits, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = ConvLSTMCell(512, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTMCell((128 + 256 // shrink * 2 if v_compress else 128) if self.fuse_level >= 3 else 128, 512, kernel_size=3, stride=1, padding=1, hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTMCell((128 + 128 // shrink * 2 if v_compress else 128) if self.fuse_level >= 2 else 128, 256, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3, bias=False)
        self.rnn4 = ConvLSTMCell(64 + 64 // shrink * 2 if v_compress else 64, 128, kernel_size=3, stride=1, padding=1, hidden_kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4, unet_output1, unet_output2):
        x = self.conv1(input)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)
        if self.v_compress and self.fuse_level >= 3:
            x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)
        if self.v_compress and self.fuse_level >= 2:
            x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)
        if self.v_compress:
            x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)
        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)
        x = F.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, shrink):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // shrink)
        self.down1 = down(64 // shrink, 128 // shrink)
        self.down2 = down(128 // shrink, 256 // shrink)
        self.down3 = down(256 // shrink, 512 // shrink)
        self.down4 = down(512 // shrink, 512 // shrink)
        self.up1 = up(1024 // shrink, 256 // shrink)
        self.up2 = up(512 // shrink, 128 // shrink)
        self.up3 = up(256 // shrink, 64 // shrink)
        self.up4 = up(128 // shrink, 64 // shrink)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out1 = self.up1(x5, x4)
        out2 = self.up2(out1, x3)
        out3 = self.up3(out2, x2)
        return [out1, out2, out3]


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvLSTMCell,
     lambda: ([], {'input_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64]), (torch.rand([4, 4, 62, 62]), torch.rand([4, 4, 62, 62]))], {}),
     False),
    (UNet,
     lambda: ([], {'n_channels': 4, 'shrink': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (double_conv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (inconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (outconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (up,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
]

class Test_chaoyuaw_pytorch_vcii(_paritybench_base):
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

