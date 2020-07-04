import sys
_module = sys.modules[__name__]
del sys
dataset = _module
decoder = _module
encoder = _module
functions = _module
sign = _module
metric = _module
modules = _module
conv_rnn = _module
sign = _module
network = _module
draw_rd = _module
train = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


from torch.autograd import Variable


from torch.nn.modules.utils import _pair


import time


import numpy as np


import torch.optim as optim


import torch.optim.lr_scheduler as LS


import torch.utils.data as data


from torchvision import transforms


class ConvRNNCellBase(nn.Module):

    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}, stride={stride}'
            )
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Sign(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)


class ConvLSTMCell(ConvRNNCellBase):

    def __init__(self, input_channels, hidden_channels, kernel_size=3,
        stride=1, padding=0, dilation=1, hidden_kernel_size=1, bias=True):
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
        self.conv_ih = nn.Conv2d(in_channels=self.input_channels,
            out_channels=gate_channels, kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding, dilation=self.
            dilation, bias=bias)
        self.conv_hh = nn.Conv2d(in_channels=self.hidden_channels,
            out_channels=gate_channels, kernel_size=hidden_kernel_size,
            stride=1, padding=hidden_padding, dilation=1, bias=bias)
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


class EncoderCell(nn.Module):

    def __init__(self):
        super(EncoderCell, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.rnn1 = ConvLSTMCell(64, 256, kernel_size=3, stride=2, padding=
            1, hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTMCell(256, 512, kernel_size=3, stride=2, padding
            =1, hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTMCell(512, 512, kernel_size=3, stride=2, padding
            =1, hidden_kernel_size=1, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):

    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):

    def __init__(self):
        super(DecoderCell, self).__init__()
        self.conv1 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0,
            bias=False)
        self.rnn1 = ConvLSTMCell(512, 512, kernel_size=3, stride=1, padding
            =1, hidden_kernel_size=1, bias=False)
        self.rnn2 = ConvLSTMCell(128, 512, kernel_size=3, stride=1, padding
            =1, hidden_kernel_size=1, bias=False)
        self.rnn3 = ConvLSTMCell(128, 256, kernel_size=3, stride=1, padding
            =1, hidden_kernel_size=3, bias=False)
        self.rnn4 = ConvLSTMCell(64, 128, kernel_size=3, stride=1, padding=
            1, hidden_kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        x = self.conv1(input)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)
        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)
        x = F.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_1zb_pytorch_image_comp_rnn(_paritybench_base):
    pass
