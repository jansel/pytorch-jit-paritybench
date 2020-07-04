import sys
_module = sys.modules[__name__]
del sys
model = _module
solver = _module
model = _module
solver = _module
model = _module
solver = _module
model = _module
solver = _module
model = _module
solver = _module
model = _module
solver = _module
model = _module
solver = _module
model = _module
solver = _module
data = _module
dataset = _module
main = _module
progress_bar = _module
super_resolve = _module

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


import math


import torch.nn as nn


from math import log10


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import torch.nn.functional as F


from torchvision.models.vgg import vgg16


import torch.nn.init as init


class DBPN(nn.Module):

    def __init__(self, num_channels, base_channels, feat_channels,
        num_stages, scale_factor):
        super(DBPN, self).__init__()
        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning('please choose the scale factor from 2, 4, 8')
            exit()
        self.feat0 = ConvBlock(num_channels, feat_channels, 3, 1, 1,
            activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, 1, 1, 0,
            activation='prelu', norm=None)
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down2 = D_DownBlock(base_channels, kernel_size, stride, padding, 2
            )
        self.up3 = D_UpBlock(base_channels, kernel_size, stride, padding, 2)
        self.down3 = D_DownBlock(base_channels, kernel_size, stride, padding, 3
            )
        self.up4 = D_UpBlock(base_channels, kernel_size, stride, padding, 3)
        self.down4 = D_DownBlock(base_channels, kernel_size, stride, padding, 4
            )
        self.up5 = D_UpBlock(base_channels, kernel_size, stride, padding, 4)
        self.down5 = D_DownBlock(base_channels, kernel_size, stride, padding, 5
            )
        self.up6 = D_UpBlock(base_channels, kernel_size, stride, padding, 5)
        self.down6 = D_DownBlock(base_channels, kernel_size, stride, padding, 6
            )
        self.up7 = D_UpBlock(base_channels, kernel_size, stride, padding, 6)
        self.output_conv = ConvBlock(num_stages * base_channels,
            num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self):
        for m in self._modules:
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)
        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        return x


class DBPNS(nn.Module):

    def __init__(self, num_channels, base_channels, feat_channels,
        num_stages, scale_factor):
        super(DBPNS, self).__init__()
        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning('please choose the scale factor from 2, 4, 8')
            exit()
        self.feat0 = ConvBlock(num_channels, feat_channels, 3, 1, 1,
            activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, 1, 1, 0,
            activation='prelu', norm=None)
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        self.output_conv = ConvBlock(num_stages * base_channels,
            num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self):
        for m in self._modules:
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        x = self.output_conv(torch.cat((h2, h1), 1))
        return x


class DBPNLL(nn.Module):

    def __init__(self, num_channels, base_channels, feat_channels,
        num_stages, scale_factor):
        super(DBPNLL, self).__init__()
        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning('please choose the scale factor from 2, 4, 8')
            exit()
        self.feat0 = ConvBlock(num_channels, feat_channels, 3, 1, 1,
            activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, 1, 1, 0,
            activation='prelu', norm=None)
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down2 = D_DownBlock(base_channels, kernel_size, stride, padding, 2
            )
        self.up3 = D_UpBlock(base_channels, kernel_size, stride, padding, 2)
        self.down3 = D_DownBlock(base_channels, kernel_size, stride, padding, 3
            )
        self.up4 = D_UpBlock(base_channels, kernel_size, stride, padding, 3)
        self.down4 = D_DownBlock(base_channels, kernel_size, stride, padding, 4
            )
        self.up5 = D_UpBlock(base_channels, kernel_size, stride, padding, 4)
        self.down5 = D_DownBlock(base_channels, kernel_size, stride, padding, 5
            )
        self.up6 = D_UpBlock(base_channels, kernel_size, stride, padding, 5)
        self.down6 = D_DownBlock(base_channels, kernel_size, stride, padding, 6
            )
        self.up7 = D_UpBlock(base_channels, kernel_size, stride, padding, 6)
        self.down7 = D_DownBlock(base_channels, kernel_size, stride, padding, 7
            )
        self.up8 = D_UpBlock(base_channels, kernel_size, stride, padding, 7)
        self.down8 = D_DownBlock(base_channels, kernel_size, stride, padding, 8
            )
        self.up9 = D_UpBlock(base_channels, kernel_size, stride, padding, 8)
        self.down9 = D_DownBlock(base_channels, kernel_size, stride, padding, 9
            )
        self.up10 = D_UpBlock(base_channels, kernel_size, stride, padding, 9)
        self.output_conv = ConvBlock(num_stages * base_channels,
            num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self):
        for m in self._modules:
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)
        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down7(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up8(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down8(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up9(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        l = self.down9(concat_h)
        concat_l = torch.cat((l, concat_l), 1)
        h = self.up10(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        return x


class DenseBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, activation=
        'relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
        padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size,
            stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size=4, stride=2,
        padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size,
            kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias
        =True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size,
            stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size,
            stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)
        if self.activation is not None:
            out = self.act(out)
        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)
        out = torch.add(out, residual)
        return out


class UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias
        =True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class UpBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        scale=4, bias=True, activation='prelu', norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
            activation=activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
            activation=activation, norm=None)
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias
        =True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        scale=4, bias=True, activation='prelu', norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
            activation=activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(torch.nn.Module):

    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2,
        num_stages=1, scale=4, bias=True, activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0,
            activation=activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size,
            stride, padding, activation=activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, scale_factor, kernel_size=3,
        stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor **
            2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):

    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias,
                activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
        self.up = torch.nn.Sequential(*modules)
        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, upsample=
        'deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                kernel_size=4, stride=2, padding=1, bias=bias, activation=
                activation, norm=norm)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=
                scale_factor, bias=bias, activation=activation, norm=norm)
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(torch.nn.Upsample(
                scale_factor=scale_factor, mode='nearest'), ConvBlock(
                input_size, output_size, kernel_size=3, stride=1, padding=1,
                bias=bias, activation=activation, norm=norm))

    def forward(self, x):
        out = self.upsample(x)
        return out


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


class Net(torch.nn.Module):

    def __init__(self, num_channels, base_channel, num_recursions, device):
        super(Net, self).__init__()
        self.num_recursions = num_recursions
        self.embedding_layer = nn.Sequential(nn.Conv2d(num_channels,
            base_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(
            inplace=True), nn.Conv2d(base_channel, base_channel,
            kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(nn.Conv2d(base_channel,
            base_channel, kernel_size=3, stride=1, padding=1), nn.ReLU(
            inplace=True))
        self.reconstruction_layer = nn.Sequential(nn.Conv2d(base_channel,
            base_channel, kernel_size=3, stride=1, padding=1), nn.Conv2d(
            base_channel, num_channels, kernel_size=3, stride=1, padding=1))
        self.w_init = torch.ones(self.num_recursions) / self.num_recursions
        self.w = self.w_init

    def forward(self, x):
        h0 = self.embedding_layer(x)
        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))
        y_d_ = list()
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d + 1]))
            out_sum += torch.mul(y_d_[d], self.w[d])
        out_sum = torch.mul(out_sum, 1.0 / torch.sum(self.w))
        final_out = torch.add(out_sum, x)
        return y_d_, final_out

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Net(nn.Module):

    def __init__(self, num_channels, base_channel, upscale_factor,
        num_residuals):
        super(Net, self).__init__()
        self.input_conv = nn.Conv2d(num_channels, base_channel, kernel_size
            =3, stride=1, padding=1)
        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_channel, kernel=3, stride
                =1, padding=1))
        self.residual_layers = nn.Sequential(*resnet_blocks)
        self.mid_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3,
            stride=1, padding=1)
        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(PixelShuffleBlock(base_channel, base_channel,
                upscale_factor=2))
        self.upscale_layers = nn.Sequential(*upscale)
        self.output_conv = nn.Conv2d(base_channel, num_channels,
            kernel_size=3, stride=1, padding=1)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.residual_layers(x)
        x = self.mid_conv(x)
        x = torch.add(x, residual)
        x = self.upscale_layers(x)
        x = self.output_conv(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride,
            padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride,
            padding)
        self.bn = nn.BatchNorm2d(num_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.bn(self.conv1(x))
        x = self.activation(x)
        x = self.bn(self.conv2(x))
        x = torch.add(x, residual)
        return x


class PixelShuffleBlock(nn.Module):

    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3,
        stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2,
            kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.ps(self.conv(x))
        return x


class Net(torch.nn.Module):

    def __init__(self, num_channels, upscale_factor, d=64, s=12, m=4):
        super(Net, self).__init__()
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=num_channels,
            out_channels=d, kernel_size=5, stride=1, padding=2), nn.PReLU())
        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d,
            out_channels=s, kernel_size=1, stride=1, padding=0), nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s,
                kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s,
            out_channels=d, kernel_size=1, stride=1, padding=0), nn.PReLU()))
        self.mid_part = torch.nn.Sequential(*self.layers)
        self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=
            num_channels, kernel_size=9, stride=upscale_factor, padding=3,
            output_padding=1)

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()


class Net(torch.nn.Module):

    def __init__(self, num_channels, base_filter, upscale_factor=2):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(nn.Conv2d(in_channels=
            num_channels, out_channels=base_filter, kernel_size=9, stride=1,
            padding=4, bias=True), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=base_filter, out_channels=base_filter // 2,
            kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=base_filter // 2, out_channels=num_channels * 
            upscale_factor ** 2, kernel_size=5, stride=1, padding=2, bias=
            True), nn.PixelShuffle(upscale_factor))

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def swish(x):
    return x * F.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel, stride=stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=
            kernel, stride=stride, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3,
            stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


class Generator(nn.Module):

    def __init__(self, n_residual_blocks, upsample_factor, num_channel=1,
        base_filter=64):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=9,
            stride=1, padding=4)
        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), ResidualBlock(
                in_channels=base_filter, out_channels=base_filter, kernel=3,
                stride=1))
        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3,
            stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)
        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), UpsampleBlock(base_filter)
                )
        self.conv3 = nn.Conv2d(base_filter, num_channel, kernel_size=9,
            stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)
        x = self.bn2(self.conv2(y)) + x
        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)
        return self.conv3(x)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):

    def __init__(self, num_channel=1, base_filter=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=3,
            stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3,
            stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)
        self.conv3 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3,
            stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filter * 2)
        self.conv4 = nn.Conv2d(base_filter * 2, base_filter * 2,
            kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filter * 2)
        self.conv5 = nn.Conv2d(base_filter * 2, base_filter * 4,
            kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filter * 4)
        self.conv6 = nn.Conv2d(base_filter * 4, base_filter * 4,
            kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(base_filter * 4)
        self.conv7 = nn.Conv2d(base_filter * 4, base_filter * 8,
            kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(base_filter * 8)
        self.conv8 = nn.Conv2d(base_filter * 8, base_filter * 8,
            kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(base_filter * 8)
        self.conv9 = nn.Conv2d(base_filter * 8, num_channel, kernel_size=1,
            stride=1, padding=0)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0
            ], -1)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Net(nn.Module):

    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3,
            stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        return x


class Net(nn.Module):

    def __init__(self, num_channels, base_channels, num_residuals):
        super(Net, self).__init__()
        self.input_conv = nn.Sequential(nn.Conv2d(num_channels,
            base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.residual_layers = nn.Sequential(*[nn.Sequential(nn.Conv2d(
            base_channels, base_channels, kernel_size=3, stride=1, padding=
            1, bias=False), nn.ReLU(inplace=True)) for _ in range(
            num_residuals)])
        self.output_conv = nn.Conv2d(base_channels, num_channels,
            kernel_size=3, stride=1, padding=1, bias=False)

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)

    def forward(self, x):
        residual = x
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        x = torch.add(x, residual)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_icpm_super_resolution(_paritybench_base):
    pass
    def test_000(self):
        self._check(ConvBlock(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(D_DownBlock(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(D_DownBlockPix(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(D_UpBlock(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(D_UpBlockPix(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(DeconvBlock(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(DenseBlock(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4])], {})

    def test_007(self):
        self._check(Discriminator(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    def test_008(self):
        self._check(DownBlock(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(DownBlockPix(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(Generator(*[], **{'n_residual_blocks': 4, 'upsample_factor': 4}), [torch.rand([4, 1, 64, 64])], {})

    def test_011(self):
        self._check(Net(*[], **{'num_channels': 4, 'base_channels': 4, 'num_residuals': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(PSBlock(*[], **{'input_size': 4, 'output_size': 4, 'scale_factor': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(PixelShuffleBlock(*[], **{'in_channel': 4, 'out_channel': 4, 'upscale_factor': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(ResnetBlock(*[], **{'num_channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(UpBlock(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(UpBlockPix(*[], **{'num_filter': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(Upsample2xBlock(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(UpsampleBlock(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(Upsampler(*[], **{'scale': 1.0, 'n_feat': 4}), [torch.rand([4, 4, 4, 4])], {})

