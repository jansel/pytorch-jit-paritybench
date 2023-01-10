import sys
_module = sys.modules[__name__]
del sys
depth_completion = _module
agent = _module
base_agent = _module
depth_completion_agent = _module
mul_model_dc_agent = _module
resnet18_Baseline_agent = _module
self_Attention_BC_agent = _module
self_Attention_agent = _module
config = _module
resnet18_Baseline_config = _module
self_Attention_BC_config = _module
self_Attention_config = _module
data = _module
data_loader = _module
eval = _module
main = _module
models = _module
model = _module
model_blocks = _module
unet_parts = _module
utils = _module
file_manager = _module
loss_func = _module
pytorch_ssim = _module
torch_utils = _module
SA_BC_SSIM_to_mat = _module
SA_SSIM_to_mat = _module
SA_to_mat = _module
parse_filename = _module
r18_to_mat = _module
raw_to_mat = _module
render_to_mat = _module
yindaz_to_mat = _module

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


import numpy as np


import scipy.io as sio


from matplotlib import cm


import torch


from torch import optim


from torch.utils.data import DataLoader


from abc import abstractmethod


import random


import warnings


from torch import Tensor


from torch.utils.data.dataset import Dataset


from torchvision import transforms


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


from torch.autograd import Variable


from math import exp


class BaseModel(torch.nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def save(self, filename):
        state_dict = {name: value.cpu() for name, value in self.state_dict().items()}
        status = {'state_dict': state_dict}
        with open(filename, 'wb') as f_model:
            torch.save(status, f_model)

    def load(self, filename):
        if filename is None:
            raise ValueError('Error when loading model: filename not given.')
        status = torch.load(filename)
        self.load_state_dict(status['state_dict'])


class GatedConvModel(BaseModel):

    def __init__(self, in_channel):
        super(GatedConvModel, self).__init__()
        self._in_channel = in_channel
        self._build_network()

    def _build_network(self):
        c_num = 48
        self.gconv_layer_1 = torch.nn.Sequential(g_conv(self._in_channel, c_num, 5, 1), g_conv(c_num, 2 * c_num, 3, 2))
        self.gconv_layer_2 = torch.nn.Sequential(g_conv(2 * c_num, 2 * c_num, 3, 1), g_conv(2 * c_num, 4 * c_num, 3, 2))
        self.gconv_layer_3 = torch.nn.Sequential(g_conv(4 * c_num, 4 * c_num, 3, 1), g_conv(4 * c_num, 4 * c_num, 3, 1))
        self.gconv_layer_4 = torch.nn.Sequential(g_conv(4 * c_num, 4 * c_num, 3, 1, dilation=2), g_conv(4 * c_num, 4 * c_num, 3, 1, dilation=4), g_conv(4 * c_num, 4 * c_num, 3, 1, dilation=8), g_conv(4 * c_num, 4 * c_num, 3, 1, dilation=16))
        self.gconv_layer_5 = torch.nn.Sequential(g_conv(4 * c_num, 4 * c_num, 3, 1), g_conv(4 * c_num, 4 * c_num, 3, 1), g_dconv(4 * c_num, 2 * c_num, 3, 1))
        self.gconv_layer_6 = torch.nn.Sequential(g_conv(2 * c_num, 2 * c_num, 3, 1), g_dconv(2 * c_num, c_num, 3, 1))
        self.gconv_layer_7 = torch.nn.Sequential(g_conv(c_num, c_num, 3, 1), g_conv(c_num, c_num // 2, 3, 1), g_conv(c_num // 2, 1, 3, 1))
        self.generator = torch.nn.Sequential(self.gconv_layer_1, self.gconv_layer_2, self.gconv_layer_3, self.gconv_layer_4, self.gconv_layer_5, self.gconv_layer_6, self.gconv_layer_7)

    def forward(self, x):
        out = self.generator(x)
        out = torch.clamp(out, min=0.0)
        return out


class GatedConvSkipConnectionModel(GatedConvModel):

    def __init__(self, in_channel):
        super().__init__(in_channel)
        pass

    def forward(self, x):
        out1 = self.gconv_layer_1(x)
        out2 = self.gconv_layer_2(out1)
        out3 = self.gconv_layer_3(out2)
        out4 = self.gconv_layer_4(out3) + out2
        out5 = self.gconv_layer_5(out4) + out1
        out6 = self.gconv_layer_6(out5)
        out = self.gconv_layer_7(out6)
        return out


class ResNet18(BaseModel):

    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self._build_network()

    def _build_network(self):
        c = 32
        self.pre = torch.nn.Sequential(torch.nn.Conv2d(self.in_channel, c, kernel_size=3, stride=1, padding=1, bias=False), torch.nn.BatchNorm2d(c), torch.nn.ReLU())
        self.layer_1 = self._make_layer(c, c, 2)
        self.layer_2 = self._make_layer(c, c * 2, 2)
        self.layer_3 = self._make_layer(c * 2, c * 4, 2)
        self.layer_4 = self._make_layer(c * 4, c * 8, 2)
        self.layer_5 = self._make_layer(c * 8, c * 4, 2)
        self.layer_6 = self._make_layer(c * 4, c * 2, 2)
        self.layer_7 = self._make_layer(c * 2, c, 2)
        self.layer_8 = self._make_layer(c, 1, 2)

    def _make_layer(self, in_channel, out_channel, block_num):
        shortcut = torch.nn.Sequential(torch.nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(out_channel))
        layers = []
        layers.append(r_block(in_channel, out_channel, kernel_size=3, padding=-1, shortcut=shortcut))
        for _ in range(1, block_num):
            layers.append(r_block(out_channel, out_channel, kernel_size=3, padding=-1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        return out


class ResNet18SkipConnection(ResNet18):

    def __init__(self, in_channel):
        super().__init__(in_channel)

    def forward(self, x):
        out = self.pre(x)
        out1 = self.layer_1(out)
        out2 = self.layer_2(out1)
        out3 = self.layer_3(out2)
        out4 = self.layer_4(out3)
        out5 = self.layer_5(out4) + out3
        out6 = self.layer_6(out5) + out2
        out7 = self.layer_7(out6) + out1
        out8 = self.layer_8(out7)
        return out8


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


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, shortcut=None):
        super().__init__()
        if padding == -1:
            padding = (np.array(kernel_size) - 1) * np.array(dilation) // 2
        self.left = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.BatchNorm2d(out_channels))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Conv3dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        if padding == -1:
            padding = tuple((np.array(kernel_size) - 1) * np.array(dilation) // 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == 'SN':
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f'Norm type {norm} not implemented')
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class NN3Dby2D(object):
    """
    Use these inner classes to mimic 3D operation by using 2D operation frame by frame.
    """


    class Base(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, xs):
            dim_length = len(xs.shape)
            if dim_length == 5:
                xs = torch.unbind(xs, dim=2)
                xs = torch.stack([self.layer(x) for x in xs], dim=2)
            elif dim_length == 4:
                xs = self.layer(xs)
            return xs


    class Conv3d(Base):

        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
            super().__init__()
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[1:]
            if isinstance(stride, tuple):
                stride = stride[1:]
            if isinstance(padding, tuple):
                padding = padding[1:]
            if isinstance(dilation, tuple):
                dilation = dilation[1:]
            self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.weight = self.layer.weight
            self.bias = self.layer.bias


    class BatchNorm3d(Base):

        def __init__(self, out_channels):
            super().__init__()
            self.layer = nn.BatchNorm2d(out_channels)


    class InstanceNorm3d(Base):

        def __init__(self, out_channels, track_running_stats=True):
            super().__init__()
            self.layer = nn.InstanceNorm2d(out_channels, track_running_stats=track_running_stats)


class GatedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), conv_by='2d'):
        super().__init__()
        if conv_by == '2d':
            module = NN3Dby2D
        elif conv_by == '3d':
            module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')
        if padding == -1:
            padding = (np.array(kernel_size) - 1) * np.array(dilation) // 2
        self.gatingConv = module.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.featureConv = module.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm
        if norm == 'BN':
            self.norm_layer = module.BatchNorm3d(out_channels)
        elif norm == 'IN':
            self.norm_layer = module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == 'SN':
            self.norm = None
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f'Norm type {norm} not implemented')
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = self.gated(gating) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class GatedDeconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=-1, dilation=1, groups=1, bias=True, norm='SN', activation=nn.LeakyReLU(0.2, inplace=True), scale_factor=2, conv_by='2d'):
        super().__init__()
        self.conv = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv3dBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GatedConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GatedDeconv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNet,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
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
     False),
]

class Test_tsunghan_wu_Depth_Completion(_paritybench_base):
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

