import sys
_module = sys.modules[__name__]
del sys
dng_to_png = _module
evaluate_accuracy = _module
load_data = _module
model = _module
msssim = _module
test_model = _module
train_model = _module
utils = _module
vgg = _module

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


from torch.utils.data import DataLoader


import numpy as np


import torch


import math


import torch.nn as nn


import torch.nn.functional as F


from math import exp


from scipy import misc


from torch.optim import Adam


class PyNET(nn.Module):

    def __init__(self, level, instance_norm=True, instance_norm_level_1=False):
        super(PyNET, self).__init__()
        self.level = level
        self.conv_l1_d1 = ConvMultiBlock(4, 32, 3, instance_norm=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv_l2_d1 = ConvMultiBlock(32, 64, 3, instance_norm=instance_norm
            )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv_l3_d1 = ConvMultiBlock(64, 128, 3, instance_norm=
            instance_norm)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv_l4_d1 = ConvMultiBlock(128, 256, 3, instance_norm=
            instance_norm)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv_l5_d1 = ConvMultiBlock(256, 512, 3, instance_norm=
            instance_norm)
        self.conv_l5_d2 = ConvMultiBlock(512, 512, 3, instance_norm=
            instance_norm)
        self.conv_l5_d3 = ConvMultiBlock(512, 512, 3, instance_norm=
            instance_norm)
        self.conv_l5_d4 = ConvMultiBlock(512, 512, 3, instance_norm=
            instance_norm)
        self.conv_t4a = UpsampleConvLayer(512, 256, 3)
        self.conv_t4b = UpsampleConvLayer(512, 256, 3)
        self.conv_l5_out = ConvLayer(512, 3, kernel_size=3, stride=1, relu=
            False)
        self.output_l5 = nn.Sigmoid()
        self.conv_l4_d3 = ConvMultiBlock(512, 256, 3, instance_norm=
            instance_norm)
        self.conv_l4_d4 = ConvMultiBlock(256, 256, 3, instance_norm=
            instance_norm)
        self.conv_l4_d5 = ConvMultiBlock(256, 256, 3, instance_norm=
            instance_norm)
        self.conv_l4_d6 = ConvMultiBlock(256, 256, 3, instance_norm=
            instance_norm)
        self.conv_l4_d8 = ConvMultiBlock(512, 256, 3, instance_norm=
            instance_norm)
        self.conv_t3a = UpsampleConvLayer(256, 128, 3)
        self.conv_t3b = UpsampleConvLayer(256, 128, 3)
        self.conv_l4_out = ConvLayer(256, 3, kernel_size=3, stride=1, relu=
            False)
        self.output_l4 = nn.Sigmoid()
        self.conv_l3_d3 = ConvMultiBlock(256, 128, 5, instance_norm=
            instance_norm)
        self.conv_l3_d4 = ConvMultiBlock(256, 128, 5, instance_norm=
            instance_norm)
        self.conv_l3_d5 = ConvMultiBlock(256, 128, 5, instance_norm=
            instance_norm)
        self.conv_l3_d6 = ConvMultiBlock(256, 128, 5, instance_norm=
            instance_norm)
        self.conv_l3_d8 = ConvMultiBlock(512, 128, 3, instance_norm=
            instance_norm)
        self.conv_t2a = UpsampleConvLayer(128, 64, 3)
        self.conv_t2b = UpsampleConvLayer(128, 64, 3)
        self.conv_l3_out = ConvLayer(128, 3, kernel_size=3, stride=1, relu=
            False)
        self.output_l3 = nn.Sigmoid()
        self.conv_l2_d3 = ConvMultiBlock(128, 64, 5, instance_norm=
            instance_norm)
        self.conv_l2_d5 = ConvMultiBlock(192, 64, 7, instance_norm=
            instance_norm)
        self.conv_l2_d6 = ConvMultiBlock(192, 64, 7, instance_norm=
            instance_norm)
        self.conv_l2_d7 = ConvMultiBlock(192, 64, 7, instance_norm=
            instance_norm)
        self.conv_l2_d8 = ConvMultiBlock(192, 64, 7, instance_norm=
            instance_norm)
        self.conv_l2_d10 = ConvMultiBlock(256, 64, 5, instance_norm=
            instance_norm)
        self.conv_l2_d12 = ConvMultiBlock(192, 64, 3, instance_norm=
            instance_norm)
        self.conv_t1a = UpsampleConvLayer(64, 32, 3)
        self.conv_t1b = UpsampleConvLayer(64, 32, 3)
        self.conv_l2_out = ConvLayer(64, 3, kernel_size=3, stride=1, relu=False
            )
        self.output_l2 = nn.Sigmoid()
        self.conv_l1_d3 = ConvMultiBlock(64, 32, 5, instance_norm=False)
        self.conv_l1_d5 = ConvMultiBlock(96, 32, 7, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d6 = ConvMultiBlock(96, 32, 9, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d7 = ConvMultiBlock(128, 32, 9, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d8 = ConvMultiBlock(128, 32, 9, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d9 = ConvMultiBlock(128, 32, 9, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d10 = ConvMultiBlock(128, 32, 7, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d12 = ConvMultiBlock(128, 32, 5, instance_norm=
            instance_norm_level_1)
        self.conv_l1_d14 = ConvMultiBlock(128, 32, 3, instance_norm=False)
        self.conv_l1_out = ConvLayer(32, 3, kernel_size=3, stride=1, relu=False
            )
        self.output_l1 = nn.Sigmoid()
        self.conv_t0 = UpsampleConvLayer(32, 16, 3)
        self.conv_l0_d1 = ConvLayer(16, 3, kernel_size=3, stride=1, relu=False)
        self.output_l0 = nn.Sigmoid()

    def level_5(self, pool4):
        conv_l5_d1 = self.conv_l5_d1(pool4)
        conv_l5_d2 = self.conv_l5_d2(conv_l5_d1)
        conv_l5_d3 = self.conv_l5_d3(conv_l5_d2)
        conv_l5_d4 = self.conv_l5_d4(conv_l5_d3)
        conv_t4a = self.conv_t4a(conv_l5_d4)
        conv_t4b = self.conv_t4b(conv_l5_d4)
        conv_l5_out = self.conv_l5_out(conv_l5_d4)
        output_l5 = self.output_l5(conv_l5_out)
        return output_l5, conv_t4a, conv_t4b

    def level_4(self, conv_l4_d1, conv_t4a, conv_t4b):
        conv_l4_d2 = torch.cat([conv_l4_d1, conv_t4a], 1)
        conv_l4_d3 = self.conv_l4_d3(conv_l4_d2)
        conv_l4_d4 = self.conv_l4_d4(conv_l4_d3) + conv_l4_d3
        conv_l4_d5 = self.conv_l4_d5(conv_l4_d4) + conv_l4_d4
        conv_l4_d6 = self.conv_l4_d6(conv_l4_d5)
        conv_l4_d7 = torch.cat([conv_l4_d6, conv_t4b], 1)
        conv_l4_d8 = self.conv_l4_d8(conv_l4_d7)
        conv_t3a = self.conv_t3a(conv_l4_d8)
        conv_t3b = self.conv_t3b(conv_l4_d8)
        conv_l4_out = self.conv_l4_out(conv_l4_d8)
        output_l4 = self.output_l4(conv_l4_out)
        return output_l4, conv_t3a, conv_t3b

    def level_3(self, conv_l3_d1, conv_t3a, conv_t3b):
        conv_l3_d2 = torch.cat([conv_l3_d1, conv_t3a], 1)
        conv_l3_d3 = self.conv_l3_d3(conv_l3_d2) + conv_l3_d2
        conv_l3_d4 = self.conv_l3_d4(conv_l3_d3) + conv_l3_d3
        conv_l3_d5 = self.conv_l3_d5(conv_l3_d4) + conv_l3_d4
        conv_l3_d6 = self.conv_l3_d6(conv_l3_d5)
        conv_l3_d7 = torch.cat([conv_l3_d6, conv_l3_d1, conv_t3b], 1)
        conv_l3_d8 = self.conv_l3_d8(conv_l3_d7)
        conv_t2a = self.conv_t2a(conv_l3_d8)
        conv_t2b = self.conv_t2b(conv_l3_d8)
        conv_l3_out = self.conv_l3_out(conv_l3_d8)
        output_l3 = self.output_l3(conv_l3_out)
        return output_l3, conv_t2a, conv_t2b

    def level_2(self, conv_l2_d1, conv_t2a, conv_t2b):
        conv_l2_d2 = torch.cat([conv_l2_d1, conv_t2a], 1)
        conv_l2_d3 = self.conv_l2_d3(conv_l2_d2)
        conv_l2_d4 = torch.cat([conv_l2_d3, conv_l2_d1], 1)
        conv_l2_d5 = self.conv_l2_d5(conv_l2_d4) + conv_l2_d4
        conv_l2_d6 = self.conv_l2_d6(conv_l2_d5) + conv_l2_d5
        conv_l2_d7 = self.conv_l2_d7(conv_l2_d6) + conv_l2_d6
        conv_l2_d8 = self.conv_l2_d8(conv_l2_d7)
        conv_l2_d9 = torch.cat([conv_l2_d8, conv_l2_d1], 1)
        conv_l2_d10 = self.conv_l2_d10(conv_l2_d9)
        conv_l2_d11 = torch.cat([conv_l2_d10, conv_t2b], 1)
        conv_l2_d12 = self.conv_l2_d12(conv_l2_d11)
        conv_t1a = self.conv_t1a(conv_l2_d12)
        conv_t1b = self.conv_t1b(conv_l2_d12)
        conv_l2_out = self.conv_l2_out(conv_l2_d12)
        output_l2 = self.output_l2(conv_l2_out)
        return output_l2, conv_t1a, conv_t1b

    def level_1(self, conv_l1_d1, conv_t1a, conv_t1b):
        conv_l1_d2 = torch.cat([conv_l1_d1, conv_t1a], 1)
        conv_l1_d3 = self.conv_l1_d3(conv_l1_d2)
        conv_l1_d4 = torch.cat([conv_l1_d3, conv_l1_d1], 1)
        conv_l1_d5 = self.conv_l1_d5(conv_l1_d4)
        conv_l1_d6 = self.conv_l1_d6(conv_l1_d5)
        conv_l1_d7 = self.conv_l1_d7(conv_l1_d6) + conv_l1_d6
        conv_l1_d8 = self.conv_l1_d8(conv_l1_d7) + conv_l1_d7
        conv_l1_d9 = self.conv_l1_d9(conv_l1_d8) + conv_l1_d8
        conv_l1_d10 = self.conv_l1_d10(conv_l1_d9)
        conv_l1_d11 = torch.cat([conv_l1_d10, conv_l1_d1], 1)
        conv_l1_d12 = self.conv_l1_d12(conv_l1_d11)
        conv_l1_d13 = torch.cat([conv_l1_d12, conv_t1b, conv_l1_d1], 1)
        conv_l1_d14 = self.conv_l1_d14(conv_l1_d13)
        conv_t0 = self.conv_t0(conv_l1_d14)
        conv_l1_out = self.conv_l1_out(conv_l1_d14)
        output_l1 = self.output_l1(conv_l1_out)
        return output_l1, conv_t0

    def level_0(self, conv_t0):
        conv_l0_d1 = self.conv_l0_d1(conv_t0)
        output_l0 = self.output_l0(conv_l0_d1)
        return output_l0

    def forward(self, x):
        conv_l1_d1 = self.conv_l1_d1(x)
        pool1 = self.pool1(conv_l1_d1)
        conv_l2_d1 = self.conv_l2_d1(pool1)
        pool2 = self.pool2(conv_l2_d1)
        conv_l3_d1 = self.conv_l3_d1(pool2)
        pool3 = self.pool3(conv_l3_d1)
        conv_l4_d1 = self.conv_l4_d1(pool3)
        pool4 = self.pool4(conv_l4_d1)
        output_l5, conv_t4a, conv_t4b = self.level_5(pool4)
        if self.level < 5:
            output_l4, conv_t3a, conv_t3b = self.level_4(conv_l4_d1,
                conv_t4a, conv_t4b)
        if self.level < 4:
            output_l3, conv_t2a, conv_t2b = self.level_3(conv_l3_d1,
                conv_t3a, conv_t3b)
        if self.level < 3:
            output_l2, conv_t1a, conv_t1b = self.level_2(conv_l2_d1,
                conv_t2a, conv_t2b)
        if self.level < 2:
            output_l1, conv_t0 = self.level_1(conv_l1_d1, conv_t1a, conv_t1b)
        if self.level < 1:
            output_l0 = self.level_0(conv_t0)
        if self.level == 0:
            enhanced = output_l0
        if self.level == 1:
            enhanced = output_l1
        if self.level == 2:
            enhanced = output_l2
        if self.level == 3:
            enhanced = output_l3
        if self.level == 4:
            enhanced = output_l4
        if self.level == 5:
            enhanced = output_l5
        return enhanced


class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, max_conv_size, instance_norm
        ):
        super(ConvMultiBlock, self).__init__()
        self.max_conv_size = max_conv_size
        self.conv_3a = ConvLayer(in_channels, out_channels, kernel_size=3,
            stride=1, instance_norm=instance_norm)
        self.conv_3b = ConvLayer(out_channels, out_channels, kernel_size=3,
            stride=1, instance_norm=instance_norm)
        if max_conv_size >= 5:
            self.conv_5a = ConvLayer(in_channels, out_channels, kernel_size
                =5, stride=1, instance_norm=instance_norm)
            self.conv_5b = ConvLayer(out_channels, out_channels,
                kernel_size=5, stride=1, instance_norm=instance_norm)
        if max_conv_size >= 7:
            self.conv_7a = ConvLayer(in_channels, out_channels, kernel_size
                =7, stride=1, instance_norm=instance_norm)
            self.conv_7b = ConvLayer(out_channels, out_channels,
                kernel_size=7, stride=1, instance_norm=instance_norm)
        if max_conv_size >= 9:
            self.conv_9a = ConvLayer(in_channels, out_channels, kernel_size
                =9, stride=1, instance_norm=instance_norm)
            self.conv_9b = ConvLayer(out_channels, out_channels,
                kernel_size=9, stride=1, instance_norm=instance_norm)

    def forward(self, x):
        out_3 = self.conv_3a(x)
        output_tensor = self.conv_3b(out_3)
        if self.max_conv_size >= 5:
            out_5 = self.conv_5a(x)
            out_5 = self.conv_5b(out_5)
            output_tensor = torch.cat([output_tensor, out_5], 1)
        if self.max_conv_size >= 7:
            out_7 = self.conv_7a(x)
            out_7 = self.conv_7b(out_7)
            output_tensor = torch.cat([output_tensor, out_7], 1)
        if self.max_conv_size >= 9:
            out_9 = self.conv_9a(x)
            out_9 = self.conv_9b(out_9)
            output_tensor = torch.cat([output_tensor, out_9], 1)
        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu
        =True, instance_norm=False):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None
        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)
        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.instance_norm:
            out = self.instance(out)
        if self.relu:
            out = self.relu(out)
        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2,
        stride=1, relu=True):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear',
            align_corners=True)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size, stride)
        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)
        if self.relu:
            out = self.relu(out)
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 
        sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0
        )
    window = _2D_window.expand(channel, 1, window_size, window_size
        ).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=
    False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel
        ) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel
        ) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel
        ) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = (2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size,
            size_average=self.size_average)


def msssim(img1, img2, window_size=11, size_average=True, val_range=None,
    normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(
        device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=
            size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


class MSSSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size,
            size_average=self.size_average)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_aiff22_PyNET_PyTorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ConvLayer(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(ConvMultiBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'max_conv_size': 4, 'instance_norm': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(MSSSIM(*[], **{}), [torch.rand([4, 1, 64, 64]), torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(UpsampleConvLayer(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

