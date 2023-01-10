import sys
_module = sys.modules[__name__]
del sys
src = _module
config = _module
dataset = _module
batch_utils = _module
brats = _module
image_utils = _module
inference = _module
loss = _module
dice = _module
models = _module
augmentation_blocks = _module
layers = _module
unet = _module
train = _module
tta = _module
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


import random


import torch.nn.functional as F


from torch.utils.data._utils.collate import default_collate


import numpy as np


import torch


from sklearn.model_selection import KFold


from torch.utils.data.dataset import Dataset


from types import SimpleNamespace


import torch.optim


import torch.utils.data


from torch.cuda.amp import autocast


import torch.nn as nn


from random import randint


from random import random


from random import sample


from random import uniform


from torch import nn


from collections import OrderedDict


from torch import nn as nn


from torch.nn import functional as F


from torch.utils.checkpoint import checkpoint_sequential


import time


import pandas as pd


import torch.nn.parallel


from torch.cuda.amp import GradScaler


from torch.utils.tensorboard import SummaryWriter


from itertools import combinations


from itertools import product


from matplotlib import pyplot as plt


from numpy import logical_and as l_and


from numpy import logical_not as l_not


from scipy.spatial.distance import directed_hausdorff


from torch import distributed as dist


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ['ET', 'TC', 'WT']
        self.device = 'cpu'

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.0
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)
        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                None
                if inputs.sum() == 0:
                    return torch.tensor(1.0, device='cuda')
                else:
                    return torch.tensor(0.0, device='cuda')
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = 2 * intersection / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class DataAugmenter(nn.Module):
    """Performs random flip and rotation batch wise, and reverse it if needed.
    Works"""

    def __init__(self, p=0.5, noise_only=False, channel_shuffling=False, drop_channnel=False):
        super(DataAugmenter, self).__init__()
        self.p = p
        self.transpose = []
        self.flip = []
        self.toggle = False
        self.noise_only = noise_only
        self.channel_shuffling = channel_shuffling
        self.drop_channel = drop_channnel

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:
                x = x * uniform(0.9, 1.1)
                std_per_channel = torch.stack(list(torch.std(x[:, i][x[:, i] > 0]) for i in range(x.size(1))))
                noise = torch.stack([torch.normal(0, std * 0.1, size=x[0, 0].shape) for std in std_per_channel])
                x = x + noise
                if random() < 0.2 and self.channel_shuffling:
                    new_channel_order = sample(range(x.size(1)), x.size(1))
                    x = x[:, new_channel_order]
                    None
                if random() < 0.2 and self.drop_channel:
                    x[:, sample(range(x.size(1)), 1)] = 0
                    None
                if self.noise_only:
                    return x
                self.transpose = sample(range(2, x.dim()), 2)
                self.flip = randint(2, x.dim() - 1)
                self.toggle = not self.toggle
                new_x = x.transpose(*self.transpose).flip(self.flip)
                return new_x
            else:
                return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle
            if isinstance(x, list):
                seg, deeps = x
                reversed_seg = seg.flip(self.flip).transpose(*self.transpose)
                reversed_deep = [deep.flip(self.flip).transpose(*self.transpose) for deep in deeps]
                return reversed_seg, reversed_deep
            else:
                return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=bias, dilation=dilation)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(OrderedDict([('conv', conv3x3(inplanes, planes, dilation=dilation)), ('bn', norm_layer(planes)), ('relu', nn.ReLU(inplace=True)), ('dropout', nn.Dropout(p=dropout))]))
        else:
            super(ConvBnRelu, self).__init__(OrderedDict([('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)), ('relu', nn.ReLU(inplace=True)), ('dropout', nn.Dropout(p=dropout))]))


class UBlock(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(OrderedDict([('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)), ('ConvBnRelu2', ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout))]))


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(inplace=True), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class UBlockCbam(nn.Sequential):

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockCbam, self).__init__(OrderedDict([('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)), ('CBAM', CBAM(outplanes, norm_layer=norm_layer))]))


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = 'Unet'

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [(width * 2 ** i) for i in range(4)]
        None
        self.deep_supervision = deep_supervision
        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)
        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)
        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)
        self.downsample = nn.MaxPool3d(2, 2)
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.outconv = conv1x1(features[0] // 2, num_classes)
        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(conv1x1(features[3], num_classes), nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True))
            self.deep_bottom2 = nn.Sequential(conv1x1(features[2], num_classes), nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True))
            self.deep3 = nn.Sequential(conv1x1(features[1], num_classes), nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            self.deep2 = nn.Sequential(conv1x1(features[0], num_classes), nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)
        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))
        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))
        out = self.outconv(up1)
        if self.deep_supervision:
            deeps = []
            for seg, deep in zip([bottom, bottom_2, up3, up2], [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps
        return out


class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = 'EquiUnet'

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [(width * 2 ** i) for i in range(4)]
        None
        self.deep_supervision = deep_supervision
        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)
        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)
        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)
        self.downsample = nn.MaxPool3d(2, 2)
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.outconv = conv1x1(features[0], num_classes)
        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(conv1x1(features[3], num_classes), nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True))
            self.deep_bottom2 = nn.Sequential(conv1x1(features[2], num_classes), nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True))
            self.deep3 = nn.Sequential(conv1x1(features[1], num_classes), nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            self.deep2 = nn.Sequential(conv1x1(features[0], num_classes), nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        self._init_weights()


class Att_EquiUnet(Unet):

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [(width * 2 ** i) for i in range(4)]
        None
        self.deep_supervision = deep_supervision
        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)
        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)
        self.bottom_2 = nn.Sequential(ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout), CBAM(features[2], norm_layer=norm_layer))
        self.downsample = nn.MaxPool3d(2, 2)
        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.outconv = conv1x1(features[0], num_classes)
        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(conv1x1(features[3], num_classes), nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True))
            self.deep_bottom2 = nn.Sequential(conv1x1(features[2], num_classes), nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True))
            self.deep3 = nn.Sequential(conv1x1(features[1], num_classes), nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))
            self.deep2 = nn.Sequential(conv1x1(features[0], num_classes), nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        self._init_weights()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnRelu,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DataAugmenter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UBlock,
     lambda: ([], {'inplanes': 4, 'midplanes': 4, 'outplanes': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UBlockCbam,
     lambda: ([], {'inplanes': 4, 'midplanes': 4, 'outplanes': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
]

class Test_lescientifik_open_brats2020(_paritybench_base):
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

