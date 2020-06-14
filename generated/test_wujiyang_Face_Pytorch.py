import sys
_module = sys.modules[__name__]
del sys
backbone = _module
arcfacenet = _module
attention = _module
cbam = _module
mobilefacenet = _module
resnet = _module
spherenet = _module
pytorch2torchscript = _module
dataset = _module
agedb = _module
casia_webface = _module
cfp = _module
lfw = _module
lfw_2 = _module
megaface = _module
eval_agedb30 = _module
eval_cfp = _module
eval_deepglint_merge = _module
eval_lfw = _module
eval_lfw_blufr = _module
eval_megaface = _module
lossfunctions = _module
agentcenterloss = _module
centerloss = _module
ArcMarginProduct = _module
CosineMarginProduct = _module
InnerProduct = _module
MultiMarginProduct = _module
SphereMarginProduct = _module
margin = _module
train = _module
train_center = _module
train_softmax = _module
utils = _module
load_images_from_bin = _module
logging = _module
plot_logit = _module
plot_theta = _module
visualize = _module

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


import torch


from torch import nn


from collections import namedtuple


import torch.nn as nn


import numpy as np


import time


import math


import scipy.io


import torch.utils.data


from torch.nn import DataParallel


import torch.nn.functional as F


from torch.nn import Parameter


from torch.optim import lr_scheduler


import torch.optim as optim


from torch.utils.data import DataLoader


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class BottleNeck_IR(nn.Module):

    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeck_IR, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return shortcut + res


class BottleNeck_IR_SE(nn.Module):

    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeck_IR_SE, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel), SEModule(out_channel, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return shortcut + res


class Bottleneck(namedtuple('Block', ['in_channel', 'out_channel', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, out_channel, num_units, stride=2):
    return [Bottleneck(in_channel, out_channel, stride)] + [Bottleneck(
        out_channel, out_channel, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [get_block(in_channel=64, out_channel=64, num_units=3),
            get_block(in_channel=64, out_channel=128, num_units=4),
            get_block(in_channel=128, out_channel=256, num_units=14),
            get_block(in_channel=256, out_channel=512, num_units=3)]
    elif num_layers == 100:
        blocks = [get_block(in_channel=64, out_channel=64, num_units=3),
            get_block(in_channel=64, out_channel=128, num_units=13),
            get_block(in_channel=128, out_channel=256, num_units=30),
            get_block(in_channel=256, out_channel=512, num_units=3)]
    elif num_layers == 152:
        blocks = [get_block(in_channel=64, out_channel=64, num_units=3),
            get_block(in_channel=64, out_channel=128, num_units=8),
            get_block(in_channel=128, out_channel=256, num_units=36),
            get_block(in_channel=256, out_channel=512, num_units=3)]
    return blocks


class SEResNet_IR(nn.Module):

    def __init__(self, num_layers, feature_dim=512, drop_ratio=0.4, mode='ir'):
        super(SEResNet_IR, self).__init__()
        assert num_layers in [50, 100, 152
            ], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'se_ir'], 'mode should be ir or se_ir'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleNeck_IR
        elif mode == 'se_ir':
            unit_module = BottleNeck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1,
            bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(
            drop_ratio), Flatten(), nn.Linear(512 * 7 * 7, feature_dim), nn
            .BatchNorm1d(feature_dim))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                    bottleneck.out_channel, bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.res_bottleneck = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            ReLU(inplace=True), nn.Conv2d(in_channel, out_channel // 4, 1, 
            1, bias=False), nn.BatchNorm2d(out_channel // 4), nn.ReLU(
            inplace=True), nn.Conv2d(out_channel // 4, out_channel // 4, 3,
            stride, padding=1, bias=False), nn.BatchNorm2d(out_channel // 4
            ), nn.ReLU(inplace=True), nn.Conv2d(out_channel // 4,
            out_channel, 1, 1, bias=False))
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1, stride, bias=
            False)

    def forward(self, x):
        res = x
        out = self.res_bottleneck(x)
        if self.in_channel != self.out_channel or self.stride != 1:
            res = self.shortcut(x)
        out += res
        return out


class AttentionModule_stage1(nn.Module):

    def __init__(self, in_channel, out_channel, size1=(56, 56), size2=(28, 
        28), size3=(14, 14)):
        super(AttentionModule_stage1, self).__init__()
        self.share_residual_block = ResidualBlock(in_channel, out_channel)
        self.trunk_branches = nn.Sequential(ResidualBlock(in_channel,
            out_channel), ResidualBlock(in_channel, out_channel))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block1 = ResidualBlock(in_channel, out_channel)
        self.skip_connect1 = ResidualBlock(in_channel, out_channel)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block2 = ResidualBlock(in_channel, out_channel)
        self.skip_connect2 = ResidualBlock(in_channel, out_channel)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block3 = nn.Sequential(ResidualBlock(in_channel,
            out_channel), ResidualBlock(in_channel, out_channel))
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.mask_block4 = ResidualBlock(in_channel, out_channel)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.mask_block5 = ResidualBlock(in_channel, out_channel)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.mask_block6 = nn.Sequential(nn.BatchNorm2d(out_channel), nn.
            ReLU(inplace=True), nn.Conv2d(out_channel, out_channel, 1, 1,
            bias=False), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 1, 1, bias=False), nn.Sigmoid()
            )
        self.last_block = ResidualBlock(in_channel, out_channel)

    def forward(self, x):
        x = self.share_residual_block(x)
        out_trunk = self.trunk_branches(x)
        out_pool1 = self.mpool1(x)
        out_block1 = self.mask_block1(out_pool1)
        out_skip_connect1 = self.skip_connect1(out_block1)
        out_pool2 = self.mpool2(out_block1)
        out_block2 = self.mask_block2(out_pool2)
        out_skip_connect2 = self.skip_connect2(out_block2)
        out_pool3 = self.mpool3(out_block2)
        out_block3 = self.mask_block3(out_pool3)
        out_inter3 = self.interpolation3(out_block3) + out_block2
        out = out_inter3 + out_skip_connect2
        out_block4 = self.mask_block4(out)
        out_inter2 = self.interpolation2(out_block4) + out_block1
        out = out_inter2 + out_skip_connect1
        out_block5 = self.mask_block5(out)
        out_inter1 = self.interpolation1(out_block5) + out_trunk
        out_block6 = self.mask_block6(out_inter1)
        out = 1 + out_block6 + out_trunk
        out_last = self.last_block(out)
        return out_last


class AttentionModule_stage2(nn.Module):

    def __init__(self, in_channels, out_channels, size1=(28, 28), size2=(14,
        14)):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(ResidualBlock(in_channels,
            out_channels), ResidualBlock(in_channels, out_channels))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels,
            out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = nn.Sequential(ResidualBlock(in_channels,
            out_channels), ResidualBlock(in_channels, out_channels))
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax4_blocks = nn.Sequential(nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels,
            out_channels, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(
            out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage3(nn.Module):

    def __init__(self, in_channels, out_channels, size1=(14, 14)):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(ResidualBlock(in_channels,
            out_channels), ResidualBlock(in_channels, out_channels))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(ResidualBlock(in_channels,
            out_channels), ResidualBlock(in_channels, out_channels))
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax2_blocks = nn.Sequential(nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels,
            out_channels, kernel_size=1, stride=1, bias=False), nn.Sigmoid())
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class ResidualAttentionNet_56(nn.Module):

    def __init__(self, feature_dim=512, drop_ratio=0.4):
        super(ResidualAttentionNet_56, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 512, 2)
        self.attention_module3 = AttentionModule_stage3(512, 512)
        self.residual_block4 = ResidualBlock(512, 512, 2)
        self.residual_block5 = ResidualBlock(512, 512)
        self.residual_block6 = ResidualBlock(512, 512)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(
            drop_ratio), Flatten(), nn.Linear(512 * 7 * 7, feature_dim), nn
            .BatchNorm1d(feature_dim))

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.output_layer(out)
        return out


class ResidualAttentionNet_92(nn.Module):

    def __init__(self, feature_dim=512, drop_ratio=0.4):
        super(ResidualAttentionNet_92, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(2048), nn.Dropout(
            drop_ratio), Flatten(), nn.Linear(2048 * 7 * 7, feature_dim),
            nn.BatchNorm1d(feature_dim))

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.output_layer(out)
        return out


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(nn.Module):
    """Squeeze and Excitation Module"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class CAModule(nn.Module):
    """Channel Attention Module"""

    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, channels //
            reduction, kernel_size=1, padding=0, bias=False), nn.ReLU(
            inplace=True), nn.Conv2d(channels // reduction, channels,
            kernel_size=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        x = self.sigmoid(x)
        return input * x


class SAModule(nn.Module):
    """Spatial Attention Module"""

    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return input * x


class BottleNeck_IR(nn.Module):
    """Improved Residual Bottlenecks"""

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        return shortcut + res


class BottleNeck_IR_SE(nn.Module):
    """Improved Residual Bottlenecks with Squeeze and Excitation Module"""

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SE, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel), SEModule(out_channel, 16))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        return shortcut + res


class BottleNeck_IR_CAM(nn.Module):
    """Improved Residual Bottlenecks with Channel Attention Module"""

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_CAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel), CAModule(out_channel, 16))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        return shortcut + res


class BottleNeck_IR_SAM(nn.Module):
    """Improved Residual Bottlenecks with Spatial Attention Module"""

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_SAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel), SAModule())
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        return shortcut + res


class BottleNeck_IR_CBAM(nn.Module):
    """Improved Residual Bottleneck with Channel Attention Module and Spatial Attention Module"""

    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR_CBAM, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel), nn.
            Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False), nn.
            BatchNorm2d(out_channel), nn.PReLU(out_channel), nn.Conv2d(
            out_channel, out_channel, (3, 3), stride, 1, bias=False), nn.
            BatchNorm2d(out_channel), CAModule(out_channel, 16), SAModule())
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel,
                out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        return shortcut + res


def get_layers(num_layers):
    if num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]


class BottleNeck(nn.Module):

    def __init__(self, inp, oup, stride, expansion):
        super(BottleNeck, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expansion, 1, 1, 0,
            bias=False), nn.BatchNorm2d(inp * expansion), nn.PReLU(inp *
            expansion), nn.Conv2d(inp * expansion, inp * expansion, 3,
            stride, 1, groups=inp * expansion, bias=False), nn.BatchNorm2d(
            inp * expansion), nn.PReLU(inp * expansion), nn.Conv2d(inp *
            expansion, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):

    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


MobileFaceNet_BottleNeck_Setting = [[2, 64, 5, 2], [4, 128, 1, 2], [2, 128,
    6, 1], [4, 128, 1, 2], [2, 128, 2, 1]]


class MobileFaceNet(nn.Module):

    def __init__(self, feature_dim=128, bottleneck_setting=
        MobileFaceNet_BottleNeck_Setting):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.cur_channel = 64
        block = BottleNeck
        self.blocks = self._make_layer(block, bottleneck_setting)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t))
                else:
                    layers.append(block(self.cur_channel, c, 1, t))
                self.cur_channel = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, feature_dim=512, drop_ratio=0.4,
        zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512 * block.
            expansion), nn.Dropout(drop_ratio), Flatten(), nn.Linear(512 *
            block.expansion * 7 * 7, feature_dim), nn.BatchNorm1d(feature_dim))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), nn.BatchNorm2d(planes * block.
                expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x


class Block(nn.Module):

    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        return x + short_cut


class SphereNet(nn.Module):

    def __init__(self, num_layers=20, feature_dim=512):
        super(SphereNet, self).__init__()
        assert num_layers in [20, 64
            ], 'SphereNet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 7, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) +
                ' IS NOT SUPPORTED! (sphere20 or sphere64)')
        filter_list = [3, 64, 128, 256, 512]
        block = Block
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1
            ], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2
            ], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3
            ], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4
            ], layers[3], stride=2)
        self.fc = nn.Linear(512 * 7 * 7, feature_dim)
        self.last_bn = nn.BatchNorm1d(feature_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.last_bn(x)
        return x


class AgentCenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, scale):
        super(AgentCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.
            feat_dim))

    def forward(self, x, labels):
        """
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        """
        cos_dis = F.linear(F.normalize(x), F.normalize(self.centers)
            ) * self.scale
        one_hot = torch.zeros_like(cos_dis)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        loss = one_hot * self.scale - one_hot * cos_dis
        return loss.mean()


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.
            feat_dim))

    def forward(self, x, labels):
        """
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size,
            self.num_classes) + torch.pow(self.centers, 2).sum(dim=1,
            keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classes = torch.arange(self.num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1000000000000.0)
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


class ArcMarginProduct(nn.Module):

    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.5,
        easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine - self.th > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.s
        return output


class CosineMarginProduct(nn.Module):

    def __init__(self, in_feature=128, out_feature=10575, s=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output


class InnerProduct(nn.Module):

    def __init__(self, in_feature=128, out_feature=10575):
        super(InnerProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        output = F.linear(input, self.weight)
        return output


class MultiMarginProduct(nn.Module):

    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m1=0.2,
        m2=0.35, easy_margin=False):
        super(MultiMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m1 = math.cos(m1)
        self.sin_m1 = math.sin(m1)
        self.th = math.cos(math.pi - m1)
        self.mm = math.sin(math.pi - m1) * m1

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m1 - sine * self.sin_m1
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine - self.th > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output - one_hot * self.m2
        output = output * self.s
        return output


class SphereMarginProduct(nn.Module):

    def __init__(self, in_feature, out_feature, m=4, base=1000.0, gamma=
        0.0001, power=2, lambda_min=5.0, iter=0):
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)
        self.margin_formula = [lambda x: x ** 0, lambda x: x ** 1, lambda x:
            2 * x ** 2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x **
            4 - 8 * x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]

    def forward(self, input, label):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma *
            self.iter) ** (-1 * self.power))
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta(-1, 1)
        cos_m_theta = self.margin_formula(self.m)(cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / math.pi).floor()
        phi_theta = (-1.0) ** k * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.
            cur_lambda)
        norm_of_feature = torch.norm(input, 2, 1)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        output *= norm_of_feature.view(-1, 1)
        return output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_wujiyang_Face_Pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(AgentCenterLoss(*[], **{'num_classes': 4, 'feat_dim': 4, 'scale': 1.0}), [torch.rand([4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(ArcMarginProduct(*[], **{}), [torch.rand([128, 128]), torch.zeros([4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(AttentionModule_stage3(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 14, 14])], {})

    def test_003(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Block(*[], **{'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(BottleNeck(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expansion': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(BottleNeck_IR(*[], **{'in_channel': 4, 'out_channel': 4, 'stride': 1, 'dim_match': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(BottleNeck_IR_SAM(*[], **{'in_channel': 4, 'out_channel': 4, 'stride': 1, 'dim_match': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(CAModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(ConvBlock(*[], **{'inp': 4, 'oup': 4, 'k': 4, 's': 4, 'p': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(InnerProduct(*[], **{}), [torch.rand([128, 128]), torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(MultiMarginProduct(*[], **{}), [torch.rand([128, 128]), torch.zeros([4], dtype=torch.int64)], {})

    def test_013(self):
        self._check(ResidualBlock(*[], **{'in_channel': 4, 'out_channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(SAModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(SEModule(*[], **{'channels': 4, 'reduction': 4}), [torch.rand([4, 4, 4, 4])], {})

