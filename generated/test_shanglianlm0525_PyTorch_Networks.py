import sys
_module = sys.modules[__name__]
del sys
CBAM = _module
GlobalContextBlock = _module
NonLocalBlock = _module
SENet = _module
SEvariants = _module
AlexNet = _module
DenseNet = _module
InceptionV1 = _module
InceptionV2 = _module
InceptionV3 = _module
InceptionV4 = _module
ResNeXt = _module
ResNet = _module
VGGNet = _module
FaceBoxes = _module
LFFD = _module
VarGFaceNet = _module
Hourglass = _module
LPN = _module
SimpleBaseline = _module
context_block = _module
PolarMask = _module
GhostNet = _module
MixNet = _module
MobileNetV1 = _module
MobileNetV2 = _module
MobileNetV3 = _module
ShuffleNet = _module
ShuffleNetV2 = _module
SqueezeNet = _module
Xception = _module
ASFF = _module
CenterNet = _module
CornerNet = _module
FCOS = _module
FPN = _module
FSAF = _module
FisheyeMODNet = _module
FoveaBox = _module
RetinaNet = _module
SSD = _module
YOLO = _module
YOLO_Nano = _module
YOLOv2 = _module
YOLOv3 = _module
SINet = _module
FCN = _module
FisheyeMODNet = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import math, functools, torchtext, typing, torchvision, warnings, logging, string, numpy, torchaudio, itertools, collections, re, random, scipy, copy, numbers, uuid, time, enum, types, inspect, torch, abc
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


import torch.nn as nn


import torchvision


from torch import nn


from functools import reduce


class ChannelAttentionModule(nn.Module):

    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(nn.Conv2d(channel, channel // ratio,
            1, bias=False), nn.ReLU(), nn.Conv2d(channel // ratio, channel,
            1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):

    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=
            7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):

    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False,
        expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places,
            out_channels=places, kernel_size=1, stride=1, bias=False), nn.
            BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=places, out_channels=places, kernel_size=3, stride=
            stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU
            (inplace=True), nn.Conv2d(in_channels=places, out_channels=
            places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion))
        self.cbam = CBAM(channel=places * self.expansion)
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places,
                out_channels=places * self.expansion, kernel_size=1, stride
                =stride, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class GlobalContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=(
        'channel_add',)):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class NonLocalBlock(nn.Module):

    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.
            inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.
            inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.
            inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel,
            out_channels=channel, kernel_size=1, stride=1, padding=0, bias=
            False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1
            ).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b,
            self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class SE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel,
            out_features=channel // ratio), nn.ReLU(inplace=True), nn.
            Linear(in_features=channel // ratio, out_features=channel), nn.
            Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class SE_ResNetBlock(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False,
        expansion=4):
        super(SE_ResNetBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places,
            out_channels=places, kernel_size=1, stride=1, bias=False), nn.
            BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=places, out_channels=places, kernel_size=3, stride=
            stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU
            (inplace=True), nn.Conv2d(in_channels=places, out_channels=
            places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places,
                out_channels=places * self.expansion, kernel_size=1, stride
                =stride, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=
        places, kernel_size=7, stride=stride, padding=3, bias=False), nn.
        BatchNorm2d(places), nn.ReLU(inplace=True), nn.MaxPool2d(
        kernel_size=3, stride=2, padding=1))


class SE_ResNet(nn.Module):

    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(SE_ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks
            [0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=
            blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=
            blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=
            blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(SE_ResNetBlock(in_places, places, stride,
            downsampling=True))
        for i in range(1, block):
            layers.append(SE_ResNetBlock(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class cSE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(cSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel,
            out_features=channel // ratio), nn.ReLU(inplace=True), nn.
            Linear(in_features=channel // ratio, out_features=channel), nn.
            Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSE_Module(nn.Module):

    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(nn.Conv2d(in_channels=
            channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(scSE_Module, self).__init__()
        self.cSE = cSE_Module(channel, ratio)
        self.sSE = sSE_Module(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(in_channels=3,
            out_channels=96, kernel_size=11, stride=4, padding=2, bias=
            False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2, padding=0), nn.Conv2d(in_channels=96, out_channels=
            192, kernel_size=5, stride=1, padding=2, bias=False), nn.ReLU(
            inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3,
            stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride
            =1, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1,
            padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0))
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(
            in_features=256 * 6 * 6, out_features=4096), nn.ReLU(inplace=
            True), nn.Dropout(p=0.5), nn.Linear(in_features=4096,
            out_features=4096), nn.ReLU(inplace=True), nn.Linear(
            in_features=4096, out_features=num_classes))

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class _TransitionLayer(nn.Module):

    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(nn.BatchNorm2d(inplace), nn.
            ReLU(inplace=True), nn.Conv2d(in_channels=inplace, out_channels
            =plance, kernel_size=1, stride=1, padding=0, bias=False), nn.
            AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition_layer(x)


class _DenseLayer(nn.Module):

    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(nn.BatchNorm2d(inplace), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels=inplace, out_channels=
            bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias
            =False), nn.BatchNorm2d(bn_size * growth_rate), nn.ReLU(inplace
            =True), nn.Conv2d(in_channels=bn_size * growth_rate,
            out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
            bias=False))
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):

    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0
        ):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate,
                growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):

    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24,
        16], num_classes=1000):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)
        blocks * 4
        num_features = init_channels
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=
            num_features, growth_rate=growth_rate, bn_size=bn_size,
            drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(inplace=num_features, plance=
            num_features // 2)
        num_features = num_features // 2
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=
            num_features, growth_rate=growth_rate, bn_size=bn_size,
            drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(inplace=num_features, plance=
            num_features // 2)
        num_features = num_features // 2
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=
            num_features, growth_rate=growth_rate, bn_size=bn_size,
            drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(inplace=num_features, plance=
            num_features // 2)
        num_features = num_features // 2
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=
            num_features, growth_rate=growth_rate, bn_size=bn_size,
            drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=kernel_size, stride=stride, padding=
        padding), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InceptionV1Module(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()
        self.branch1_conv = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce,
            out_channels=out_channels2, kernel_size=3)
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce,
            out_channels=out_channels3, kernel_size=5)
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels4, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionAux(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()
        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=128, kernel_size=1)
        self.auxiliary_linear1 = nn.Linear(in_features=128 * 4 * 4,
            out_features=1024)
        self.auxiliary_relu = nn.ReLU6(inplace=True)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear2 = nn.Linear(in_features=1024, out_features=
            out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.auxiliary_relu(self.auxiliary_linear1(x))
        out = self.auxiliary_linear2(self.auxiliary_dropout(x))
        return out


class InceptionV1(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV1, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn
            .MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1), nn.
            BatchNorm2d(64))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=
            192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV1Module(in_channels=192,
            out_channels1=64, out_channels2reduce=96, out_channels2=128,
            out_channels3reduce=16, out_channels3=32, out_channels4=32),
            InceptionV1Module(in_channels=256, out_channels1=128,
            out_channels2reduce=128, out_channels2=192, out_channels3reduce
            =32, out_channels3=96, out_channels4=64), nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1))
        self.block4_1 = InceptionV1Module(in_channels=480, out_channels1=
            192, out_channels2reduce=96, out_channels2=208,
            out_channels3reduce=16, out_channels3=48, out_channels4=64)
        if self.stage == 'train':
            self.aux_logits1 = InceptionAux(in_channels=512, out_channels=
                num_classes)
        self.block4_2 = nn.Sequential(InceptionV1Module(in_channels=512,
            out_channels1=160, out_channels2reduce=112, out_channels2=224,
            out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV1Module(in_channels=512, out_channels1=128,
            out_channels2reduce=128, out_channels2=256, out_channels3reduce
            =24, out_channels3=64, out_channels4=64), InceptionV1Module(
            in_channels=512, out_channels1=112, out_channels2reduce=144,
            out_channels2=288, out_channels3reduce=32, out_channels3=64,
            out_channels4=64))
        if self.stage == 'train':
            self.aux_logits2 = InceptionAux(in_channels=528, out_channels=
                num_classes)
        self.block4_3 = nn.Sequential(InceptionV1Module(in_channels=528,
            out_channels1=256, out_channels2reduce=160, out_channels2=320,
            out_channels3reduce=32, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block5 = nn.Sequential(InceptionV1Module(in_channels=832,
            out_channels1=256, out_channels2reduce=160, out_channels2=320,
            out_channels3reduce=32, out_channels3=128, out_channels4=128),
            InceptionV1Module(in_channels=832, out_channels1=384,
            out_channels2reduce=192, out_channels2=384, out_channels3reduce
            =48, out_channels3=128, out_channels4=128))
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux1 = x = self.block4_1(x)
        aux2 = x = self.block4_2(x)
        x = self.block4_3(x)
        out = self.block5(x)
        out = self.avgpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.stage == 'train':
            aux1 = self.aux_logits1(aux1)
            aux2 = self.aux_logits2(aux2)
            return aux1, aux2, out
        else:
            return out


class InceptionV2ModuleA(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels2reduce, out_channels=out_channels2,
            kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels3reduce, out_channels=out_channels3,
            kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3,
            out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
            padding=1), ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


def ConvBNReLUFactorization(in_channels, out_channels, kernel_sizes, paddings):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings),
        nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True), nn.Conv2d(
        in_channels=out_channels, out_channels=out_channels, kernel_size=
        kernel_sizes, stride=1, padding=paddings), nn.BatchNorm2d(
        out_channels), nn.ReLU6(inplace=True))


class InceptionV2ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
            out_channels=out_channels2reduce, kernel_sizes=[1, 3], paddings
            =[0, 1]), ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_sizes=[
            3, 1], paddings=[1, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
            out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings
            =[1, 0]), ConvBNReLUFactorization(in_channels=
            out_channels3reduce, out_channels=out_channels3reduce,
            kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(
            in_channels=out_channels3reduce, out_channels=
            out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
            out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
            padding=1), ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_sizes=[
            1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_sizes=[
            3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce,
            out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=
            out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1],
            paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=
            out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3],
            paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
            padding=1), ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)],
            dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)],
            dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1,
        out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels1reduce, out_channels=out_channels1,
            kernel_size=3, stride=2, padding=1))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels2reduce, out_channels=out_channels2,
            kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=
            out_channels2, out_channels=out_channels2, kernel_size=3,
            stride=2, padding=1))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionAux(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()
        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(in_channels=128, out_channels=768,
            kernel_size=5, stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=
            out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out


class InceptionV2(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV2, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=
            64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels
            =192, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV2ModuleA(in_channels=192,
            out_channels1=64, out_channels2reduce=64, out_channels2=64,
            out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV2ModuleA(in_channels=256, out_channels1=64,
            out_channels2reduce=64, out_channels2=96, out_channels3reduce=
            64, out_channels3=96, out_channels4=64), InceptionV3ModuleD(
            in_channels=320, out_channels1reduce=128, out_channels1=160,
            out_channels2reduce=64, out_channels2=96))
        self.block4 = nn.Sequential(InceptionV2ModuleB(in_channels=576,
            out_channels1=224, out_channels2reduce=64, out_channels2=96,
            out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=192,
            out_channels2reduce=96, out_channels2=128, out_channels3reduce=
            96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(
            in_channels=576, out_channels1=160, out_channels2reduce=128,
            out_channels2=160, out_channels3reduce=128, out_channels3=128,
            out_channels4=128), InceptionV2ModuleB(in_channels=576,
            out_channels1=96, out_channels2reduce=128, out_channels2=192,
            out_channels3reduce=160, out_channels3=160, out_channels4=128),
            InceptionV3ModuleD(in_channels=576, out_channels1reduce=128,
            out_channels1=192, out_channels2reduce=192, out_channels2=256))
        self.block5 = nn.Sequential(InceptionV2ModuleC(in_channels=1024,
            out_channels1=352, out_channels2reduce=192, out_channels2=160,
            out_channels3reduce=160, out_channels3=112, out_channels4=128),
            InceptionV2ModuleC(in_channels=1024, out_channels1=352,
            out_channels2reduce=192, out_channels2=160, out_channels3reduce
            =192, out_channels3=112, out_channels4=128))
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


class InceptionV3ModuleA(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels2reduce, out_channels=out_channels2,
            kernel_size=5, padding=2))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels3reduce, out_channels=out_channels3,
            kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3,
            out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
            padding=1), ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
            out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings
            =[0, 3]), ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_sizes=[
            7, 1], paddings=[3, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
            out_channels=out_channels3reduce, kernel_sizes=[7, 1], paddings
            =[3, 0]), ConvBNReLUFactorization(in_channels=
            out_channels3reduce, out_channels=out_channels3reduce,
            kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(
            in_channels=out_channels3reduce, out_channels=
            out_channels3reduce, kernel_sizes=[7, 1], paddings=[3, 0]),
            ConvBNReLUFactorization(in_channels=out_channels3reduce,
            out_channels=out_channels3, kernel_sizes=[1, 7], paddings=[0, 3]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
            padding=1), ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce,
        out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_sizes=[
            1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_sizes=[
            3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce,
            out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=
            out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1],
            paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=
            out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3],
            paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1,
            padding=1), ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)],
            dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)],
            dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1,
        out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels1reduce, out_channels=out_channels1,
            kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels2reduce, out_channels=out_channels2,
            kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=
            out_channels2, out_channels=out_channels2, kernel_size=3, stride=2)
            )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV3ModuleE(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1,
        out_channels2reduce, out_channels2):
        super(InceptionV3ModuleE, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(
            in_channels=out_channels1reduce, out_channels=out_channels1,
            kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce,
            out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings
            =[0, 3]), ConvBNReLUFactorization(in_channels=
            out_channels2reduce, out_channels=out_channels2reduce,
            kernel_sizes=[7, 1], paddings=[3, 0]), ConvBNReLU(in_channels=
            out_channels2reduce, out_channels=out_channels2, kernel_size=3,
            stride=2))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionAux(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()
        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels,
            out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(in_channels=128, out_channels=768,
            kernel_size=5, stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=
            out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out


class InceptionV3(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=
            32, kernel_size=3, stride=2), ConvBNReLU(in_channels=32,
            out_channels=32, kernel_size=3, stride=1), ConvBNReLU(
            in_channels=32, out_channels=64, kernel_size=3, stride=1,
            padding=1), nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels
            =80, kernel_size=3, stride=1), ConvBNReLU(in_channels=80,
            out_channels=192, kernel_size=3, stride=1, padding=1), nn.
            MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(InceptionV3ModuleA(in_channels=192,
            out_channels1=64, out_channels2reduce=48, out_channels2=64,
            out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV3ModuleA(in_channels=256, out_channels1=64,
            out_channels2reduce=48, out_channels2=64, out_channels3reduce=
            64, out_channels3=96, out_channels4=64), InceptionV3ModuleA(
            in_channels=288, out_channels1=64, out_channels2reduce=48,
            out_channels2=64, out_channels3reduce=64, out_channels3=96,
            out_channels4=64))
        self.block4 = nn.Sequential(InceptionV3ModuleD(in_channels=288,
            out_channels1reduce=384, out_channels1=384, out_channels2reduce
            =64, out_channels2=96), InceptionV3ModuleB(in_channels=768,
            out_channels1=192, out_channels2reduce=128, out_channels2=192,
            out_channels3reduce=128, out_channels3=192, out_channels4=192),
            InceptionV3ModuleB(in_channels=768, out_channels1=192,
            out_channels2reduce=160, out_channels2=192, out_channels3reduce
            =160, out_channels3=192, out_channels4=192), InceptionV3ModuleB
            (in_channels=768, out_channels1=192, out_channels2reduce=160,
            out_channels2=192, out_channels3reduce=160, out_channels3=192,
            out_channels4=192), InceptionV3ModuleB(in_channels=768,
            out_channels1=192, out_channels2reduce=192, out_channels2=192,
            out_channels3reduce=192, out_channels3=192, out_channels4=192))
        if self.stage == 'train':
            self.aux_logits = InceptionAux(in_channels=768, out_channels=
                num_classes)
        self.block5 = nn.Sequential(InceptionV3ModuleE(in_channels=768,
            out_channels1reduce=192, out_channels1=320, out_channels2reduce
            =192, out_channels2=192), InceptionV3ModuleC(in_channels=1280,
            out_channels1=320, out_channels2reduce=384, out_channels2=384,
            out_channels3reduce=448, out_channels3=384, out_channels4=192),
            InceptionV3ModuleC(in_channels=2048, out_channels1=320,
            out_channels2reduce=384, out_channels2=384, out_channels3reduce
            =448, out_channels3=384, out_channels4=192))
        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux = x = self.block4(x)
        x = self.block5(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        if self.stage == 'train':
            aux = self.aux_logits(aux)
            return aux, out
        else:
            return out


class InceptionV4(nn.Module):

    def __init__(self):
        super(InceptionV4, self).__init__()

    def forward(self):
        return out


class ResNeXtBlock(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False,
        expansion=2, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places,
            out_channels=places, kernel_size=1, stride=1, bias=False), nn.
            BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=places, out_channels=places, kernel_size=3, stride=
            stride, padding=1, bias=False, groups=cardinality), nn.
            BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=places, out_channels=places * self.expansion,
            kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places *
            self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places,
                out_channels=places * self.expansion, kernel_size=1, stride
                =stride, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False,
        expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places,
            out_channels=places, kernel_size=1, stride=1, bias=False), nn.
            BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=places, out_channels=places, kernel_size=3, stride=
            stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU
            (inplace=True), nn.Conv2d(in_channels=places, out_channels=
            places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places,
                out_channels=places * self.expansion, kernel_size=1, stride
                =stride, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks
            [0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=
            blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=
            blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=
            blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def Conv3x3BNReLU(in_channels, out_channels, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=stride, padding=1), nn.
        BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class VGGNet(nn.Module):

    def __init__(self, block_nums, num_classes=1000):
        super(VGGNet, self).__init__()
        self.stage1 = self._make_layers(in_channels=3, out_channels=64,
            block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128,
            block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256,
            block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512,
            block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512,
            block_num=block_nums[4])
        self.classifier = nn.Sequential(nn.Linear(in_features=512 * 7 * 7,
            out_features=4096), nn.ReLU6(inplace=True), nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU6(
            inplace=True), nn.Dropout(p=0.2), nn.Linear(in_features=4096,
            out_features=num_classes))

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels, out_channels))
        for i in range(1, block_num):
            layers.append(Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class Conv2dCReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding
        ):
        super(Conv2dCReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        out = torch.cat([x, -x], dim=1)
        return self.relu(out)


class InceptionModules(nn.Module):

    def __init__(self):
        super(InceptionModules, self).__init__()
        self.branch1_conv1 = nn.Conv2d(in_channels=128, out_channels=32,
            kernel_size=1, stride=1)
        self.branch1_conv1_bn = nn.BatchNorm2d(32)
        self.branch2_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_conv1 = nn.Conv2d(in_channels=128, out_channels=32,
            kernel_size=1, stride=1)
        self.branch2_conv1_bn = nn.BatchNorm2d(32)
        self.branch3_conv1 = nn.Conv2d(in_channels=128, out_channels=24,
            kernel_size=1, stride=1)
        self.branch3_conv1_bn = nn.BatchNorm2d(24)
        self.branch3_conv2 = nn.Conv2d(in_channels=24, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.branch3_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv1 = nn.Conv2d(in_channels=128, out_channels=24,
            kernel_size=1, stride=1)
        self.branch4_conv1_bn = nn.BatchNorm2d(24)
        self.branch4_conv2 = nn.Conv2d(in_channels=24, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.branch4_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv3 = nn.Conv2d(in_channels=32, out_channels=32,
            kernel_size=3, stride=1, padding=1)
        self.branch4_conv3_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x1 = self.branch1_conv1_bn(self.branch1_conv1(x))
        x2 = self.branch2_conv1_bn(self.branch2_conv1(self.branch2_pool(x)))
        x3 = self.branch3_conv2_bn(self.branch3_conv2(self.branch3_conv1_bn
            (self.branch3_conv1(x))))
        x4 = self.branch4_conv3_bn(self.branch4_conv3(self.branch4_conv2_bn
            (self.branch4_conv2(self.branch4_conv1_bn(self.branch4_conv1(x)))))
            )
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class FaceBoxes(nn.Module):

    def __init__(self, num_classes, phase):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.RapidlyDigestedConvolutionalLayers = nn.Sequential(Conv2dCReLU
            (in_channels=3, out_channels=24, kernel_size=7, stride=4,
            padding=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv2dCReLU(in_channels=48, out_channels=64, kernel_size=5,
            stride=2, padding=2), nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1))
        self.MultipleScaleConvolutionalLayers = nn.Sequential(InceptionModules
            (), InceptionModules(), InceptionModules())
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128,
            kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=128,
            kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.loc_layer1 = nn.Conv2d(in_channels=128, out_channels=21 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=128, out_channels=21 *
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer2 = nn.Conv2d(in_channels=256, out_channels=4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=256, out_channels=
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer3 = nn.Conv2d(in_channels=256, out_channels=4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=256, out_channels=
            num_classes, kernel_size=3, stride=1, padding=1)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        elif self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.RapidlyDigestedConvolutionalLayers(x)
        out1 = self.MultipleScaleConvolutionalLayers(x)
        out2 = self.conv3_2(self.conv3_1(out1))
        out3 = self.conv4_2(self.conv4_1(out2))
        loc1 = self.loc_layer1(out1)
        conf1 = self.conf_layer1(out1)
        loc2 = self.loc_layer2(out2)
        conf2 = self.conf_layer2(out2)
        loc3 = self.loc_layer3(out3)
        conf3 = self.conf_layer3(out3)
        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc2.
            size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc3.
            size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(
            conf1.size(0), -1), conf2.permute(0, 2, 3, 1).contiguous().view
            (conf2.size(0), -1), conf3.permute(0, 2, 3, 1).contiguous().
            view(conf3.size(0), -1)], dim=1)
        if self.phase == 'test':
            out = locs.view(locs.size(0), -1, 4), self.softmax(confs.view(-
                1, self.num_classes))
        else:
            out = locs.view(locs.size(0), -1, 4), confs.view(-1, self.
                num_classes)
        return out


def Conv1x1ReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=1, stride=1), nn.ReLU6(inplace=True))


class LossBranch(nn.Module):

    def __init__(self, in_channels, mid_channels=64):
        super(LossBranch, self).__init__()
        self.conv1 = Conv1x1ReLU(in_channels, mid_channels)
        self.conv2_score = Conv1x1ReLU(mid_channels, mid_channels)
        self.classify = nn.Conv2d(in_channels=mid_channels, out_channels=2,
            kernel_size=1, stride=1)
        self.conv2_bbox = Conv1x1ReLU(mid_channels, mid_channels)
        self.regress = nn.Conv2d(in_channels=mid_channels, out_channels=4,
            kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        cls = self.classify(self.conv2_score(x))
        reg = self.regress(self.conv2_bbox(x))
        return cls, reg


def Conv3x3ReLU(in_channels, out_channels, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=stride, padding=padding), nn.
        ReLU6(inplace=True))


class LFFDBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(LFFDBlock, self).__init__()
        mid_channels = out_channels
        self.downsampling = True if stride == 2 else False
        if self.downsampling:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
                mid_channels, kernel_size=3, stride=stride, padding=0)
        self.branch1_relu1 = nn.ReLU6(inplace=True)
        self.branch1_conv1 = Conv3x3ReLU(in_channels=mid_channels,
            out_channels=mid_channels, stride=1, padding=1)
        self.branch1_conv2 = nn.Conv2d(in_channels=mid_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.downsampling:
            x = self.conv(x)
        out = self.branch1_conv2(self.branch1_conv1(self.branch1_relu1(x)))
        return self.relu(out + x)


class LFFD(nn.Module):

    def __init__(self, classes_num=2):
        super(LFFD, self).__init__()
        self.tiny_part1 = nn.Sequential(Conv3x3ReLU(in_channels=3,
            out_channels=64, stride=2, padding=0), LFFDBlock(in_channels=64,
            out_channels=64, stride=2), LFFDBlock(in_channels=64,
            out_channels=64, stride=1), LFFDBlock(in_channels=64,
            out_channels=64, stride=1))
        self.tiny_part2 = LFFDBlock(in_channels=64, out_channels=64, stride=1)
        self.small_part1 = LFFDBlock(in_channels=64, out_channels=64, stride=2)
        self.small_part2 = LFFDBlock(in_channels=64, out_channels=64, stride=1)
        self.medium_part = nn.Sequential(LFFDBlock(in_channels=64,
            out_channels=128, stride=2), LFFDBlock(in_channels=128,
            out_channels=128, stride=1))
        self.large_part1 = LFFDBlock(in_channels=128, out_channels=128,
            stride=2)
        self.large_part2 = LFFDBlock(in_channels=128, out_channels=128,
            stride=1)
        self.large_part3 = LFFDBlock(in_channels=128, out_channels=128,
            stride=1)
        self.loss_branch1 = LossBranch(in_channels=64)
        self.loss_branch2 = LossBranch(in_channels=64)
        self.loss_branch3 = LossBranch(in_channels=64)
        self.loss_branch4 = LossBranch(in_channels=64)
        self.loss_branch5 = LossBranch(in_channels=128)
        self.loss_branch6 = LossBranch(in_channels=128)
        self.loss_branch7 = LossBranch(in_channels=128)
        self.loss_branch8 = LossBranch(in_channels=128)

    def forward(self, x):
        branch1 = self.tiny_part1(x)
        branch2 = self.tiny_part2(branch1)
        branch3 = self.small_part1(branch2)
        branch4 = self.small_part2(branch3)
        branch5 = self.medium_part(branch4)
        branch6 = self.large_part1(branch5)
        branch7 = self.large_part2(branch6)
        branch8 = self.large_part3(branch7)
        cls1, loc1 = self.loss_branch1(branch1)
        cls2, loc2 = self.loss_branch2(branch2)
        cls3, loc3 = self.loss_branch3(branch3)
        cls4, loc4 = self.loss_branch4(branch4)
        cls5, loc5 = self.loss_branch5(branch5)
        cls6, loc6 = self.loss_branch6(branch6)
        cls7, loc7 = self.loss_branch7(branch7)
        cls8, loc8 = self.loss_branch8(branch8)
        cls = torch.cat([cls1.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls2.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls3.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls4.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls5.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls6.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls7.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), cls8.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1)], dim=1)
        loc = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc8.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1)], dim=1)
        out = cls, loc
        return out


class SqueezeAndExcite(nn.Module):

    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(nn.Linear(in_features=in_channels,
            out_features=mid_channels), nn.ReLU6(inplace=True), nn.Linear(
            in_features=mid_channels, out_features=out_channels), nn.ReLU6(
            inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


def VarGConv(in_channels, out_channels, kernel_size, stride, S):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
        stride, padding=kernel_size // 2, groups=in_channels // S, bias=
        False), nn.BatchNorm2d(out_channels), nn.PReLU())


def VarGPointConv(in_channels, out_channels, stride, S, isRelu):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride,
        padding=0, groups=in_channels // S, bias=False), nn.BatchNorm2d(
        out_channels), nn.PReLU() if isRelu else nn.Sequential())


class VarGBlock_S1(nn.Module):

    def __init__(self, in_plances, kernel_size, stride=1, S=8):
        super(VarGBlock_S1, self).__init__()
        plances = 2 * in_plances
        self.varGConv1 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv1 = VarGPointConv(plances, in_plances, stride, S,
            isRelu=True)
        self.varGConv2 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv2 = VarGPointConv(plances, in_plances, stride, S,
            isRelu=False)
        self.se = SqueezeAndExcite(in_plances, in_plances)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = x
        x = self.varGPointConv1(self.varGConv1(x))
        x = self.varGPointConv2(self.varGConv2(x))
        x = self.se(x)
        out += x
        return self.prelu(out)


class VarGBlock_S2(nn.Module):

    def __init__(self, in_plances, kernel_size, stride=2, S=8):
        super(VarGBlock_S2, self).__init__()
        plances = 2 * in_plances
        self.varGConvBlock_branch1 = nn.Sequential(VarGConv(in_plances,
            plances, kernel_size, stride, S), VarGPointConv(plances,
            plances, 1, S, isRelu=True))
        self.varGConvBlock_branch2 = nn.Sequential(VarGConv(in_plances,
            plances, kernel_size, stride, S), VarGPointConv(plances,
            plances, 1, S, isRelu=True))
        self.varGConvBlock_3 = nn.Sequential(VarGConv(plances, plances * 2,
            kernel_size, 1, S), VarGPointConv(plances * 2, plances, 1, S,
            isRelu=False))
        self.shortcut = nn.Sequential(VarGConv(in_plances, plances,
            kernel_size, stride, S), VarGPointConv(plances, plances, 1, S,
            isRelu=False))
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.shortcut(x)
        x1 = x2 = x
        x1 = self.varGConvBlock_branch1(x1)
        x2 = self.varGConvBlock_branch2(x2)
        x_new = x1 + x2
        x_new = self.varGConvBlock_3(x_new)
        out += x_new
        return self.prelu(out)


class HeadBlock(nn.Module):

    def __init__(self, in_plances, kernel_size, S=8):
        super(HeadBlock, self).__init__()
        self.varGConvBlock = nn.Sequential(VarGConv(in_plances, in_plances,
            kernel_size, 2, S), VarGPointConv(in_plances, in_plances, 1, S,
            isRelu=True), VarGConv(in_plances, in_plances, kernel_size, 1,
            S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=False))
        self.shortcut = nn.Sequential(VarGConv(in_plances, in_plances,
            kernel_size, 2, S), VarGPointConv(in_plances, in_plances, 1, S,
            isRelu=False))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.varGConvBlock(x)
        out += x
        return out


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(
        out_channels), nn.ReLU6(inplace=True))


class TailEmbedding(nn.Module):

    def __init__(self, in_plances, plances=512, S=8):
        super(TailEmbedding, self).__init__()
        self.embedding = nn.Sequential(Conv1x1BNReLU(in_plances, 1024), nn.
            Conv2d(1024, 1024, 7, 1, padding=0, groups=1024 // S, bias=
            False), nn.Conv2d(1024, 512, 1, 1, padding=0, groups=512, bias=
            False))
        self.fc = nn.Linear(in_features=512, out_features=plances)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class VarGFaceNet(nn.Module):

    def __init__(self, num_classes=512):
        super(VarGFaceNet, self).__init__()
        S = 8
        self.conv1 = Conv3x3BNReLU(3, 40, 1)
        self.head = HeadBlock(40, 3)
        self.stage2 = nn.Sequential(VarGBlock_S2(40, 3, 2), VarGBlock_S1(80,
            3, 1), VarGBlock_S1(80, 3, 1))
        self.stage3 = nn.Sequential(VarGBlock_S2(80, 3, 2), VarGBlock_S1(
            160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1),
            VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(
            160, 3, 1))
        self.stage4 = nn.Sequential(VarGBlock_S2(160, 3, 2), VarGBlock_S1(
            320, 3, 1), VarGBlock_S1(320, 3, 1), VarGBlock_S1(320, 3, 1))
        self.tail = TailEmbedding(320, num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.head(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        out = self.tail(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.bottleneck = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=mid_channels, kernel_size=1, stride=1), ConvBNReLU
            (in_channels=mid_channels, out_channels=mid_channels,
            kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=
            mid_channels, out_channels=out_channels, kernel_size=1, stride=1))
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out + self.shortcut(x)


class HourglassModule(nn.Module):

    def __init__(self, nChannels=256, nModules=2, numReductions=4):
        super(HourglassModule, self).__init__()
        self.nChannels = nChannels
        self.nModules = nModules
        self.numReductions = numReductions
        self.residual_block = self._make_residual_layer(self.nModules, self
            .nChannels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.after_pool_block = self._make_residual_layer(self.nModules,
            self.nChannels)
        if numReductions > 1:
            self.hourglass_module = HourglassModule(self.nChannels, self.
                numReductions - 1, self.nModules)
        else:
            self.num1res_block = self._make_residual_layer(self.nModules,
                self.nChannels)
        self.lowres_block = self._make_residual_layer(self.nModules, self.
            nChannels)
        self.upsample = nn.Upsample(scale_factor=2)

    def _make_residual_layer(self, nModules, nChannels):
        _residual_blocks = []
        for _ in range(nModules):
            _residual_blocks.append(ResidualBlock(in_channels=nChannels,
                out_channels=nChannels))
        return nn.Sequential(*_residual_blocks)

    def forward(self, x):
        out1 = self.residual_block(x)
        out2 = self.max_pool(x)
        out2 = self.after_pool_block(out2)
        if self.numReductions > 1:
            out2 = self.hourglass_module(out2)
        else:
            out2 = self.num1res_block(out2)
        out2 = self.lowres_block(out2)
        out2 = self.upsample(out2)
        return out1 + out2


class Hourglass(nn.Module):

    def __init__(self, nJoints):
        super(Hourglass, self).__init__()
        self.first_conv = ConvBNReLU(in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3)
        self.residual_block1 = ResidualBlock(in_channels=64, out_channels=128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_block2 = ResidualBlock(in_channels=128, out_channels=128)
        self.residual_block3 = ResidualBlock(in_channels=128, out_channels=256)
        self.hourglass_module1 = HourglassModule(nChannels=256, nModules=2,
            numReductions=4)
        self.hourglass_module2 = HourglassModule(nChannels=256, nModules=2,
            numReductions=4)
        self.after_hourglass_conv1 = ConvBNReLU(in_channels=256,
            out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv1 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=1, stride=1)
        self.out_conv1 = nn.Conv2d(in_channels=256, out_channels=nJoints,
            kernel_size=1, stride=1)
        self.remap_conv1 = nn.Conv2d(in_channels=nJoints, out_channels=256,
            kernel_size=1, stride=1)
        self.after_hourglass_conv2 = ConvBNReLU(in_channels=256,
            out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv2 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv2d(in_channels=256, out_channels=nJoints,
            kernel_size=1, stride=1)
        self.remap_conv2 = nn.Conv2d(in_channels=nJoints, out_channels=256,
            kernel_size=1, stride=1)

    def forward(self, x):
        x = self.max_pool(self.residual_block1(self.first_conv(x)))
        x = self.residual_block3(self.residual_block2(x))
        x = self.hourglass_module1(x)
        residual1 = x = self.after_hourglass_conv1(x)
        out1 = self.out_conv1(x)
        residual2 = x = residual1 + self.remap_conv1(out1) + self.proj_conv1(x)
        x = self.hourglass_module2(x)
        x = self.after_hourglass_conv2(x)
        out2 = self.out_conv2(x)
        x = residual2 + self.remap_conv2(out2) + self.proj_conv2(x)
        return out1, out2


class LBwithGCBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LBwithGCBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes,
            kernel_size=1, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(planes)
        self.conv1_bn_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,
            kernel_size=3, stride=stride, padding=1)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv2_bn_relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes *
            self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(planes * self.expansion)
        self.gcb = ContextBlock(planes * self.expansion, ratio=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1_bn_relu(self.conv1_bn(self.conv1(x)))
        out = self.conv2_bn_relu(self.conv2_bn(self.conv2(out)))
        out = self.conv3_bn(self.conv3(out))
        out = self.gcb(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


def computeGCD(a, b):
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return b


def GroupDeconv(inplanes, planes, kernel_size, stride, padding, output_padding
    ):
    groups = computeGCD(inplanes, planes)
    return nn.Sequential(nn.ConvTranspose2d(in_channels=inplanes,
        out_channels=2 * planes, kernel_size=kernel_size, stride=stride,
        padding=padding, output_padding=output_padding, groups=groups), nn.
        Conv2d(2 * planes, planes, kernel_size=1, stride=1, padding=0))


class LPN(nn.Module):

    def __init__(self, nJoints):
        super(LPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(LBwithGCBlock, 64, 3)
        self.layer2 = self._make_layer(LBwithGCBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(LBwithGCBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(LBwithGCBlock, 512, 3, stride=1)
        self.deconv_layers = self._make_deconv_group_layer()
        self.final_layer = nn.Conv2d(in_channels=self.inplanes,
            out_channels=nJoints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride), nn.
                BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_group_layer(self):
        layers = []
        planes = 256
        for i in range(2):
            planes = planes // 2
            layers.append(GroupDeconv(inplanes=self.inplanes, planes=planes,
                kernel_size=4, stride=2, padding=1, output_padding=0))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class ResBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class SimpleBaseline(nn.Module):

    def __init__(self, nJoints):
        super(SimpleBaseline, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, 64, 3)
        self.layer2 = self._make_layer(ResBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 3, stride=2)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=nJoints,
            kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes,
                out_channels=256, kernel_size=4, stride=2, padding=1,
                output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = 256
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=(
        'channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


def conf_centernessLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=1, padding=1))


def locLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=1, padding=1))


class PolarMask(nn.Module):

    def __init__(self, num_classes=21):
        super(PolarMask, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256,
            kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256,
            kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer3 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer4 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer5 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer6 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer7 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6(p6)
        loc3 = self.loc_layer3(p3)
        conf_centerness3 = self.conf_centerness_layer3(p3)
        conf3, centerness3 = conf_centerness3.split([self.num_classes, 1],
            dim=1)
        loc4 = self.loc_layer4(p4)
        conf_centerness4 = self.conf_centerness_layer4(p4)
        conf4, centerness4 = conf_centerness4.split([self.num_classes, 1],
            dim=1)
        loc5 = self.loc_layer5(p5)
        conf_centerness5 = self.conf_centerness_layer5(p5)
        conf5, centerness5 = conf_centerness5.split([self.num_classes, 1],
            dim=1)
        loc6 = self.loc_layer6(p6)
        conf_centerness6 = self.conf_centerness_layer6(p6)
        conf6, centerness6 = conf_centerness6.split([self.num_classes, 1],
            dim=1)
        loc7 = self.loc_layer7(p7)
        conf_centerness7 = self.conf_centerness_layer7(p7)
        conf7, centerness7 = conf_centerness7.split([self.num_classes, 1],
            dim=1)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.
            size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.
            size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.
            size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.
            size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.
            size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(
            conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view
            (conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().
            view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous()
            .view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous(
            ).view(conf7.size(0), -1)], dim=1)
        centernesses = torch.cat([centerness3.permute(0, 2, 3, 1).
            contiguous().view(centerness3.size(0), -1), centerness4.permute
            (0, 2, 3, 1).contiguous().view(centerness4.size(0), -1),
            centerness5.permute(0, 2, 3, 1).contiguous().view(centerness5.
            size(0), -1), centerness6.permute(0, 2, 3, 1).contiguous().view
            (centerness6.size(0), -1), centerness7.permute(0, 2, 3, 1).
            contiguous().view(centerness7.size(0), -1)], dim=1)
        out = locs, confs, centernesses
        return out


class SqueezeAndExcite(nn.Module):

    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(nn.Linear(in_features=in_channels,
            out_features=mid_channels), nn.ReLU6(inplace=True), nn.Linear(
            in_features=mid_channels, out_features=out_channels), nn.ReLU6(
            inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


def DW_Conv3x3BNReLU(in_channels, out_channels, stride, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=stride, padding=1, groups=
        groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace
        =True))


class GhostModule(nn.Module):

    def __init__(self, in_channels, out_channels, s=2, kernel_size=1,
        stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channels = out_channels // s
        ghost_channels = intrinsic_channels * (s - 1)
        self.primary_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
            out_channels=intrinsic_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False), nn.
            BatchNorm2d(intrinsic_channels), nn.ReLU(inplace=True) if
            use_relu else nn.Sequential())
        self.cheap_op = DW_Conv3x3BNReLU(in_channels=intrinsic_channels,
            out_channels=ghost_channels, stride=stride, groups=
            intrinsic_channels)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_op(y)
        out = torch.cat([y, z], dim=1)
        return out


class GhostBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size,
        stride, use_se, se_kernel_size=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.bottleneck = nn.Sequential(GhostModule(in_channels=in_channels,
            out_channels=mid_channels, kernel_size=1, use_relu=True), 
            DW_Conv3x3BNReLU(in_channels=mid_channels, out_channels=
            mid_channels, stride=stride, groups=mid_channels) if self.
            stride > 1 else nn.Sequential(), SqueezeAndExcite(mid_channels,
            mid_channels, se_kernel_size) if use_se else nn.Sequential(),
            GhostModule(in_channels=mid_channels, out_channels=out_channels,
            kernel_size=1, use_relu=False))
        if self.stride > 1:
            self.shortcut = DW_Conv3x3BNReLU(in_channels=in_channels,
                out_channels=out_channels, stride=stride)
        else:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels
                =out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out


class GhostNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(GhostNet, self).__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3,
            out_channels=16, kernel_size=3, stride=2, padding=1), nn.
            BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.features = nn.Sequential(GhostBottleneck(in_channels=16,
            mid_channels=16, out_channels=16, kernel_size=3, stride=1,
            use_se=False), GhostBottleneck(in_channels=16, mid_channels=64,
            out_channels=24, kernel_size=3, stride=2, use_se=False),
            GhostBottleneck(in_channels=24, mid_channels=72, out_channels=
            24, kernel_size=3, stride=1, use_se=False), GhostBottleneck(
            in_channels=24, mid_channels=72, out_channels=40, kernel_size=5,
            stride=2, use_se=True, se_kernel_size=28), GhostBottleneck(
            in_channels=40, mid_channels=120, out_channels=40, kernel_size=
            5, stride=1, use_se=True, se_kernel_size=28), GhostBottleneck(
            in_channels=40, mid_channels=120, out_channels=40, kernel_size=
            5, stride=1, use_se=True, se_kernel_size=28), GhostBottleneck(
            in_channels=40, mid_channels=240, out_channels=80, kernel_size=
            3, stride=1, use_se=False), GhostBottleneck(in_channels=80,
            mid_channels=200, out_channels=80, kernel_size=3, stride=1,
            use_se=False), GhostBottleneck(in_channels=80, mid_channels=184,
            out_channels=80, kernel_size=3, stride=2, use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=184, out_channels=
            80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(
            in_channels=80, mid_channels=480, out_channels=112, kernel_size
            =3, stride=1, use_se=True, se_kernel_size=14), GhostBottleneck(
            in_channels=112, mid_channels=672, out_channels=112,
            kernel_size=3, stride=1, use_se=True, se_kernel_size=14),
            GhostBottleneck(in_channels=112, mid_channels=672, out_channels
            =160, kernel_size=5, stride=2, use_se=True, se_kernel_size=7),
            GhostBottleneck(in_channels=160, mid_channels=960, out_channels
            =160, kernel_size=5, stride=1, use_se=True, se_kernel_size=7),
            GhostBottleneck(in_channels=160, mid_channels=960, out_channels
            =160, kernel_size=5, stride=1, use_se=True, se_kernel_size=7))
        self.last_stage = nn.Sequential(nn.Conv2d(in_channels=160,
            out_channels=960, kernel_size=1, stride=1), nn.BatchNorm2d(960),
            nn.ReLU6(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1,
            stride=1), nn.ReLU6(inplace=True))
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_stage(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class HardSwish(nn.Module):

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class MDConv(nn.Module):

    def __init__(self, nchannels, kernel_sizes, stride):
        super(MDConv, self).__init__()
        self.nchannels = nchannels
        self.groups = len(kernel_sizes)
        self.split_channels = [(nchannels // self.groups) for _ in range(
            self.groups)]
        self.split_channels[0] += nchannels - sum(self.split_channels)
        self.layers = []
        for i in range(self.groups):
            self.layers.append(nn.Conv2d(in_channels=self.split_channels[i],
                out_channels=self.split_channels[i], kernel_size=
                kernel_sizes[i], stride=stride, padding=int(kernel_sizes[i] //
                2), groups=self.split_channels[i]))

    def forward(self, x):
        split_x = torch.split(x, self.split_channels, dim=1)
        outputs = [layer(sp_x) for layer, sp_x in zip(self.layers, split_x)]
        return torch.cat(outputs, dim=1)


class SqueezeAndExcite(nn.Module):

    def __init__(self, nchannels, squeeze_channels, se_ratio=1):
        super(SqueezeAndExcite, self).__init__()
        squeeze_channels = int(squeeze_channels * se_ratio)
        self.SEblock = nn.Sequential(nn.Conv2d(in_channels=nchannels,
            out_channels=squeeze_channels, kernel_size=1, stride=1, padding
            =0), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=
            squeeze_channels, out_channels=nchannels, kernel_size=1, stride
            =1, padding=0), nn.Sigmoid())

    def forward(self, x):
        out = torch.mean(x, (2, 3), keepdim=True)
        out = self.SEblock(out)
        return out * x


def Conv1x1BN(in_channels, out_channels, groups):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=1, stride=1, groups=groups), nn.
        BatchNorm2d(out_channels))


def Conv1x1BNActivation(in_channels, out_channels, activate):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels
        ), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish())


class MDConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, stride,
        expand_ratio, activate='relu', se_ratio=1):
        super(MDConvBlock, self).__init__()
        self.stride = stride
        self.se_ratio = se_ratio
        mid_channels = in_channels * expand_ratio
        self.expand_conv = Conv1x1BNActivation(in_channels, mid_channels,
            activate)
        self.md_conv = nn.Sequential(MDConv(mid_channels, kernel_sizes,
            stride), nn.BatchNorm2d(mid_channels), nn.ReLU6(inplace=True) if
            activate == 'relu' else HardSwish(inplace=True))
        if self.se_ratio > 0:
            self.squeeze_excite = SqueezeAndExcite(mid_channels, in_channels)
        self.projection_conv = Conv1x1BN(mid_channels, out_channels)

    def forward(self, x):
        x = self.expand_conv(x)
        x = self.md_conv(x)
        if self.se_ratio > 0:
            x = self.squeeze_excite(x)
        out = self.projection_conv(x)
        return out


class MixNet(nn.Module):
    mixnet_s = [(16, 16, [3], 1, 1, 'ReLU', 0.0), (16, 24, [3], 2, 6,
        'ReLU', 0.0), (24, 24, [3], 1, 3, 'ReLU', 0.0), (24, 40, [3, 5, 7],
        2, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40,
        [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
        (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25), (80, 80, [3, 5], 1, 6,
        'Swish', 0.25), (80, 80, [3, 5], 1, 6, 'Swish', 0.25), (80, 120, [3,
        5, 7], 1, 6, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish',
        0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 200, [3, 
        5, 7, 9, 11], 2, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6,
        'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]
    mixnet_m = [(24, 24, [3], 1, 1, 'ReLU', 0.0), (24, 32, [3, 5, 7], 2, 6,
        'ReLU', 0.0), (32, 32, [3], 1, 3, 'ReLU', 0.0), (32, 40, [3, 5, 7, 
        9], 2, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40,
        40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 
        0.5), (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 
        9], 1, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 
        0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 120, [3], 
        1, 6, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
        (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 
        9], 1, 3, 'Swish', 0.5), (120, 200, [3, 5, 7, 9], 2, 6, 'Swish', 
        0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 
        5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6,
        'Swish', 0.5)]

    def __init__(self, type='mixnet_s'):
        super(MixNet, self).__init__()
        if type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
        elif type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
        self.stem = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            stem_channels, kernel_size=3, stride=2, padding=1), nn.
            BatchNorm2d(stem_channels), HardSwish(inplace=True))
        layers = []
        for in_channels, out_channels, kernel_sizes, stride, expand_ratio, activate, se_ratio in config:
            layers.append(MDConvBlock(in_channels, out_channels,
                kernel_sizes=kernel_sizes, stride=stride, expand_ratio=
                expand_ratio, activate=activate, se_ratio=se_ratio))
        self.bottleneck = nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        out = self.bottleneck(x)
        return out


def BottleneckV1(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        in_channels, kernel_size=3, stride=stride, padding=1, groups=
        in_channels), nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), nn.ReLU6(
        inplace=True))


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3,
            out_channels=32, kernel_size=3, stride=2, padding=1), nn.
            BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.bottleneck = nn.Sequential(BottleneckV1(32, 64, stride=1),
            BottleneckV1(64, 128, stride=2), BottleneckV1(128, 128, stride=
            1), BottleneckV1(128, 256, stride=2), BottleneckV1(256, 256,
            stride=1), BottleneckV1(256, 512, stride=2), BottleneckV1(512, 
            512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(
            512, 512, stride=1), BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1), BottleneckV1(512, 1024,
            stride=2), BottleneckV1(1024, 1024, stride=1))
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        out = self.softmax(x)
        return out


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            mid_channels), Conv3x3BNReLU(mid_channels, mid_channels, stride,
            groups=mid_channels), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)
        self.layer1 = self.make_layer(in_channels=32, out_channels=16,
            stride=1, block_num=1)
        self.layer2 = self.make_layer(in_channels=16, out_channels=24,
            stride=2, block_num=2)
        self.layer3 = self.make_layer(in_channels=24, out_channels=32,
            stride=2, block_num=3)
        self.layer4 = self.make_layer(in_channels=32, out_channels=64,
            stride=2, block_num=4)
        self.layer5 = self.make_layer(in_channels=64, out_channels=96,
            stride=1, block_num=3)
        self.layer6 = self.make_layer(in_channels=96, out_channels=160,
            stride=2, block_num=3)
        self.layer7 = self.make_layer(in_channels=160, out_channels=320,
            stride=1, block_num=1)
        self.last_conv = Conv1x1BNReLU(320, 1280)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)

    def make_layer(self, in_channels, out_channels, stride, block_num):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class HardSwish(nn.Module):

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class SqueezeAndExcite(nn.Module):

    def __init__(self, in_channels, out_channels, se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size, stride=1)
        self.SEblock = nn.Sequential(nn.Linear(in_features=in_channels,
            out_features=mid_channels), nn.ReLU6(inplace=True), nn.Linear(
            in_features=mid_channels, out_features=out_channels), HardSwish
            (inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


def ConvBNActivation(in_channels, out_channels, kernel_size, stride, activate):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=kernel_size, stride=stride, padding=(
        kernel_size - 1) // 2, groups=in_channels), nn.BatchNorm2d(
        out_channels), nn.ReLU6(inplace=True) if activate == 'relu' else
        HardSwish())


class SEInvertedBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size,
        stride, activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels,
            kernel_size, stride, activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels,
                se_kernel_size)
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels,
            activate)
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):

    def __init__(self, num_classes=1000, type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3,
            out_channels=16, kernel_size=3, stride=2, padding=1), nn.
            BatchNorm2d(16), HardSwish(inplace=True))
        if type == 'large':
            self.large_bottleneck = nn.Sequential(SEInvertedBottleneck(
                in_channels=16, mid_channels=16, out_channels=16,
                kernel_size=3, stride=1, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=16, mid_channels=64,
                out_channels=24, kernel_size=3, stride=2, activate='relu',
                use_se=False), SEInvertedBottleneck(in_channels=24,
                mid_channels=72, out_channels=24, kernel_size=3, stride=1,
                activate='relu', use_se=False), SEInvertedBottleneck(
                in_channels=24, mid_channels=72, out_channels=40,
                kernel_size=5, stride=2, activate='relu', use_se=True,
                se_kernel_size=28), SEInvertedBottleneck(in_channels=40,
                mid_channels=120, out_channels=40, kernel_size=5, stride=1,
                activate='relu', use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40, mid_channels=120,
                out_channels=40, kernel_size=5, stride=1, activate='relu',
                use_se=True, se_kernel_size=28), SEInvertedBottleneck(
                in_channels=40, mid_channels=240, out_channels=80,
                kernel_size=3, stride=1, activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=200,
                out_channels=80, kernel_size=3, stride=1, activate='hswish',
                use_se=False), SEInvertedBottleneck(in_channels=80,
                mid_channels=184, out_channels=80, kernel_size=3, stride=2,
                activate='hswish', use_se=False), SEInvertedBottleneck(
                in_channels=80, mid_channels=184, out_channels=80,
                kernel_size=3, stride=1, activate='hswish', use_se=False),
                SEInvertedBottleneck(in_channels=80, mid_channels=480,
                out_channels=112, kernel_size=3, stride=1, activate=
                'hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672,
                out_channels=112, kernel_size=3, stride=1, activate=
                'hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672,
                out_channels=160, kernel_size=5, stride=2, activate=
                'hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960,
                out_channels=160, kernel_size=5, stride=1, activate=
                'hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960,
                out_channels=160, kernel_size=5, stride=1, activate=
                'hswish', use_se=True, se_kernel_size=7))
            self.large_last_stage = nn.Sequential(nn.Conv2d(in_channels=160,
                out_channels=960, kernel_size=1, stride=1), nn.BatchNorm2d(
                960), HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7,
                stride=1), nn.Conv2d(in_channels=960, out_channels=1280,
                kernel_size=1, stride=1), HardSwish(inplace=True))
        else:
            self.small_bottleneck = nn.Sequential(SEInvertedBottleneck(
                in_channels=16, mid_channels=16, out_channels=16,
                kernel_size=3, stride=2, activate='relu', use_se=True,
                se_kernel_size=56), SEInvertedBottleneck(in_channels=16,
                mid_channels=72, out_channels=24, kernel_size=3, stride=2,
                activate='relu', use_se=False), SEInvertedBottleneck(
                in_channels=24, mid_channels=88, out_channels=24,
                kernel_size=3, stride=1, activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=96,
                out_channels=40, kernel_size=5, stride=2, activate='hswish',
                use_se=True, se_kernel_size=14), SEInvertedBottleneck(
                in_channels=40, mid_channels=240, out_channels=40,
                kernel_size=5, stride=1, activate='hswish', use_se=True,
                se_kernel_size=14), SEInvertedBottleneck(in_channels=40,
                mid_channels=240, out_channels=40, kernel_size=5, stride=1,
                activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=120,
                out_channels=48, kernel_size=5, stride=1, activate='hswish',
                use_se=True, se_kernel_size=14), SEInvertedBottleneck(
                in_channels=48, mid_channels=144, out_channels=48,
                kernel_size=5, stride=1, activate='hswish', use_se=True,
                se_kernel_size=14), SEInvertedBottleneck(in_channels=48,
                mid_channels=288, out_channels=96, kernel_size=5, stride=2,
                activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576,
                out_channels=96, kernel_size=5, stride=1, activate='hswish',
                use_se=True, se_kernel_size=7), SEInvertedBottleneck(
                in_channels=96, mid_channels=576, out_channels=96,
                kernel_size=5, stride=1, activate='hswish', use_se=True,
                se_kernel_size=7))
            self.small_last_stage = nn.Sequential(nn.Conv2d(in_channels=96,
                out_channels=576, kernel_size=1, stride=1), nn.BatchNorm2d(
                576), HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7,
                stride=1), nn.Conv2d(in_channels=576, out_channels=1280,
                kernel_size=1, stride=1), HardSwish(inplace=True))
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        if self.type == 'large':
            x = self.large_bottleneck(x)
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


class ShuffleNetUnits(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = (out_channels - in_channels if self.stride > 1 else
            out_channels)
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            mid_channels, groups), ChannelShuffle(groups), Conv3x3BNReLU(
            mid_channels, mid_channels, stride, groups), Conv1x1BN(
            mid_channels, out_channels, groups))
        if self.stride > 1:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1
            ) if self.stride > 1 else out + x
        return self.relu(out)


class ShuffleNet(nn.Module):

    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3,
            out_channels=24, stride=2, groups=1), nn.MaxPool2d(kernel_size=
            3, stride=2, padding=1))
        self.stage2 = self._make_layer(24, planes[0], groups, layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], groups, layers
            [1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], groups, layers
            [2], False)
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2] * 7 * 7, out_features
            =num_classes)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num,
        is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels
            =out_channels, stride=2, groups=1 if is_stage2 else groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels,
                out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class HalfSplit(nn.Module):

    def __init__(self, dim=0, first_half=True):
        super(HalfSplit, self).__init__()
        self.first_half = first_half
        self.dim = dim

    def forward(self, input):
        splits = torch.chunk(input, 2, dim=self.dim)
        return splits[0] if self.first_half else splits[1]


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


def Conv3x3BN(in_channels, out_channels, stride, groups):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=stride, padding=1, groups=
        groups), nn.BatchNorm2d(out_channels))


class ShuffleNetUnits(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        if self.stride > 1:
            mid_channels = out_channels - in_channels
        else:
            mid_channels = out_channels // 2
            in_channels = mid_channels
            self.first_half = HalfSplit(dim=1, first_half=True)
            self.second_split = HalfSplit(dim=1, first_half=False)
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            in_channels), Conv3x3BN(in_channels, mid_channels, stride,
            groups), Conv1x1BNReLU(mid_channels, mid_channels))
        if self.stride > 1:
            self.shortcut = nn.Sequential(Conv3x3BN(in_channels=in_channels,
                out_channels=in_channels, stride=stride, groups=groups),
                Conv1x1BNReLU(in_channels, in_channels))
        self.channel_shuffle = ChannelShuffle(groups)

    def forward(self, x):
        if self.stride > 1:
            x1 = self.bottleneck(x)
            x2 = self.shortcut(x)
        else:
            x1 = self.first_half(x)
            x2 = self.second_split(x)
            x1 = self.bottleneck(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out)
        return out


class ShuffleNetV2(nn.Module):

    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.groups = groups
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3,
            out_channels=24, stride=2, groups=1), nn.MaxPool2d(kernel_size=
            3, stride=2, padding=1))
        self.stage2 = self._make_layer(24, planes[0], layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], layers[1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], False)
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2] * 7 * 7, out_features
            =num_classes)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels
            =out_channels, stride=2, groups=1 if is_stage2 else self.groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels,
                out_channels=out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class FireModule(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(FireModule, self).__init__()
        mid_channels = out_channels // 4
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=
            mid_channels, kernel_size=1, stride=1)
        self.squeeze_relu = nn.ReLU6(inplace=True)
        self.expand3x3 = nn.Conv2d(in_channels=mid_channels, out_channels=
            out_channels, kernel_size=3, stride=1, padding=1)
        self.expand3x3_relu = nn.ReLU6(inplace=True)
        self.expand1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=
            out_channels, kernel_size=1, stride=1)
        self.expand1x1_relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.squeeze_relu(self.squeeze(x))
        y = self.expand3x3_relu(self.expand3x3(x))
        z = self.expand1x1_relu(self.expand1x1(x))
        out = torch.cat([y, z], dim=1)
        return out


class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=3,
            out_channels=96, kernel_size=7, stride=2, padding=3), nn.
            BatchNorm2d(96), nn.ReLU6(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2), FireModule(in_channels=96,
            out_channels=64), FireModule(in_channels=128, out_channels=64),
            FireModule(in_channels=128, out_channels=128), nn.MaxPool2d(
            kernel_size=3, stride=2), FireModule(in_channels=256,
            out_channels=128), FireModule(in_channels=256, out_channels=192
            ), FireModule(in_channels=384, out_channels=192), FireModule(
            in_channels=384, out_channels=256), nn.MaxPool2d(kernel_size=3,
            stride=2), FireModule(in_channels=512, out_channels=256), nn.
            Dropout(p=0.5), nn.Conv2d(in_channels=512, out_channels=
            num_classes, kernel_size=1, stride=1), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=13, stride=1))

    def forward(self, x):
        out = self.bottleneck(x)
        return out.view(out.size(1), -1)


def ConvBN(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=kernel_size, stride=stride, padding=
        padding), nn.BatchNorm2d(out_channels))


def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels
        ), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
        padding=0))


def ReluSeparableConvolution(in_channels, out_channels):
    return nn.Sequential(nn.ReLU6(inplace=True), SeparableConvolution(
        in_channels, out_channels))


class EntryBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, first_relu=True):
        super(EntryBottleneck, self).__init__()
        mid_channels = out_channels
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=2)
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(
            in_channels=in_channels, out_channels=mid_channels) if
            first_relu else SeparableConvolution(in_channels=in_channels,
            out_channels=mid_channels), ReluSeparableConvolution(
            in_channels=mid_channels, out_channels=out_channels), nn.
            MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


class MiddleBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MiddleBottleneck, self).__init__()
        mid_channels = out_channels
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(
            in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels
            =mid_channels), ReluSeparableConvolution(in_channels=
            mid_channels, out_channels=out_channels))

    def forward(self, x):
        out = self.bottleneck(x)
        return out + x


class ExitBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ExitBottleneck, self).__init__()
        mid_channels = in_channels
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=2)
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(
            in_channels=in_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels
            =out_channels), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


def ConvBNRelu(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(ConvBN(in_channels, out_channels, kernel_size,
        stride), nn.ReLU6(inplace=True))


def SeparableConvolutionRelu(in_channels, out_channels):
    return nn.Sequential(SeparableConvolution(in_channels, out_channels),
        nn.ReLU6(inplace=True))


class Xception(nn.Module):

    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.entryFlow = nn.Sequential(ConvBNRelu(in_channels=3,
            out_channels=32, kernel_size=3, stride=2), ConvBNRelu(
            in_channels=32, out_channels=64, kernel_size=3, stride=1),
            EntryBottleneck(in_channels=64, out_channels=128, first_relu=
            False), EntryBottleneck(in_channels=128, out_channels=256,
            first_relu=True), EntryBottleneck(in_channels=256, out_channels
            =728, first_relu=True))
        self.middleFlow = nn.Sequential(MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728), MiddleBottleneck(in_channels=728,
            out_channels=728))
        self.exitFlow = nn.Sequential(ExitBottleneck(in_channels=728,
            out_channels=1024), SeparableConvolutionRelu(in_channels=1024,
            out_channels=1536), SeparableConvolutionRelu(in_channels=1536,
            out_channels=2048), nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entryFlow(x)
        x = self.middleFlow(x)
        x = self.exitFlow(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


def Conv1x1BnRelu(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.
        BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def downSampling1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=2, padding=1, bias=False), nn.
        BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def downSampling2(in_channels, out_channels):
    return nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        downSampling1(in_channels=in_channels, out_channels=out_channels))


def upSampling1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.
        BatchNorm2d(out_channels), nn.ReLU6(inplace=True), nn.Upsample(
        scale_factor=2, mode='nearest'))


def upSampling2(in_channels, out_channels):
    return nn.Sequential(upSampling1(in_channels, out_channels), nn.
        Upsample(scale_factor=2, mode='nearest'))


class ASFF(nn.Module):

    def __init__(self, level, channel1, channel2, channel3, out_channel):
        super(ASFF, self).__init__()
        self.level = level
        funsed_channel = 8
        if self.level == 1:
            self.level2_1 = downSampling1(channel2, channel1)
            self.level3_1 = downSampling2(channel3, channel1)
            self.weight1 = Conv1x1BnRelu(channel1, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel1, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel1, funsed_channel)
            self.expand_conv = Conv1x1BnRelu(channel1, out_channel)
        if self.level == 2:
            self.level1_2 = upSampling1(channel1, channel2)
            self.level3_2 = downSampling1(channel3, channel2)
            self.weight1 = Conv1x1BnRelu(channel2, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel2, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel2, funsed_channel)
            self.expand_conv = Conv1x1BnRelu(channel2, out_channel)
        if self.level == 3:
            self.level1_3 = upSampling2(channel1, channel3)
            self.level2_3 = upSampling1(channel2, channel3)
            self.weight1 = Conv1x1BnRelu(channel3, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel3, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel3, funsed_channel)
            self.expand_conv = Conv1x1BnRelu(channel3, out_channel)
        self.weight_level = nn.Conv2d(funsed_channel * 3, 3, kernel_size=1,
            stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, z):
        if self.level == 1:
            level_x = x
            level_y = self.level2_1(y)
            level_z = self.level3_1(z)
        if self.level == 2:
            level_x = self.level1_2(x)
            level_y = y
            level_z = self.level3_2(z)
        if self.level == 3:
            level_x = self.level1_3(x)
            level_y = self.level2_3(y)
            level_z = z
        weight1 = self.weight1(level_x)
        weight2 = self.weight2(level_y)
        weight3 = self.weight3(level_z)
        level_weight = torch.cat((weight1, weight2, weight3), 1)
        weight_level = self.weight_level(level_weight)
        weight_level = self.softmax(weight_level)
        fused_level = level_x * weight_level[:, (0), :, :
            ] + level_y * weight_level[:, (1), :, :] + level_z * weight_level[:
            , (2), :, :]
        out = self.expand_conv(fused_level)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.bottleneck = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=mid_channels, kernel_size=1, stride=1), ConvBNReLU
            (in_channels=mid_channels, out_channels=mid_channels,
            kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=
            mid_channels, out_channels=out_channels, kernel_size=1, stride=1))
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out + self.shortcut(x)


class HourglassNetwork(nn.Module):

    def __init__(self):
        super(HourglassNetwork, self).__init__()

    def forward(self, x):
        return out


class PredictionModule(nn.Module):

    def __init__(self):
        super(PredictionModule, self).__init__()

    def forward(self, x):
        return out


class CornerNet(nn.Module):

    def __init__(self):
        super(CornerNet, self).__init__()

    def forward(self, x):
        return out


class FCOS(nn.Module):

    def __init__(self, num_classes=21):
        super(FCOS, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256,
            kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256,
            kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer3 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes + 1)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer4 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes + 1)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer5 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes + 1)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer6 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes + 1)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer7 = conf_centernessLayer(in_channels=256,
            out_channels=self.num_classes + 1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6(p6)
        loc3 = self.loc_layer3(p3)
        conf_centerness3 = self.conf_centerness_layer3(p3)
        conf3, centerness3 = conf_centerness3.split([self.num_classes, 1],
            dim=1)
        loc4 = self.loc_layer4(p4)
        conf_centerness4 = self.conf_centerness_layer4(p4)
        conf4, centerness4 = conf_centerness4.split([self.num_classes, 1],
            dim=1)
        loc5 = self.loc_layer5(p5)
        conf_centerness5 = self.conf_centerness_layer5(p5)
        conf5, centerness5 = conf_centerness5.split([self.num_classes, 1],
            dim=1)
        loc6 = self.loc_layer6(p6)
        conf_centerness6 = self.conf_centerness_layer6(p6)
        conf6, centerness6 = conf_centerness6.split([self.num_classes, 1],
            dim=1)
        loc7 = self.loc_layer7(p7)
        conf_centerness7 = self.conf_centerness_layer7(p7)
        conf7, centerness7 = conf_centerness7.split([self.num_classes, 1],
            dim=1)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.
            size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.
            size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.
            size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.
            size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.
            size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(
            conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view
            (conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().
            view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous()
            .view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous(
            ).view(conf7.size(0), -1)], dim=1)
        centernesses = torch.cat([centerness3.permute(0, 2, 3, 1).
            contiguous().view(centerness3.size(0), -1), centerness4.permute
            (0, 2, 3, 1).contiguous().view(centerness4.size(0), -1),
            centerness5.permute(0, 2, 3, 1).contiguous().view(centerness5.
            size(0), -1), centerness6.permute(0, 2, 3, 1).contiguous().view
            (centerness6.size(0), -1), centerness7.permute(0, 2, 3, 1).
            contiguous().view(centerness7.size(0), -1)], dim=1)
        out = locs, confs, centernesses
        return out


class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256,
            kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256,
            kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c2 = x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p2 = self.upsample2(p3) + self.lateral2(c2)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth4(p2)
        return p2, p3, p4, p5


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


class ShuffleNetUnits(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = (out_channels - in_channels if self.stride > 1 else
            out_channels)
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            mid_channels, groups), ChannelShuffle(groups), Conv3x3BNReLU(
            mid_channels, mid_channels, stride, groups), Conv1x1BN(
            mid_channels, out_channels, groups))
        if self.stride > 1:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1
            ) if self.stride > 1 else out + x
        return self.relu(out)


class FisheyeMODNet(nn.Module):

    def __init__(self, groups=1, num_classes=2):
        super(FisheyeMODNet, self).__init__()
        layers = [4, 8, 4]
        self.stage1a = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(
            kernel_size=2, stride=2))
        self.stage2a = self._make_layer(24, 120, groups, layers[0])
        self.stage1b = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(
            kernel_size=2, stride=2))
        self.stage2b = self._make_layer(24, 120, groups, layers[0])
        self.stage3 = self._make_layer(240, 480, groups, layers[1])
        self.stage4 = self._make_layer(480, 960, groups, layers[2])
        self.adapt_conv3 = nn.Conv2d(960, num_classes, kernel_size=1)
        self.adapt_conv2 = nn.Conv2d(480, num_classes, kernel_size=1)
        self.adapt_conv1 = nn.Conv2d(240, num_classes, kernel_size=1)
        self.up_sampling3 = nn.ConvTranspose2d(in_channels=num_classes,
            out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling2 = nn.ConvTranspose2d(in_channels=num_classes,
            out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling1 = nn.ConvTranspose2d(in_channels=num_classes,
            out_channels=num_classes, kernel_size=16, stride=8, padding=4)
        self.softmax = nn.Softmax(dim=1)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels
            =out_channels, stride=2, groups=groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels,
                out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = self.stage2a(self.stage1a(x))
        y = self.stage2b(self.stage1b(y))
        feature1 = torch.cat([x, y], dim=1)
        feature2 = self.stage3(feature1)
        feature3 = self.stage4(feature2)
        out3 = self.up_sampling3(self.adapt_conv3(feature3))
        out2 = self.up_sampling2(self.adapt_conv2(feature2) + out3)
        out1 = self.up_sampling1(self.adapt_conv1(feature1) + out2)
        out = self.softmax(out1)
        return out


def confLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=
        in_channels), nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=1, padding=1))


class FoveaBox(nn.Module):

    def __init__(self, num_classes=80):
        super(FoveaBox, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256,
            kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256,
            kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.downsample6_relu = nn.ReLU6(inplace=True)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer3 = confLayer(in_channels=256, out_channels=self.
            num_classes)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer4 = confLayer(in_channels=256, out_channels=self.
            num_classes)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer5 = confLayer(in_channels=256, out_channels=self.
            num_classes)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer6 = confLayer(in_channels=256, out_channels=self.
            num_classes)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer7 = confLayer(in_channels=256, out_channels=self.
            num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6_relu(self.downsample6(p6))
        loc3 = self.loc_layer3(p3)
        conf3 = self.conf_layer3(p3)
        loc4 = self.loc_layer4(p4)
        conf4 = self.conf_layer4(p4)
        loc5 = self.loc_layer5(p5)
        conf5 = self.conf_layer5(p5)
        loc6 = self.loc_layer6(p6)
        conf6 = self.conf_layer6(p6)
        loc7 = self.loc_layer7(p7)
        conf7 = self.conf_layer7(p7)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.
            size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.
            size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.
            size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.
            size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.
            size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(
            conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view
            (conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().
            view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous()
            .view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous(
            ).view(conf7.size(0), -1)], dim=1)
        out = locs, confs
        return out


class RetinaNet(nn.Module):

    def __init__(self, num_classes=80, num_anchores=9):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        resnet = torchvision.models.resnet50()
        layers = list(resnet.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.layer2 = nn.Sequential(*layers[5])
        self.layer3 = nn.Sequential(*layers[6])
        self.layer4 = nn.Sequential(*layers[7])
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256,
            kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256,
            kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256,
            kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=
            256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.downsample6_relu = nn.ReLU6(inplace=True)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4 *
            num_anchores)
        self.conf_layer3 = confLayer(in_channels=256, out_channels=self.
            num_classes * num_anchores)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4 *
            num_anchores)
        self.conf_layer4 = confLayer(in_channels=256, out_channels=self.
            num_classes * num_anchores)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4 *
            num_anchores)
        self.conf_layer5 = confLayer(in_channels=256, out_channels=self.
            num_classes * num_anchores)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4 *
            num_anchores)
        self.conf_layer6 = confLayer(in_channels=256, out_channels=self.
            num_classes * num_anchores)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4 *
            num_anchores)
        self.conf_layer7 = confLayer(in_channels=256, out_channels=self.
            num_classes * num_anchores)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        c3 = x = self.layer2(x)
        c4 = x = self.layer3(x)
        c5 = x = self.layer4(x)
        p5 = self.lateral5(c5)
        p4 = self.upsample4(p5) + self.lateral4(c4)
        p3 = self.upsample3(p4) + self.lateral3(c3)
        p6 = self.downsample5(p5)
        p7 = self.downsample6_relu(self.downsample6(p6))
        loc3 = self.loc_layer3(p3)
        conf3 = self.conf_layer3(p3)
        loc4 = self.loc_layer4(p4)
        conf4 = self.conf_layer4(p4)
        loc5 = self.loc_layer5(p5)
        conf5 = self.conf_layer5(p5)
        loc6 = self.loc_layer6(p6)
        conf6 = self.conf_layer6(p6)
        loc7 = self.loc_layer7(p7)
        conf7 = self.conf_layer7(p7)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.
            size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.
            size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.
            size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.
            size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.
            size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(
            conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view
            (conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().
            view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous()
            .view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous(
            ).view(conf7.size(0), -1)], dim=1)
        out = locs, confs
        return out


def ConvTransBNReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
        out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        padding=kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU6(
        inplace=True))


class SSD(nn.Module):

    def __init__(self, phase='train', num_classes=21):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.detector1 = nn.Sequential(Conv3x3BNReLU(in_channels=3,
            out_channels=64, stride=1), Conv3x3BNReLU(in_channels=64,
            out_channels=64, stride=1), nn.MaxPool2d(kernel_size=2, stride=
            2, ceil_mode=True), Conv3x3BNReLU(in_channels=64, out_channels=
            128, stride=1), Conv3x3BNReLU(in_channels=128, out_channels=128,
            stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True
            ), Conv3x3BNReLU(in_channels=128, out_channels=256, stride=1),
            Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1),
            Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1), nn.
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv3x3BNReLU(in_channels=256, out_channels=512, stride=1),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1),
            Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1))
        self.detector2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2,
            ceil_mode=True), Conv3x3BNReLU(in_channels=512, out_channels=
            512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512,
            stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512,
            stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True
            ), ConvTransBNReLU(in_channels=512, out_channels=1024,
            kernel_size=3, stride=2), Conv1x1BNReLU(in_channels=1024,
            out_channels=1024))
        self.detector3 = nn.Sequential(Conv1x1BNReLU(in_channels=1024,
            out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=
            512, stride=2))
        self.detector4 = nn.Sequential(Conv1x1BNReLU(in_channels=512,
            out_channels=128), Conv3x3BNReLU(in_channels=128, out_channels=
            256, stride=2))
        self.detector5 = nn.Sequential(Conv1x1BNReLU(in_channels=256,
            out_channels=128), Conv3x3ReLU(in_channels=128, out_channels=
            256, stride=1, padding=0))
        self.detector6 = nn.Sequential(Conv1x1BNReLU(in_channels=256,
            out_channels=128), Conv3x3ReLU(in_channels=128, out_channels=
            256, stride=1, padding=0))
        self.loc_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=512, out_channels=4 *
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 *
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=512, out_channels=6 *
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer4 = nn.Conv2d(in_channels=256, out_channels=6 *
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer5 = nn.Conv2d(in_channels=256, out_channels=4 *
            num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * 4,
            kernel_size=3, stride=1, padding=1)
        self.conf_layer6 = nn.Conv2d(in_channels=256, out_channels=4 *
            num_classes, kernel_size=3, stride=1, padding=1)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        elif self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        nn.init.constant_(m.bias, 0)
                    else:
                        nn.init.xavier_normal_(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_map1 = self.detector1(x)
        feature_map2 = self.detector2(feature_map1)
        feature_map3 = self.detector3(feature_map2)
        feature_map4 = self.detector4(feature_map3)
        feature_map5 = self.detector5(feature_map4)
        out = feature_map6 = self.detector6(feature_map5)
        loc1 = self.loc_layer1(feature_map1)
        conf1 = self.conf_layer1(feature_map1)
        loc2 = self.loc_layer2(feature_map2)
        conf2 = self.conf_layer2(feature_map2)
        loc3 = self.loc_layer3(feature_map3)
        conf3 = self.conf_layer3(feature_map3)
        loc4 = self.loc_layer4(feature_map4)
        conf4 = self.conf_layer4(feature_map4)
        loc5 = self.loc_layer5(feature_map5)
        conf5 = self.conf_layer5(feature_map5)
        loc6 = self.loc_layer6(feature_map6)
        conf6 = self.conf_layer6(feature_map6)
        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.
            size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc2.
            size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc3.
            size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.
            size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.
            size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.
            size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(
            conf1.size(0), -1), conf2.permute(0, 2, 3, 1).contiguous().view
            (conf2.size(0), -1), conf3.permute(0, 2, 3, 1).contiguous().
            view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous()
            .view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous(
            ).view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous
            ().view(conf6.size(0), -1)], dim=1)
        if self.phase == 'test':
            out = locs.view(locs.size(0), -1, 4), self.softmax(confs.view(
                confs.size(0), -1, self.num_classes))
        else:
            out = locs.view(locs.size(0), -1, 4), confs.view(confs.size(0),
                -1, self.num_classes)
        return out


class YOLO(nn.Module):

    def __init__(self):
        super(YOLO, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels
            =64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(
            kernel_size=2, stride=2), Conv3x3BNReLU(in_channels=64,
            out_channels=192), nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1BNReLU(in_channels=192, out_channels=128), Conv3x3BNReLU
            (in_channels=128, out_channels=256), Conv1x1BNReLU(in_channels=
            256, out_channels=256), Conv3x3BNReLU(in_channels=256,
            out_channels=512), nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU
            (in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=
            512, out_channels=256), Conv3x3BNReLU(in_channels=256,
            out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=
            256), Conv3x3BNReLU(in_channels=256, out_channels=512),
            Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU
            (in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=
            512, out_channels=512), Conv3x3BNReLU(in_channels=512,
            out_channels=1024), nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
            Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024, stride=2),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024),
            Conv3x3BNReLU(in_channels=1024, out_channels=1024))
        self.classifier = nn.Sequential(nn.Linear(1024 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 1470))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def DWConv3x3BNReLU(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=
        out_channels, kernel_size=3, stride=stride, padding=1, groups=
        in_channels), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class EP(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(EP, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            mid_channels), DWConv3x3BNReLU(mid_channels, mid_channels,
            stride), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class PEP(nn.Module):

    def __init__(self, in_channels, proj_channels, out_channels, stride,
        expansion_factor=6):
        super(PEP, self).__init__()
        self.stride = stride
        mid_channels = proj_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            proj_channels), Conv1x1BNReLU(proj_channels, mid_channels),
            DWConv3x3BNReLU(mid_channels, mid_channels, stride), Conv1x1BN(
            mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class FCA(nn.Module):

    def __init__(self, channel, ratio=8):
        super(FCA, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel,
            out_features=channel // ratio), nn.ReLU(inplace=True), nn.
            Linear(in_features=channel // ratio, out_features=channel), nn.
            Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class YOLO_Nano(nn.Module):

    def __init__(self, out_channel=75):
        super(YOLO_Nano, self).__init__()
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3,
            out_channels=12, stride=1), Conv3x3BNReLU(in_channels=12,
            out_channels=24, stride=2), PEP(in_channels=24, proj_channels=7,
            out_channels=24, stride=1), EP(in_channels=24, out_channels=70,
            stride=2), PEP(in_channels=70, proj_channels=25, out_channels=
            70, stride=1), PEP(in_channels=70, proj_channels=24,
            out_channels=70, stride=1), EP(in_channels=70, out_channels=150,
            stride=2), PEP(in_channels=150, proj_channels=56, out_channels=
            150, stride=1), Conv1x1BNReLU(in_channels=150, out_channels=150
            ), FCA(channel=150, ratio=8), PEP(in_channels=150,
            proj_channels=73, out_channels=150, stride=1), PEP(in_channels=
            150, proj_channels=71, out_channels=150, stride=1), PEP(
            in_channels=150, proj_channels=75, out_channels=150, stride=1))
        self.stage2 = nn.Sequential(EP(in_channels=150, out_channels=325,
            stride=2))
        self.stage3 = nn.Sequential(PEP(in_channels=325, proj_channels=132,
            out_channels=325, stride=1), PEP(in_channels=325, proj_channels
            =124, out_channels=325, stride=1), PEP(in_channels=325,
            proj_channels=141, out_channels=325, stride=1), PEP(in_channels
            =325, proj_channels=140, out_channels=325, stride=1), PEP(
            in_channels=325, proj_channels=137, out_channels=325, stride=1),
            PEP(in_channels=325, proj_channels=135, out_channels=325,
            stride=1), PEP(in_channels=325, proj_channels=133, out_channels
            =325, stride=1), PEP(in_channels=325, proj_channels=140,
            out_channels=325, stride=1))
        self.stage4 = nn.Sequential(EP(in_channels=325, out_channels=545,
            stride=2), PEP(in_channels=545, proj_channels=276, out_channels
            =545, stride=1), Conv1x1BNReLU(in_channels=545, out_channels=
            230), EP(in_channels=230, out_channels=489, stride=1), PEP(
            in_channels=489, proj_channels=213, out_channels=469, stride=1),
            Conv1x1BNReLU(in_channels=469, out_channels=189))
        self.stage5 = nn.Sequential(Conv1x1BNReLU(in_channels=189,
            out_channels=105), nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True))
        self.stage6 = nn.Sequential(PEP(in_channels=105 + 325,
            proj_channels=113, out_channels=325, stride=1), PEP(in_channels
            =325, proj_channels=99, out_channels=207, stride=1),
            Conv1x1BNReLU(in_channels=207, out_channels=98))
        self.stage7 = nn.Sequential(Conv1x1BNReLU(in_channels=98,
            out_channels=47), nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True))
        self.out_stage1 = nn.Sequential(PEP(in_channels=150 + 47,
            proj_channels=58, out_channels=122, stride=1), PEP(in_channels=
            122, proj_channels=52, out_channels=87, stride=1), PEP(
            in_channels=87, proj_channels=47, out_channels=93, stride=1),
            Conv1x1BNReLU(in_channels=93, out_channels=out_channel))
        self.out_stage2 = nn.Sequential(EP(in_channels=98, out_channels=183,
            stride=1), Conv1x1BNReLU(in_channels=183, out_channels=out_channel)
            )
        self.out_stage3 = nn.Sequential(EP(in_channels=189, out_channels=
            462, stride=1), Conv1x1BNReLU(in_channels=462, out_channels=
            out_channel))

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        x6 = self.stage6(torch.cat([x3, x5], dim=1))
        x7 = self.stage7(x6)
        out1 = self.out_stage1(torch.cat([x1, x7], dim=1))
        out2 = self.out_stage2(x6)
        out3 = self.out_stage3(x4)
        return out1, out2, out3


class Darknet19(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.feature = nn.Sequential(Conv3x3BNReLU(in_channels=3,
            out_channels=32), nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNReLU(in_channels=32, out_channels=64), nn.MaxPool2d(
            kernel_size=2, stride=2), Conv3x3BNReLU(in_channels=64,
            out_channels=128), Conv1x1BNReLU(in_channels=128, out_channels=
            64), Conv3x3BNReLU(in_channels=64, out_channels=128), nn.
            MaxPool2d(kernel_size=2, stride=2), Conv3x3BNReLU(in_channels=
            128, out_channels=256), Conv1x1BNReLU(in_channels=256,
            out_channels=128), Conv3x3BNReLU(in_channels=128, out_channels=
            256), nn.MaxPool2d(kernel_size=2, stride=2), Conv3x3BNReLU(
            in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=
            512, out_channels=256), Conv3x3BNReLU(in_channels=256,
            out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=
            256), Conv3x3BNReLU(in_channels=256, out_channels=512), nn.
            MaxPool2d(kernel_size=2, stride=2), Conv3x3BNReLU(in_channels=
            512, out_channels=1024), Conv1x1BNReLU(in_channels=1024,
            out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=
            1024), Conv1x1BNReLU(in_channels=1024, out_channels=512),
            Conv3x3BNReLU(in_channels=512, out_channels=1024))
        self.classifier = nn.Sequential(Conv1x1BNReLU(in_channels=1024,
            out_channels=num_classes), nn.AvgPool2d(kernel_size=7, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        x = torch.squeeze(x, dim=3).contiguous()
        x = torch.squeeze(x, dim=2).contiguous()
        out = self.softmax(x)
        return out


class Residual(nn.Module):

    def __init__(self, nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels // 2
        self.conv1x1 = Conv1x1BNReLU(in_channels=nchannels, out_channels=
            mid_channels)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels, out_channels
            =nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x


class Darknet19(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.first_conv = Conv3x3BNReLU(in_channels=3, out_channels=32)
        self.block1 = self._make_layers(in_channels=32, out_channels=64,
            block_num=1)
        self.block2 = self._make_layers(in_channels=64, out_channels=128,
            block_num=2)
        self.block3 = self._make_layers(in_channels=128, out_channels=256,
            block_num=8)
        self.block4 = self._make_layers(in_channels=256, out_channels=512,
            block_num=8)
        self.block5 = self._make_layers(in_channels=512, out_channels=1024,
            block_num=4)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layers(self, in_channels, out_channels, block_num):
        _layers = []
        _layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=
            out_channels, stride=2))
        for _ in range(block_num):
            _layers.append(Residual(nchannels=out_channels))
        return nn.Sequential(*_layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        out = self.softmax(x)
        return out


class FCN8s(nn.Module):

    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        vgg = torchvision.models.vgg16()
        features = list(vgg.features.children())
        self.padd = nn.ZeroPad2d([100, 100, 100, 100])
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])
        self.pool3_conv1x1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.pool4_conv1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.output5 = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, 4096,
            kernel_size=1), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(
            4096, num_classes, kernel_size=1))
        self.up_pool3_out = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=16, stride=8)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2)
        self.up_pool5_out = nn.ConvTranspose2d(num_classes, num_classes,
            kernel_size=4, stride=2)

    def forward(self, x):
        _, _, w, h = x.size()
        x = self.padd(x)
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        output5 = self.up_pool5_out(self.output5(pool5))
        pool4_out = self.pool4_conv1x1(0.01 * pool4)
        output4 = self.up_pool4_out(pool4_out[:, :, 5:5 + output5.size()[2],
            5:5 + output5.size()[3]] + output5)
        pool3_out = self.pool3_conv1x1(0.0001 * pool3)
        output3 = self.up_pool3_out(pool3_out[:, :, 9:9 + output4.size()[2],
            9:9 + output4.size()[3]] + output4)
        out = self.up_pool3_out(output3)
        out = out[:, :, 31:31 + h, 31:31 + w].contiguous()
        return out


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


class ShuffleNetUnits(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = (out_channels - in_channels if self.stride > 1 else
            out_channels)
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels,
            mid_channels, groups), ChannelShuffle(groups), Conv3x3BNReLU(
            mid_channels, mid_channels, stride, groups), Conv1x1BN(
            mid_channels, out_channels, groups))
        if self.stride > 1:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1
            ) if self.stride > 1 else out + x
        return self.relu(out)


class FisheyeMODNet(nn.Module):

    def __init__(self, groups=1, num_classes=2):
        super(FisheyeMODNet, self).__init__()
        layers = [4, 8, 4]
        self.stage1a = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(
            kernel_size=2, stride=2))
        self.stage2a = self._make_layer(24, 120, groups, layers[0])
        self.stage1b = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=
            24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(
            kernel_size=2, stride=2))
        self.stage2b = self._make_layer(24, 120, groups, layers[0])
        self.stage3 = self._make_layer(240, 480, groups, layers[1])
        self.stage4 = self._make_layer(480, 960, groups, layers[2])
        self.adapt_conv3 = nn.Conv2d(960, num_classes, kernel_size=1)
        self.adapt_conv2 = nn.Conv2d(480, num_classes, kernel_size=1)
        self.adapt_conv1 = nn.Conv2d(240, num_classes, kernel_size=1)
        self.up_sampling3 = nn.ConvTranspose2d(in_channels=num_classes,
            out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling2 = nn.ConvTranspose2d(in_channels=num_classes,
            out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling1 = nn.ConvTranspose2d(in_channels=num_classes,
            out_channels=num_classes, kernel_size=16, stride=8, padding=4)
        self.softmax = nn.Softmax(dim=1)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels
            =out_channels, stride=2, groups=groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels,
                out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        x = self.stage2a(self.stage1a(x))
        y = self.stage2b(self.stage1b(y))
        feature1 = torch.cat([x, y], dim=1)
        feature2 = self.stage3(feature1)
        feature3 = self.stage4(feature2)
        out3 = self.up_sampling3(self.adapt_conv3(feature3))
        out2 = self.up_sampling2(self.adapt_conv2(feature2) + out3)
        out1 = self.up_sampling1(self.adapt_conv1(feature1) + out2)
        out = self.softmax(out1)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = out_channels // 2
        self.bottleneck = nn.Sequential(ConvBNReLU(in_channels=in_channels,
            out_channels=mid_channels, kernel_size=1, stride=1), ConvBNReLU
            (in_channels=mid_channels, out_channels=mid_channels,
            kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=
            mid_channels, out_channels=out_channels, kernel_size=1, stride=1))
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out + self.shortcut(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_shanglianlm0525_PyTorch_Networks(_paritybench_base):
    pass
    def test_000(self):
        self._check(CBAM(*[], **{'channel': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_001(self):
        self._check(ChannelAttentionModule(*[], **{'channel': 64}), [torch.rand([4, 64, 4, 4])], {})

    def test_002(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(ContextBlock(*[], **{'inplanes': 4, 'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Conv2dCReLU(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(DenseBlock(*[], **{'num_layers': 1, 'inplances': 4, 'growth_rate': 4, 'bn_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(FCA(*[], **{'channel': 8}), [torch.rand([4, 8, 4, 8])], {})

    def test_007(self):
        self._check(FCN8s(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_008(self):
        self._check(FPN(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_009(self):
        self._check(FaceBoxes(*[], **{'num_classes': 4, 'phase': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_010(self):
        self._check(FireModule(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(GhostModule(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(GlobalContextBlock(*[], **{'inplanes': 4, 'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(HalfSplit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(HardSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(InceptionModules(*[], **{}), [torch.rand([4, 128, 64, 64])], {})

    @_fails_compile()
    def test_016(self):
        self._check(LBwithGCBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_017(self):
        self._check(LFFD(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_018(self):
        self._check(LFFDBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(LPN(*[], **{'nJoints': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_020(self):
        self._check(LossBranch(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_021(self):
        self._check(MDConv(*[], **{'nchannels': 4, 'kernel_sizes': [4, 4], 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_022(self):
        self._check(MiddleBottleneck(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_023(self):
        self._check(NonLocalBlock(*[], **{'channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(SE_Module(*[], **{'channel': 16}), [torch.rand([4, 16, 4, 16])], {})

    def test_025(self):
        self._check(SimpleBaseline(*[], **{'nJoints': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_026(self):
        self._check(SpatialAttentionModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_027(self):
        self._check(SqueezeAndExcite(*[], **{'in_channels': 4, 'out_channels': 4, 'se_kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_028(self):
        self._check(SqueezeNet(*[], **{}), [torch.rand([4, 3, 256, 256])], {})

    @_fails_compile()
    def test_029(self):
        self._check(_DenseLayer(*[], **{'inplace': 4, 'growth_rate': 4, 'bn_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_030(self):
        self._check(_TransitionLayer(*[], **{'inplace': 4, 'plance': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_031(self):
        self._check(cSE_Module(*[], **{'channel': 16}), [torch.rand([4, 16, 4, 16])], {})

    def test_032(self):
        self._check(sSE_Module(*[], **{'channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_033(self):
        self._check(scSE_Module(*[], **{'channel': 16}), [torch.rand([4, 16, 4, 16])], {})

