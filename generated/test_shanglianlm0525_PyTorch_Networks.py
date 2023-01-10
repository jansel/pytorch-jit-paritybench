import sys
_module = sys.modules[__name__]
del sys
AFF = _module
ANN = _module
CBAM = _module
CCNet = _module
GAM = _module
GlobalContextBlock = _module
NAM = _module
NonLocalBlock = _module
SENet = _module
SEvariants = _module
TripletAttention = _module
AlexNet = _module
DenseNet = _module
Efficientnet = _module
InceptionV1 = _module
InceptionV2 = _module
InceptionV3 = _module
InceptionV4 = _module
ResNeXt = _module
ResNet = _module
VGGNet = _module
repVGGNet = _module
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
MobileNetXt = _module
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
VoVNet = _module
VoVNetV2 = _module
YOLO = _module
YOLO_Nano = _module
YOLOv2 = _module
YOLOv3 = _module
DynamicReLU = _module
PyramidalConvolution = _module
SINet = _module
DeeplabV3Plus = _module
ENet = _module
FCN = _module
FastSCNN = _module
FisheyeMODNet = _module
ICNet = _module
LEDnet = _module
LRNnet = _module
LWnet = _module
SegNet = _module
Unet = _module
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


import torch


import torch.nn as nn


import torchvision


import numpy as np


from math import log


import math


from torch import nn


from functools import reduce


import torch.nn.functional as F


from torchvision.models.resnet import resnet18


from torchvision.models.resnet import resnet34


from torchvision.models.resnet import resnet50


from torchvision.models.resnet import resnet101


from torchvision.models.resnet import resnet152


class MS_CAM(nn.Module):

    def __init__(self, channel, ratio=16):
        super(MS_CAM, self).__init__()
        mid_channel = channel // ratio
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(mid_channel), nn.ReLU(inplace=True), nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channel))
        self.local_att = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(mid_channel), nn.ReLU(inplace=True), nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(channel))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        g_x = self.global_att(x)
        l_x = self.local_att(x)
        w = self.sigmoid(l_x * g_x.expand_as(l_x))
        return w * x


class AFF(nn.Module):

    def __init__(self):
        super(AFF, self).__init__()

    def forward(self, x):
        pass


class SpatialPyramidPooling(nn.Module):

    def __init__(self, output_sizes=[1, 3, 6, 8]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_layers = nn.ModuleList()
        for output_size in output_sizes:
            self.pool_layers.append(nn.AdaptiveMaxPool2d(output_size=output_size))

    def forward(self, x):
        outputs = []
        for pool_layer in self.pool_layers:
            outputs.append(pool_layer(x).flatten())
        out = torch.cat(outputs, dim=0)
        return out


class APNB(nn.Module):

    def __init__(self, channel):
        super(APNB, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class AFNB(nn.Module):

    def __init__(self, channel):
        super(AFNB, self).__init__()
        self.inter_channel = channel // 2
        self.output_sizes = [1, 3, 6, 8]
        self.sample_dim = np.sum([(size * size) for size in self.output_sizes])
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta_spp = SpatialPyramidPooling(self.output_sizes)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g_spp = SpatialPyramidPooling(self.output_sizes)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        xxx = self.conv_theta_spp(self.conv_theta(x))
        None
        x_theta = self.conv_theta_spp(self.conv_theta(x)).view(b, self.sample_dim, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g_spp(self.conv_g(x)).view(b, self.sample_dim, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class ChannelAttentionModule(nn.Module):

    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(nn.Conv2d(channel, channel // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(channel // ratio, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):

    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
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

    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.cbam = CBAM(channel=places * self.expansion)
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(places * self.expansion))
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


def INF(B, H, W):
    return -torch.diag(torch.tensor(float('inf')).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


class SE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channel // ratio, out_features=channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class ECA_Module(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(ECA_Module, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECA_ResNetBlock(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ECA_ResNetBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places * self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(places * self.expansion))
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
    return nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class ECA_ResNet(nn.Module):

    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ECA_ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(ECA_ResNetBlock(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(ECA_ResNetBlock(places * self.expansion, places))
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


class GAM(nn.Module):

    def __init__(self, channels, rate=4):
        super(GAM, self).__init__()
        mid_channels = channels // rate
        self.channel_attention = nn.Sequential(nn.Linear(channels, mid_channels), nn.ReLU(inplace=True), nn.Linear(mid_channels, channels))
        self.spatial_attention = nn.Sequential(nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(channels))

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


class GlobalContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
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
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
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


class NAM(nn.Module):

    def __init__(self, channel):
        super(NAM, self).__init__()
        self.channel = channel
        self.bn2 = nn.BatchNorm2d(self.channel, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.sigmoid(x) * residual
        return out


class NonLocalBlock(nn.Module):

    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class SE_ResNetBlock(nn.Module):

    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(SE_ResNetBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places * self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SE_ResNet(nn.Module):

    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(SE_ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(SE_ResNetBlock(in_places, places, stride, downsampling=True))
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
        self.excitation = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channel // ratio, out_features=channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSE_Module(nn.Module):

    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

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


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(1))
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttention(nn.Module):

    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()
        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return 1 / 3 * (x_out1 + x_out2 + x_out3)
        else:
            return 1 / 2 * (x_out1 + x_out2)


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0), nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0), nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features=256 * 6 * 6, out_features=4096), nn.Dropout(p=0.5), nn.Linear(in_features=4096, out_features=4096), nn.Linear(in_features=4096, out_features=num_classes))

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class _TransitionLayer(nn.Module):

    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(nn.BatchNorm2d(inplace), nn.ReLU(inplace=True), nn.Conv2d(in_channels=inplace, out_channels=plance, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition_layer(x)


class _DenseLayer(nn.Module):

    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(nn.BatchNorm2d(inplace), nn.ReLU(inplace=True), nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(bn_size * growth_rate), nn.ReLU(inplace=True), nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):

    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):

    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=1000):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)
        num_features = init_channels
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
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


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEBlock(nn.Module):

    def __init__(self, channels, ratio=16):
        super().__init__()
        mid_channels = channels // ratio
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True), Swish(), nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))


def Conv1x1BNAct(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), Swish())


def ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups), nn.BatchNorm2d(out_channels), Swish())


class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=6):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expansion_factor = expansion_factor
        mid_channels = in_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNAct(in_channels, mid_channels), ConvBNAct(mid_channels, mid_channels, kernel_size, stride, groups=mid_channels), SEBlock(mid_channels), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class EfficientNet(nn.Module):
    params = {'efficientnet_b0': (1.0, 1.0, 224, 0.2), 'efficientnet_b1': (1.0, 1.1, 240, 0.2), 'efficientnet_b2': (1.1, 1.2, 260, 0.3), 'efficientnet_b3': (1.2, 1.4, 300, 0.3), 'efficientnet_b4': (1.4, 1.8, 380, 0.4), 'efficientnet_b5': (1.6, 2.2, 456, 0.4), 'efficientnet_b6': (1.8, 2.6, 528, 0.5), 'efficientnet_b7': (2.0, 3.1, 600, 0.5)}

    def __init__(self, subtype='efficientnet_b0', num_classes=1000):
        super(EfficientNet, self).__init__()
        self.width_coeff = self.params[subtype][0]
        self.depth_coeff = self.params[subtype][1]
        self.dropout_rate = self.params[subtype][3]
        self.depth_div = 8
        self.stage1 = ConvBNAct(3, self._calculate_width(32), kernel_size=3, stride=2)
        self.stage2 = self.make_layer(self._calculate_width(32), self._calculate_width(16), kernel_size=3, stride=1, block=self._calculate_depth(1))
        self.stage3 = self.make_layer(self._calculate_width(16), self._calculate_width(24), kernel_size=3, stride=2, block=self._calculate_depth(2))
        self.stage4 = self.make_layer(self._calculate_width(24), self._calculate_width(40), kernel_size=5, stride=2, block=self._calculate_depth(2))
        self.stage5 = self.make_layer(self._calculate_width(40), self._calculate_width(80), kernel_size=3, stride=2, block=self._calculate_depth(3))
        self.stage6 = self.make_layer(self._calculate_width(80), self._calculate_width(112), kernel_size=5, stride=1, block=self._calculate_depth(3))
        self.stage7 = self.make_layer(self._calculate_width(112), self._calculate_width(192), kernel_size=5, stride=2, block=self._calculate_depth(4))
        self.stage8 = self.make_layer(self._calculate_width(192), self._calculate_width(320), kernel_size=3, stride=1, block=self._calculate_depth(1))
        self.classifier = nn.Sequential(Conv1x1BNAct(320, 1280), nn.AdaptiveAvgPool2d(1), nn.Dropout2d(0.2), Flatten(), nn.Linear(1280, num_classes))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def _calculate_width(self, x):
        x *= self.width_coeff
        new_x = max(self.depth_div, int(x + self.depth_div / 2) // self.depth_div * self.depth_div)
        if new_x < 0.9 * x:
            new_x += self.depth_div
        return int(new_x)

    def _calculate_depth(self, x):
        return int(math.ceil(x * self.depth_coeff))

    def make_layer(self, in_places, places, kernel_size, stride, block):
        layers = []
        layers.append(MBConvBlock(in_places, places, kernel_size, stride))
        for i in range(1, block):
            layers.append(MBConvBlock(places, places, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        out = self.classifier(x)
        return out


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InceptionV1Module(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()
        self.branch1_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3)
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5)
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)

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
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out


class InceptionV1(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV1, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1), nn.BatchNorm2d(64))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(192), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV1Module(in_channels=192, out_channels1=64, out_channels2reduce=96, out_channels2=128, out_channels3reduce=16, out_channels3=32, out_channels4=32), InceptionV1Module(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192, out_channels3reduce=32, out_channels3=96, out_channels4=64), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block4_1 = InceptionV1Module(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208, out_channels3reduce=16, out_channels3=48, out_channels4=64)
        if self.stage == 'train':
            self.aux_logits1 = InceptionAux(in_channels=512, out_channels=num_classes)
        self.block4_2 = nn.Sequential(InceptionV1Module(in_channels=512, out_channels1=160, out_channels2reduce=112, out_channels2=224, out_channels3reduce=24, out_channels3=64, out_channels4=64), InceptionV1Module(in_channels=512, out_channels1=128, out_channels2reduce=128, out_channels2=256, out_channels3reduce=24, out_channels3=64, out_channels4=64), InceptionV1Module(in_channels=512, out_channels1=112, out_channels2reduce=144, out_channels2=288, out_channels3reduce=32, out_channels3=64, out_channels4=64))
        if self.stage == 'train':
            self.aux_logits2 = InceptionAux(in_channels=528, out_channels=num_classes)
        self.block4_3 = nn.Sequential(InceptionV1Module(in_channels=528, out_channels1=256, out_channels2reduce=160, out_channels2=320, out_channels3reduce=32, out_channels3=128, out_channels4=128), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block5 = nn.Sequential(InceptionV1Module(in_channels=832, out_channels1=256, out_channels2reduce=160, out_channels2=320, out_channels3reduce=32, out_channels3=128, out_channels4=128), InceptionV1Module(in_channels=832, out_channels1=384, out_channels2reduce=192, out_channels2=384, out_channels3reduce=48, out_channels3=128, out_channels4=128))
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

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


def ConvBNReLUFactorization(in_channels, out_channels, kernel_sizes, paddings):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InceptionV2ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)], dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV2(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV2, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV2ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=64, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32), InceptionV2ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=64, out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=64), InceptionV3ModuleD(in_channels=320, out_channels1reduce=128, out_channels1=160, out_channels2reduce=64, out_channels2=96))
        self.block4 = nn.Sequential(InceptionV2ModuleB(in_channels=576, out_channels1=224, out_channels2reduce=64, out_channels2=96, out_channels3reduce=96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=192, out_channels2reduce=96, out_channels2=128, out_channels3reduce=96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=160, out_channels2reduce=128, out_channels2=160, out_channels3reduce=128, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=96, out_channels2reduce=128, out_channels2=192, out_channels3reduce=160, out_channels3=160, out_channels4=128), InceptionV3ModuleD(in_channels=576, out_channels1reduce=128, out_channels1=192, out_channels2reduce=192, out_channels2=256))
        self.block5 = nn.Sequential(InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=160, out_channels3=112, out_channels4=128), InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=192, out_channels3=112, out_channels4=128))
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

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=5, padding=2))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[7, 1], paddings=[3, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[7, 1], paddings=[3, 0]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_sizes=[7, 1], paddings=[3, 0]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV3ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)], dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleE(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleE, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 7], paddings=[0, 3]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[7, 1], paddings=[3, 0]), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV3(nn.Module):

    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2), ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1), ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels=80, kernel_size=3, stride=1), ConvBNReLU(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(InceptionV3ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32), InceptionV3ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64), InceptionV3ModuleA(in_channels=288, out_channels1=64, out_channels2reduce=48, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=64))
        self.block4 = nn.Sequential(InceptionV3ModuleD(in_channels=288, out_channels1reduce=384, out_channels1=384, out_channels2reduce=64, out_channels2=96), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=128, out_channels2=192, out_channels3reduce=128, out_channels3=192, out_channels4=192), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=160, out_channels2=192, out_channels3reduce=160, out_channels3=192, out_channels4=192), InceptionV3ModuleB(in_channels=768, out_channels1=192, out_channels2reduce=192, out_channels2=192, out_channels3reduce=192, out_channels3=192, out_channels4=192))
        if self.stage == 'train':
            self.aux_logits = InceptionAux(in_channels=768, out_channels=num_classes)
        self.block5 = nn.Sequential(InceptionV3ModuleE(in_channels=768, out_channels1reduce=192, out_channels1=320, out_channels2reduce=192, out_channels2=192), InceptionV3ModuleC(in_channels=1280, out_channels1=320, out_channels2reduce=384, out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192), InceptionV3ModuleC(in_channels=2048, out_channels1=320, out_channels2reduce=384, out_channels2=384, out_channels3reduce=448, out_channels3=384, out_channels4=192))
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

    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=2, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places * self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(places * self.expansion))
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

    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True), nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(places * self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(places * self.expansion))
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

    def __init__(self, backbone='resnet50', pretrained_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        else:
            backbone = resnet152(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        if pretrained_path:
            backbone.load_state_dict(torch.load(pretrained_path))
        self.early_extractor = nn.Sequential(*list(backbone.children())[:5])
        self.later_extractor = nn.Sequential(*list(backbone.children())[5:7])
        conv4_block1 = self.later_extractor[-1][0]
        conv4_block1.conv1.stride = 1, 1
        conv4_block1.conv2.stride = 1, 1
        conv4_block1.downsample[0].stride = 1, 1

    def forward(self, x):
        x = self.early_extractor(x)
        out = self.later_extractor(x)
        return out, x


def Conv3x3BNReLU(in_channels, out_channels, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class VGG(nn.Module):

    def __init__(self, block_nums, num_classes=1000):
        super(VGG, self).__init__()
        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])
        self.classifier = nn.Sequential(nn.Linear(in_features=512 * 7 * 7, out_features=4096), nn.Dropout(p=0.2), nn.Linear(in_features=4096, out_features=4096), nn.Dropout(p=0.2), nn.Linear(in_features=4096, out_features=num_classes))
        self._init_params()

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels, out_channels))
        for i in range(1, block_num):
            layers.append(Conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def Conv3x3BN(in_channels, out_channels, stride, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels))


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=groups, bias=True)
        else:
            self.conv1 = Conv3x3BN(in_channels, out_channels, stride=stride, groups=groups, bias=False)
            self.conv2 = Conv1x1BN(in_channels, out_channels, stride=stride, groups=groups, bias=False)
            self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        if self.identity is None:
            return self.act(self.conv1(x) + self.conv2(x))
        else:
            return self.act(self.conv1(x) + self.conv2(x) + self.identity(x))


class RepVGG(nn.Module):

    def __init__(self, block_nums, width_multiplier=None, group=1, num_classes=1000, deploy=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy
        self.group = group
        assert len(width_multiplier) == 4
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=min(64, int(64 * width_multiplier[0])), stride=2, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_layers(in_channels=min(64, int(64 * width_multiplier[0])), out_channels=int(64 * width_multiplier[0]), stride=2, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=int(64 * width_multiplier[0]), out_channels=int(128 * width_multiplier[1]), stride=2, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=int(128 * width_multiplier[1]), out_channels=int(256 * width_multiplier[2]), stride=2, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=int(256 * width_multiplier[2]), out_channels=int(512 * width_multiplier[3]), stride=2, block_num=block_nums[3])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self._init_params()

    def _make_layers(self, in_channels, out_channels, stride, block_num):
        layers = []
        layers.append(RepVGGBlock(in_channels, out_channels, stride=stride, groups=self.group if self.cur_layer_idx % 2 == 0 else 1, deploy=self.deploy))
        self.cur_layer_idx += 1
        for i in range(block_num):
            layers.append(RepVGGBlock(out_channels, out_channels, stride=1, groups=self.group if self.cur_layer_idx % 2 == 0 else 1, deploy=self.deploy))
            self.cur_layer_idx += 1
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


class Conv2dCReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dCReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        out = torch.cat([x, -x], dim=1)
        return self.relu(out)


class InceptionModules(nn.Module):

    def __init__(self):
        super(InceptionModules, self).__init__()
        self.branch1_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.branch1_conv1_bn = nn.BatchNorm2d(32)
        self.branch2_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.branch2_conv1_bn = nn.BatchNorm2d(32)
        self.branch3_conv1 = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1, stride=1)
        self.branch3_conv1_bn = nn.BatchNorm2d(24)
        self.branch3_conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv1 = nn.Conv2d(in_channels=128, out_channels=24, kernel_size=1, stride=1)
        self.branch4_conv1_bn = nn.BatchNorm2d(24)
        self.branch4_conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_conv2_bn = nn.BatchNorm2d(32)
        self.branch4_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_conv3_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x1 = self.branch1_conv1_bn(self.branch1_conv1(x))
        x2 = self.branch2_conv1_bn(self.branch2_conv1(self.branch2_pool(x)))
        x3 = self.branch3_conv2_bn(self.branch3_conv2(self.branch3_conv1_bn(self.branch3_conv1(x))))
        x4 = self.branch4_conv3_bn(self.branch4_conv3(self.branch4_conv2_bn(self.branch4_conv2(self.branch4_conv1_bn(self.branch4_conv1(x))))))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class FaceBoxes(nn.Module):

    def __init__(self, num_classes, phase):
        super(FaceBoxes, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.RapidlyDigestedConvolutionalLayers = nn.Sequential(Conv2dCReLU(in_channels=3, out_channels=24, kernel_size=7, stride=4, padding=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), Conv2dCReLU(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=2), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.MultipleScaleConvolutionalLayers = nn.Sequential(InceptionModules(), InceptionModules(), InceptionModules())
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer1 = nn.Conv2d(in_channels=128, out_channels=21 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=128, out_channels=21 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer2 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer3 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
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
        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc2.size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(conf1.size(0), -1), conf2.permute(0, 2, 3, 1).contiguous().view(conf2.size(0), -1), conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1)], dim=1)
        if self.phase == 'test':
            out = locs.view(locs.size(0), -1, 4), self.softmax(confs.view(-1, self.num_classes))
        else:
            out = locs.view(locs.size(0), -1, 4), confs.view(-1, self.num_classes)
        return out


def Conv1x1ReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.ReLU6(inplace=True))


class LossBranch(nn.Module):

    def __init__(self, in_channels, mid_channels=64):
        super(LossBranch, self).__init__()
        self.conv1 = Conv1x1ReLU(in_channels, mid_channels)
        self.conv2_score = Conv1x1ReLU(mid_channels, mid_channels)
        self.classify = nn.Conv2d(in_channels=mid_channels, out_channels=2, kernel_size=1, stride=1)
        self.conv2_bbox = Conv1x1ReLU(mid_channels, mid_channels)
        self.regress = nn.Conv2d(in_channels=mid_channels, out_channels=4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        cls = self.classify(self.conv2_score(x))
        reg = self.regress(self.conv2_bbox(x))
        return cls, reg


def Conv3x3ReLU(in_channels, out_channels, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding), nn.ReLU6(inplace=True))


class LFFDBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(LFFDBlock, self).__init__()
        mid_channels = out_channels
        self.downsampling = True if stride == 2 else False
        if self.downsampling:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=0)
        self.branch1_relu1 = nn.ReLU6(inplace=True)
        self.branch1_conv1 = Conv3x3ReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1, padding=1)
        self.branch1_conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        if self.downsampling:
            x = self.conv(x)
        out = self.branch1_conv2(self.branch1_conv1(self.branch1_relu1(x)))
        return self.relu(out + x)


class LFFD(nn.Module):

    def __init__(self, classes_num=2):
        super(LFFD, self).__init__()
        self.tiny_part1 = nn.Sequential(Conv3x3ReLU(in_channels=3, out_channels=64, stride=2, padding=0), LFFDBlock(in_channels=64, out_channels=64, stride=2), LFFDBlock(in_channels=64, out_channels=64, stride=1), LFFDBlock(in_channels=64, out_channels=64, stride=1))
        self.tiny_part2 = LFFDBlock(in_channels=64, out_channels=64, stride=1)
        self.small_part1 = LFFDBlock(in_channels=64, out_channels=64, stride=2)
        self.small_part2 = LFFDBlock(in_channels=64, out_channels=64, stride=1)
        self.medium_part = nn.Sequential(LFFDBlock(in_channels=64, out_channels=128, stride=2), LFFDBlock(in_channels=128, out_channels=128, stride=1))
        self.large_part1 = LFFDBlock(in_channels=128, out_channels=128, stride=2)
        self.large_part2 = LFFDBlock(in_channels=128, out_channels=128, stride=1)
        self.large_part3 = LFFDBlock(in_channels=128, out_channels=128, stride=1)
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
        cls = torch.cat([cls1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls2.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls3.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls4.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls5.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls6.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls7.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), cls8.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1)], dim=1)
        loc = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc8.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1)], dim=1)
        out = cls, loc
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
        self.SEblock = nn.Sequential(nn.Linear(in_features=in_channels, out_features=mid_channels), nn.ReLU6(inplace=True), nn.Linear(in_features=mid_channels, out_features=out_channels), HardSwish(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


def VarGConv(in_channels, out_channels, kernel_size, stride, S):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels // S, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU())


def VarGPointConv(in_channels, out_channels, stride, S, isRelu):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, groups=in_channels // S, bias=False), nn.BatchNorm2d(out_channels), nn.PReLU() if isRelu else nn.Sequential())


class VarGBlock_S1(nn.Module):

    def __init__(self, in_plances, kernel_size, stride=1, S=8):
        super(VarGBlock_S1, self).__init__()
        plances = 2 * in_plances
        self.varGConv1 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv1 = VarGPointConv(plances, in_plances, stride, S, isRelu=True)
        self.varGConv2 = VarGConv(in_plances, plances, kernel_size, stride, S)
        self.varGPointConv2 = VarGPointConv(plances, in_plances, stride, S, isRelu=False)
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
        self.varGConvBlock_branch1 = nn.Sequential(VarGConv(in_plances, plances, kernel_size, stride, S), VarGPointConv(plances, plances, 1, S, isRelu=True))
        self.varGConvBlock_branch2 = nn.Sequential(VarGConv(in_plances, plances, kernel_size, stride, S), VarGPointConv(plances, plances, 1, S, isRelu=True))
        self.varGConvBlock_3 = nn.Sequential(VarGConv(plances, plances * 2, kernel_size, 1, S), VarGPointConv(plances * 2, plances, 1, S, isRelu=False))
        self.shortcut = nn.Sequential(VarGConv(in_plances, plances, kernel_size, stride, S), VarGPointConv(plances, plances, 1, S, isRelu=False))
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
        self.varGConvBlock = nn.Sequential(VarGConv(in_plances, in_plances, kernel_size, 2, S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=True), VarGConv(in_plances, in_plances, kernel_size, 1, S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=False))
        self.shortcut = nn.Sequential(VarGConv(in_plances, in_plances, kernel_size, 2, S), VarGPointConv(in_plances, in_plances, 1, S, isRelu=False))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.varGConvBlock(x)
        out += x
        return out


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class TailEmbedding(nn.Module):

    def __init__(self, in_plances, plances=512, S=8):
        super(TailEmbedding, self).__init__()
        self.embedding = nn.Sequential(Conv1x1BNReLU(in_plances, 1024), nn.Conv2d(1024, 1024, 7, 1, padding=0, groups=1024 // S, bias=False), nn.Conv2d(1024, 512, 1, 1, padding=0, groups=512, bias=False))
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
        self.stage2 = nn.Sequential(VarGBlock_S2(40, 3, 2), VarGBlock_S1(80, 3, 1), VarGBlock_S1(80, 3, 1))
        self.stage3 = nn.Sequential(VarGBlock_S2(80, 3, 2), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1), VarGBlock_S1(160, 3, 1))
        self.stage4 = nn.Sequential(VarGBlock_S2(160, 3, 2), VarGBlock_S1(320, 3, 1), VarGBlock_S1(320, 3, 1), VarGBlock_S1(320, 3, 1))
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
        self.bottleneck = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1))
        self.shortcut = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        return out + self.shortcut(x)


class HourglassModule(nn.Module):

    def __init__(self, nChannels=256, nModules=2, numReductions=4):
        super(HourglassModule, self).__init__()
        self.nChannels = nChannels
        self.nModules = nModules
        self.numReductions = numReductions
        self.residual_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.after_pool_block = self._make_residual_layer(self.nModules, self.nChannels)
        if numReductions > 1:
            self.hourglass_module = HourglassModule(self.nChannels, self.numReductions - 1, self.nModules)
        else:
            self.num1res_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.lowres_block = self._make_residual_layer(self.nModules, self.nChannels)
        self.upsample = nn.Upsample(scale_factor=2)

    def _make_residual_layer(self, nModules, nChannels):
        _residual_blocks = []
        for _ in range(nModules):
            _residual_blocks.append(ResidualBlock(in_channels=nChannels, out_channels=nChannels))
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
        self.first_conv = ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.residual_block1 = ResidualBlock(in_channels=64, out_channels=128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_block2 = ResidualBlock(in_channels=128, out_channels=128)
        self.residual_block3 = ResidualBlock(in_channels=128, out_channels=256)
        self.hourglass_module1 = HourglassModule(nChannels=256, nModules=2, numReductions=4)
        self.hourglass_module2 = HourglassModule(nChannels=256, nModules=2, numReductions=4)
        self.after_hourglass_conv1 = ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.out_conv1 = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1)
        self.remap_conv1 = nn.Conv2d(in_channels=nJoints, out_channels=256, kernel_size=1, stride=1)
        self.after_hourglass_conv2 = ConvBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.proj_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1)
        self.remap_conv2 = nn.Conv2d(in_channels=nJoints, out_channels=256, kernel_size=1, stride=1)

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


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
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
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
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


class LBwithGCBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LBwithGCBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(planes)
        self.conv1_bn_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv2_bn_relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, stride=1, padding=0)
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


def GroupDeconv(inplanes, planes, kernel_size, stride, padding, output_padding):
    groups = computeGCD(inplanes, planes)
    return nn.Sequential(nn.ConvTranspose2d(in_channels=inplanes, out_channels=2 * planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups), nn.Conv2d(2 * planes, planes, kernel_size=1, stride=1, padding=0))


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
        self.final_layer = nn.Conv2d(in_channels=self.inplanes, out_channels=nJoints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride), nn.BatchNorm2d(planes * block.expansion))
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
            layers.append(GroupDeconv(inplanes=self.inplanes, planes=planes, kernel_size=4, stride=2, padding=1, output_padding=0))
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, 64, 3)
        self.layer2 = self._make_layer(ResBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 3, stride=2)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=nJoints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
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


def conf_centernessLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))


def locLayer(in_channels, out_channels):
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))


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
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer3 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer4 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer5 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer6 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=36)
        self.conf_centerness_layer7 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        conf3, centerness3 = conf_centerness3.split([self.num_classes, 1], dim=1)
        loc4 = self.loc_layer4(p4)
        conf_centerness4 = self.conf_centerness_layer4(p4)
        conf4, centerness4 = conf_centerness4.split([self.num_classes, 1], dim=1)
        loc5 = self.loc_layer5(p5)
        conf_centerness5 = self.conf_centerness_layer5(p5)
        conf5, centerness5 = conf_centerness5.split([self.num_classes, 1], dim=1)
        loc6 = self.loc_layer6(p6)
        conf_centerness6 = self.conf_centerness_layer6(p6)
        conf6, centerness6 = conf_centerness6.split([self.num_classes, 1], dim=1)
        loc7 = self.loc_layer7(p7)
        conf_centerness7 = self.conf_centerness_layer7(p7)
        conf7, centerness7 = conf_centerness7.split([self.num_classes, 1], dim=1)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        centernesses = torch.cat([centerness3.permute(0, 2, 3, 1).contiguous().view(centerness3.size(0), -1), centerness4.permute(0, 2, 3, 1).contiguous().view(centerness4.size(0), -1), centerness5.permute(0, 2, 3, 1).contiguous().view(centerness5.size(0), -1), centerness6.permute(0, 2, 3, 1).contiguous().view(centerness6.size(0), -1), centerness7.permute(0, 2, 3, 1).contiguous().view(centerness7.size(0), -1)], dim=1)
        out = locs, confs, centernesses
        return out


def DW_Conv3x3BNReLU(in_channels, out_channels, stride, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class GhostModule(nn.Module):

    def __init__(self, in_channels, out_channels, s=2, kernel_size=1, stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channels = out_channels // s
        ghost_channels = intrinsic_channels * (s - 1)
        self.primary_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=intrinsic_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(intrinsic_channels), nn.ReLU(inplace=True) if use_relu else nn.Sequential())
        self.cheap_op = DW_Conv3x3BNReLU(in_channels=intrinsic_channels, out_channels=ghost_channels, stride=stride, groups=intrinsic_channels)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_op(y)
        out = torch.cat([y, z], dim=1)
        return out


class GhostBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, se_kernel_size=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.bottleneck = nn.Sequential(GhostModule(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, use_relu=True), DW_Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=stride, groups=mid_channels) if self.stride > 1 else nn.Sequential(), SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size) if use_se else nn.Sequential(), GhostModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, use_relu=False))
        if self.stride > 1:
            self.shortcut = DW_Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out


class GhostNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(GhostNet, self).__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.features = nn.Sequential(GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, use_se=False), GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2, use_se=True, se_kernel_size=28), GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28), GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28), GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2, use_se=False), GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1, use_se=False), GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14), GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14), GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2, use_se=True, se_kernel_size=7), GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True, se_kernel_size=7), GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True, se_kernel_size=7))
        self.last_stage = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1), nn.BatchNorm2d(960), nn.ReLU6(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1), nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1), nn.ReLU6(inplace=True))
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


class MDConv(nn.Module):

    def __init__(self, nchannels, kernel_sizes, stride):
        super(MDConv, self).__init__()
        self.nchannels = nchannels
        self.groups = len(kernel_sizes)
        self.split_channels = [(nchannels // self.groups) for _ in range(self.groups)]
        self.split_channels[0] += nchannels - sum(self.split_channels)
        self.layers = []
        for i in range(self.groups):
            self.layers.append(nn.Conv2d(in_channels=self.split_channels[i], out_channels=self.split_channels[i], kernel_size=kernel_sizes[i], stride=stride, padding=int(kernel_sizes[i] // 2), groups=self.split_channels[i]))

    def forward(self, x):
        split_x = torch.split(x, self.split_channels, dim=1)
        outputs = [layer(sp_x) for layer, sp_x in zip(self.layers, split_x)]
        return torch.cat(outputs, dim=1)


def Conv1x1BNActivation(in_channels, out_channels, activate):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish())


class MDConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, stride, expand_ratio, activate='relu', se_ratio=1):
        super(MDConvBlock, self).__init__()
        self.stride = stride
        self.se_ratio = se_ratio
        mid_channels = in_channels * expand_ratio
        self.expand_conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.md_conv = nn.Sequential(MDConv(mid_channels, kernel_sizes, stride), nn.BatchNorm2d(mid_channels), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish(inplace=True))
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
    mixnet_s = [(16, 16, [3], 1, 1, 'ReLU', 0.0), (16, 24, [3], 2, 6, 'ReLU', 0.0), (24, 24, [3], 1, 3, 'ReLU', 0.0), (24, 40, [3, 5, 7], 2, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25), (80, 80, [3, 5], 1, 6, 'Swish', 0.25), (80, 80, [3, 5], 1, 6, 'Swish', 0.25), (80, 120, [3, 5, 7], 1, 6, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 200, [3, 5, 7, 9, 11], 2, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]
    mixnet_m = [(24, 24, [3], 1, 1, 'ReLU', 0.0), (24, 32, [3, 5, 7], 2, 6, 'ReLU', 0.0), (32, 32, [3], 1, 3, 'ReLU', 0.0), (32, 40, [3, 5, 7, 9], 2, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 40, [3, 5], 1, 6, 'Swish', 0.5), (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25), (80, 120, [3], 1, 6, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5), (120, 200, [3, 5, 7, 9], 2, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5), (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]

    def __init__(self, type='mixnet_s'):
        super(MixNet, self).__init__()
        if type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
        elif type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
        self.stem = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=stem_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(stem_channels), HardSwish(inplace=True))
        layers = []
        for in_channels, out_channels, kernel_sizes, stride, expand_ratio, activate, se_ratio in config:
            layers.append(MDConvBlock(in_channels, out_channels, kernel_sizes=kernel_sizes, stride=stride, expand_ratio=expand_ratio, activate=activate, se_ratio=se_ratio))
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
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels), nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        self.bottleneck = nn.Sequential(BottleneckV1(32, 64, stride=1), BottleneckV1(64, 128, stride=2), BottleneckV1(128, 128, stride=1), BottleneckV1(128, 256, stride=2), BottleneckV1(256, 256, stride=1), BottleneckV1(256, 512, stride=2), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 512, stride=1), BottleneckV1(512, 1024, stride=2), BottleneckV1(1024, 1024, stride=1))
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
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, mid_channels), Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels), Conv1x1BN(mid_channels, out_channels))
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
        self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, block_num=1)
        self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, block_num=2)
        self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, block_num=3)
        self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, block_num=4)
        self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, block_num=3)
        self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=2, block_num=3)
        self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, block_num=1)
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


def ConvBNActivation(in_channels, out_channels, kernel_size, stride, activate):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=in_channels), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True) if activate == 'relu' else HardSwish())


class SEInvertedBottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size, stride, activate)
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels, activate)
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
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), HardSwish(inplace=True))
        if type == 'large':
            self.large_bottleneck = nn.Sequential(SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2, activate='relu', use_se=True, se_kernel_size=28), SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, activate='relu', use_se=True, se_kernel_size=28), SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, activate='relu', use_se=True, se_kernel_size=28), SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=1, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=2, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=1, activate='hswish', use_se=False), SEInvertedBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7))
            self.large_last_stage = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1), nn.BatchNorm2d(960), HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1), nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1), HardSwish(inplace=True))
        else:
            self.small_bottleneck = nn.Sequential(SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2, activate='relu', use_se=True, se_kernel_size=56), SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1, activate='relu', use_se=False), SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=14), SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7), SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7))
            self.small_last_stage = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1), nn.BatchNorm2d(576), HardSwish(inplace=True), nn.AvgPool2d(kernel_size=7, stride=1), nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1), HardSwish(inplace=True))
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


class SandglassBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(SandglassBlock, self).__init__()
        self.stride = stride
        mid_channels = in_channels // expansion_factor
        self.identity = stride == 1 and in_channels == out_channels
        self.bottleneck = nn.Sequential(Conv3x3BNReLU(in_channels, in_channels, 1, groups=in_channels), Conv1x1BN(in_channels, mid_channels), Conv1x1BNReLU(mid_channels, out_channels), Conv3x3BN(out_channels, out_channels, stride, groups=out_channels))

    def forward(self, x):
        out = self.bottleneck(x)
        if self.identity:
            return out + x
        else:
            return out


class MobileNetXt(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNetXt, self).__init__()
        self.first_conv = Conv3x3BNReLU(3, 32, 2, groups=1)
        self.layer1 = self.make_layer(in_channels=32, out_channels=96, stride=2, expansion_factor=2, block_num=1)
        self.layer2 = self.make_layer(in_channels=96, out_channels=144, stride=1, expansion_factor=6, block_num=1)
        self.layer3 = self.make_layer(in_channels=144, out_channels=192, stride=2, expansion_factor=6, block_num=3)
        self.layer4 = self.make_layer(in_channels=192, out_channels=288, stride=2, expansion_factor=6, block_num=3)
        self.layer5 = self.make_layer(in_channels=288, out_channels=384, stride=1, expansion_factor=6, block_num=4)
        self.layer6 = self.make_layer(in_channels=384, out_channels=576, stride=2, expansion_factor=6, block_num=4)
        self.layer7 = self.make_layer(in_channels=576, out_channels=960, stride=1, expansion_factor=6, block_num=2)
        self.layer8 = self.make_layer(in_channels=960, out_channels=1280, stride=1, expansion_factor=6, block_num=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=1280, out_features=num_classes)

    def make_layer(self, in_channels, out_channels, stride, expansion_factor, block_num):
        layers = []
        layers.append(SandglassBlock(in_channels, out_channels, stride, expansion_factor))
        for i in range(1, block_num):
            layers.append(SandglassBlock(out_channels, out_channels, 1, expansion_factor))
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
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class ShuffleNetUnits(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride
        out_channels = out_channels - in_channels if self.stride > 1 else out_channels
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, mid_channels, groups), ChannelShuffle(groups), Conv3x3BNReLU(mid_channels, mid_channels, stride, groups), Conv1x1BN(mid_channels, out_channels, groups))
        if self.stride > 1:
            self.shortcut = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat([self.shortcut(x), out], dim=1) if self.stride > 1 else out + x
        return self.relu(out)


class ShuffleNet(nn.Module):

    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage2 = self._make_layer(24, planes[0], groups, layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], groups, layers[1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], groups, layers[2], False)
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2] * 7 * 7, out_features=num_classes)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=1 if is_stage2 else groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
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

    def __init__(self, dim=1):
        super(HalfSplit, self).__init__()
        self.dim = dim

    def forward(self, input):
        splits = torch.chunk(input, 2, dim=self.dim)
        return splits[0], splits[1]


class ShuffleNetV2(nn.Module):

    def __init__(self, planes, layers, groups, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.groups = groups
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage2 = self._make_layer(24, planes[0], layers[0], True)
        self.stage3 = self._make_layer(planes[0], planes[1], layers[1], False)
        self.stage4 = self._make_layer(planes[1], planes[2], layers[2], False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[2], out_features=num_classes)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, block_num, is_stage2):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=1 if is_stage2 else self.groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=self.groups))
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
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class FireModule(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(FireModule, self).__init__()
        mid_channels = out_channels // 4
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1)
        self.squeeze_relu = nn.ReLU6(inplace=True)
        self.expand3x3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.expand3x3_relu = nn.ReLU6(inplace=True)
        self.expand1x1 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1)
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
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(96), nn.ReLU6(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), FireModule(in_channels=96, out_channels=64), FireModule(in_channels=128, out_channels=64), FireModule(in_channels=128, out_channels=128), nn.MaxPool2d(kernel_size=3, stride=2), FireModule(in_channels=256, out_channels=128), FireModule(in_channels=256, out_channels=192), FireModule(in_channels=384, out_channels=192), FireModule(in_channels=384, out_channels=256), nn.MaxPool2d(kernel_size=3, stride=2), FireModule(in_channels=512, out_channels=256), nn.Dropout(p=0.5), nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=13, stride=1))

    def forward(self, x):
        out = self.bottleneck(x)
        return out.view(out.size(1), -1)


def ConvBN(in_channels, out_channels, kernel_size, stride, padding=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(out_channels))


def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))


def ReluSeparableConvolution(in_channels, out_channels):
    return nn.Sequential(nn.ReLU6(inplace=True), SeparableConvolution(in_channels, out_channels))


class EntryBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, first_relu=True):
        super(EntryBottleneck, self).__init__()
        mid_channels = out_channels
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels) if first_relu else SeparableConvolution(in_channels=in_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


class MiddleBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MiddleBottleneck, self).__init__()
        mid_channels = out_channels
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels))

    def forward(self, x):
        out = self.bottleneck(x)
        return out + x


class ExitBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ExitBottleneck, self).__init__()
        mid_channels = in_channels
        self.shortcut = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.bottleneck = nn.Sequential(ReluSeparableConvolution(in_channels=in_channels, out_channels=mid_channels), ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out + x


def ConvBNRelu(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(ConvBN(in_channels, out_channels, kernel_size, stride), nn.ReLU6(inplace=True))


def SeparableConvolutionRelu(in_channels, out_channels):
    return nn.Sequential(SeparableConvolution(in_channels, out_channels), nn.ReLU6(inplace=True))


class Xception(nn.Module):

    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.entryFlow = nn.Sequential(ConvBNRelu(in_channels=3, out_channels=32, kernel_size=3, stride=2), ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1), EntryBottleneck(in_channels=64, out_channels=128, first_relu=False), EntryBottleneck(in_channels=128, out_channels=256, first_relu=True), EntryBottleneck(in_channels=256, out_channels=728, first_relu=True))
        self.middleFlow = nn.Sequential(MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728), MiddleBottleneck(in_channels=728, out_channels=728))
        self.exitFlow = nn.Sequential(ExitBottleneck(in_channels=728, out_channels=1024), SeparableConvolutionRelu(in_channels=1024, out_channels=1536), SeparableConvolutionRelu(in_channels=1536, out_channels=2048), nn.AdaptiveAvgPool2d((1, 1)))
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entryFlow(x)
        x = self.middleFlow(x)
        x = self.exitFlow(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


def Conv1x1BnRelu(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def downSampling1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def downSampling2(in_channels, out_channels):
    return nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), downSampling1(in_channels=in_channels, out_channels=out_channels))


def upSampling1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True), nn.Upsample(scale_factor=2, mode='nearest'))


def upSampling2(in_channels, out_channels):
    return nn.Sequential(upSampling1(in_channels, out_channels), nn.Upsample(scale_factor=2, mode='nearest'))


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
        self.weight_level = nn.Conv2d(funsed_channel * 3, 3, kernel_size=1, stride=1, padding=0)
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
        fused_level = level_x * weight_level[:, 0, :, :] + level_y * weight_level[:, 1, :, :] + level_z * weight_level[:, 2, :, :]
        out = self.expand_conv(fused_level)
        return out


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
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer3 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer4 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer5 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer6 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4)
        self.conf_centerness_layer7 = conf_centernessLayer(in_channels=256, out_channels=self.num_classes + 1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        conf3, centerness3 = conf_centerness3.split([self.num_classes, 1], dim=1)
        loc4 = self.loc_layer4(p4)
        conf_centerness4 = self.conf_centerness_layer4(p4)
        conf4, centerness4 = conf_centerness4.split([self.num_classes, 1], dim=1)
        loc5 = self.loc_layer5(p5)
        conf_centerness5 = self.conf_centerness_layer5(p5)
        conf5, centerness5 = conf_centerness5.split([self.num_classes, 1], dim=1)
        loc6 = self.loc_layer6(p6)
        conf_centerness6 = self.conf_centerness_layer6(p6)
        conf6, centerness6 = conf_centerness6.split([self.num_classes, 1], dim=1)
        loc7 = self.loc_layer7(p7)
        conf_centerness7 = self.conf_centerness_layer7(p7)
        conf7, centerness7 = conf_centerness7.split([self.num_classes, 1], dim=1)
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        centernesses = torch.cat([centerness3.permute(0, 2, 3, 1).contiguous().view(centerness3.size(0), -1), centerness4.permute(0, 2, 3, 1).contiguous().view(centerness4.size(0), -1), centerness5.permute(0, 2, 3, 1).contiguous().view(centerness5.size(0), -1), centerness6.permute(0, 2, 3, 1).contiguous().view(centerness6.size(0), -1), centerness7.permute(0, 2, 3, 1).contiguous().view(centerness7.size(0), -1)], dim=1)
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
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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


class FisheyeMODNet(nn.Module):

    def __init__(self, groups=1, num_classes=2):
        super(FisheyeMODNet, self).__init__()
        layers = [4, 8, 4]
        self.stage1a = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage2a = self._make_layer(24, 120, groups, layers[0])
        self.stage1b = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1), nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage2b = self._make_layer(24, 120, groups, layers[0])
        self.stage3 = self._make_layer(240, 480, groups, layers[1])
        self.stage4 = self._make_layer(480, 960, groups, layers[2])
        self.adapt_conv3 = nn.Conv2d(960, num_classes, kernel_size=1)
        self.adapt_conv2 = nn.Conv2d(480, num_classes, kernel_size=1)
        self.adapt_conv1 = nn.Conv2d(240, num_classes, kernel_size=1)
        self.up_sampling3 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling2 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sampling1 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4)
        self.softmax = nn.Softmax(dim=1)
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num):
        layers = []
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2, groups=groups))
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
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
    return nn.Sequential(Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), Conv3x3ReLU(in_channels=in_channels, out_channels=in_channels), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))


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
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample6_relu = nn.ReLU6(inplace=True)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer3 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer4 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer5 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer6 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4)
        self.conf_layer7 = confLayer(in_channels=256, out_channels=self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
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
        self.lateral5 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.upsample4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downsample6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.downsample6_relu = nn.ReLU6(inplace=True)
        self.downsample5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.loc_layer3 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer3 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer4 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer4 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer5 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer5 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer6 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer6 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.loc_layer7 = locLayer(in_channels=256, out_channels=4 * num_anchores)
        self.conf_layer7 = confLayer(in_channels=256, out_channels=self.num_classes * num_anchores)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        locs = torch.cat([loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1), loc7.permute(0, 2, 3, 1).contiguous().view(loc7.size(0), -1)], dim=1)
        confs = torch.cat([conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1), conf7.permute(0, 2, 3, 1).contiguous().view(conf7.size(0), -1)], dim=1)
        out = locs, confs
        return out


def ConvTransBNReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class SSD(nn.Module):

    def __init__(self, phase='train', num_classes=21):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.detector1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=64, stride=1), Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1), Conv3x3BNReLU(in_channels=128, out_channels=128, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=128, out_channels=256, stride=1), Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1), Conv3x3BNReLU(in_channels=256, out_channels=256, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=256, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1))
        self.detector2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), Conv3x3BNReLU(in_channels=512, out_channels=512, stride=1), nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), ConvTransBNReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=2), Conv1x1BNReLU(in_channels=1024, out_channels=1024))
        self.detector3 = nn.Sequential(Conv1x1BNReLU(in_channels=1024, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512, stride=2))
        self.detector4 = nn.Sequential(Conv1x1BNReLU(in_channels=512, out_channels=128), Conv3x3BNReLU(in_channels=128, out_channels=256, stride=2))
        self.detector5 = nn.Sequential(Conv1x1BNReLU(in_channels=256, out_channels=128), Conv3x3ReLU(in_channels=128, out_channels=256, stride=1, padding=0))
        self.detector6 = nn.Sequential(Conv1x1BNReLU(in_channels=256, out_channels=128), Conv3x3ReLU(in_channels=128, out_channels=256, stride=1, padding=0))
        self.loc_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer1 = nn.Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer2 = nn.Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer3 = nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer4 = nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer5 = nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)
        self.loc_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        self.conf_layer6 = nn.Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, stride=1, padding=1)
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
        locs = torch.cat([loc1.permute(0, 2, 3, 1).contiguous().view(loc1.size(0), -1), loc2.permute(0, 2, 3, 1).contiguous().view(loc2.size(0), -1), loc3.permute(0, 2, 3, 1).contiguous().view(loc3.size(0), -1), loc4.permute(0, 2, 3, 1).contiguous().view(loc4.size(0), -1), loc5.permute(0, 2, 3, 1).contiguous().view(loc5.size(0), -1), loc6.permute(0, 2, 3, 1).contiguous().view(loc6.size(0), -1)], dim=1)
        confs = torch.cat([conf1.permute(0, 2, 3, 1).contiguous().view(conf1.size(0), -1), conf2.permute(0, 2, 3, 1).contiguous().view(conf2.size(0), -1), conf3.permute(0, 2, 3, 1).contiguous().view(conf3.size(0), -1), conf4.permute(0, 2, 3, 1).contiguous().view(conf4.size(0), -1), conf5.permute(0, 2, 3, 1).contiguous().view(conf5.size(0), -1), conf6.permute(0, 2, 3, 1).contiguous().view(conf6.size(0), -1)], dim=1)
        if self.phase == 'test':
            out = locs.view(locs.size(0), -1, 4), self.softmax(confs.view(confs.size(0), -1, self.num_classes))
        else:
            out = locs.view(locs.size(0), -1, 4), confs.view(confs.size(0), -1, self.num_classes)
        return out


class OSA_module(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, block_nums=5):
        super(OSA_module, self).__init__()
        self._layers = nn.ModuleList()
        self._layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1))
        for idx in range(block_nums - 1):
            self._layers.append(Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))
        self.conv1x1 = Conv1x1BNReLU(in_channels + mid_channels * block_nums, out_channels)

    def forward(self, x):
        outputs = []
        outputs.append(x)
        for _layer in self._layers:
            x = _layer(x)
            outputs.append(x)
        out = torch.cat(outputs, dim=1)
        out = self.conv1x1(out)
        return out


class eSE_Module(nn.Module):

    def __init__(self, channel, ratio=16):
        super(eSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, padding=0), nn.ReLU(inplace=True), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        z = self.excitation(y)
        return x * z.expand_as(x)


class OSAv2_module(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, block_nums=5):
        super(OSAv2_module, self).__init__()
        self._layers = nn.ModuleList()
        self._layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1))
        for idx in range(block_nums - 1):
            self._layers.append(Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))
        self.conv1x1 = Conv1x1BNReLU(in_channels + mid_channels * block_nums, out_channels)
        self.ese = eSE_Module(out_channels)
        self.pass_conv1x1 = Conv1x1BNReLU(in_channels, out_channels)

    def forward(self, x):
        residual = x
        outputs = []
        outputs.append(x)
        for _layer in self._layers:
            x = _layer(x)
            outputs.append(x)
        out = self.ese(self.conv1x1(torch.cat(outputs, dim=1)))
        return out + self.pass_conv1x1(residual)


class VoVNet(nn.Module):

    def __init__(self, planes, layers, num_classes=2):
        super(VoVNet, self).__init__()
        self.groups = 1
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=64, stride=2, groups=self.groups), Conv3x3BNReLU(in_channels=64, out_channels=64, stride=1, groups=self.groups), Conv3x3BNReLU(in_channels=64, out_channels=128, stride=1, groups=self.groups))
        self.stage2 = self._make_layer(planes[0][0], planes[0][1], planes[0][2], layers[0])
        self.stage3 = self._make_layer(planes[1][0], planes[1][1], planes[1][2], layers[1])
        self.stage4 = self._make_layer(planes[2][0], planes[2][1], planes[2][2], layers[2])
        self.stage5 = self._make_layer(planes[3][0], planes[3][1], planes[3][2], layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=planes[3][2], out_features=num_classes)

    def _make_layer(self, in_channels, mid_channels, out_channels, block_num):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for idx in range(block_num):
            layers.append(OSAv2_module(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.linear(x)
        return out


class SAG_Mask(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAG_Mask, self).__init__()
        mid_channels = in_channels
        self.fisrt_convs = nn.Sequential(Conv3x3BNReLU(in_channels=in_channels, out_channels=mid_channels, stride=1), Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1), Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1), Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=1))
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels * 2, out_channels=mid_channels, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.deconv = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.conv1x1 = Conv1x1BN(mid_channels, out_channels)

    def forward(self, x):
        residual = x = self.fisrt_convs(x)
        aggregate = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        sag = self.sigmoid(self.conv3x3(aggregate))
        sag_x = residual + sag * x
        out = self.conv1x1(self.deconv(sag_x))
        return out


class YOLO(nn.Module):

    def __init__(self):
        super(YOLO, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(kernel_size=2, stride=2), Conv3x3BNReLU(in_channels=64, out_channels=192), nn.MaxPool2d(kernel_size=2, stride=2), Conv1x1BNReLU(in_channels=192, out_channels=128), Conv3x3BNReLU(in_channels=128, out_channels=256), Conv1x1BNReLU(in_channels=256, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), nn.MaxPool2d(kernel_size=2, stride=2), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=256), Conv3x3BNReLU(in_channels=256, out_channels=512), Conv1x1BNReLU(in_channels=512, out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=1024), nn.MaxPool2d(kernel_size=2, stride=2), Conv1x1BNReLU(in_channels=1024, out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=1024), Conv1x1BNReLU(in_channels=1024, out_channels=512), Conv3x3BNReLU(in_channels=512, out_channels=1024), Conv3x3BNReLU(in_channels=1024, out_channels=1024), Conv3x3BNReLU(in_channels=1024, out_channels=1024, stride=2), Conv3x3BNReLU(in_channels=1024, out_channels=1024), Conv3x3BNReLU(in_channels=1024, out_channels=1024))
        self.classifier = nn.Sequential(nn.Linear(1024 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 1470))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def DWConv3x3BNReLU(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class EP(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(EP, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, mid_channels), DWConv3x3BNReLU(mid_channels, mid_channels, stride), Conv1x1BN(mid_channels, out_channels))
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class PEP(nn.Module):

    def __init__(self, in_channels, proj_channels, out_channels, stride, expansion_factor=6):
        super(PEP, self).__init__()
        self.stride = stride
        mid_channels = proj_channels * expansion_factor
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_channels, proj_channels), Conv1x1BNReLU(proj_channels, mid_channels), DWConv3x3BNReLU(mid_channels, mid_channels, stride), Conv1x1BN(mid_channels, out_channels))
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
        self.excitation = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channel // ratio, out_features=channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class YOLO_Nano(nn.Module):

    def __init__(self, out_channel=75):
        super(YOLO_Nano, self).__init__()
        self.stage1 = nn.Sequential(Conv3x3BNReLU(in_channels=3, out_channels=12, stride=1), Conv3x3BNReLU(in_channels=12, out_channels=24, stride=2), PEP(in_channels=24, proj_channels=7, out_channels=24, stride=1), EP(in_channels=24, out_channels=70, stride=2), PEP(in_channels=70, proj_channels=25, out_channels=70, stride=1), PEP(in_channels=70, proj_channels=24, out_channels=70, stride=1), EP(in_channels=70, out_channels=150, stride=2), PEP(in_channels=150, proj_channels=56, out_channels=150, stride=1), Conv1x1BNReLU(in_channels=150, out_channels=150), FCA(channel=150, ratio=8), PEP(in_channels=150, proj_channels=73, out_channels=150, stride=1), PEP(in_channels=150, proj_channels=71, out_channels=150, stride=1), PEP(in_channels=150, proj_channels=75, out_channels=150, stride=1))
        self.stage2 = nn.Sequential(EP(in_channels=150, out_channels=325, stride=2))
        self.stage3 = nn.Sequential(PEP(in_channels=325, proj_channels=132, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=124, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=141, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=140, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=137, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=135, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=133, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=140, out_channels=325, stride=1))
        self.stage4 = nn.Sequential(EP(in_channels=325, out_channels=545, stride=2), PEP(in_channels=545, proj_channels=276, out_channels=545, stride=1), Conv1x1BNReLU(in_channels=545, out_channels=230), EP(in_channels=230, out_channels=489, stride=1), PEP(in_channels=489, proj_channels=213, out_channels=469, stride=1), Conv1x1BNReLU(in_channels=469, out_channels=189))
        self.stage5 = nn.Sequential(Conv1x1BNReLU(in_channels=189, out_channels=105), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.stage6 = nn.Sequential(PEP(in_channels=105 + 325, proj_channels=113, out_channels=325, stride=1), PEP(in_channels=325, proj_channels=99, out_channels=207, stride=1), Conv1x1BNReLU(in_channels=207, out_channels=98))
        self.stage7 = nn.Sequential(Conv1x1BNReLU(in_channels=98, out_channels=47), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.out_stage1 = nn.Sequential(PEP(in_channels=150 + 47, proj_channels=58, out_channels=122, stride=1), PEP(in_channels=122, proj_channels=52, out_channels=87, stride=1), PEP(in_channels=87, proj_channels=47, out_channels=93, stride=1), Conv1x1BNReLU(in_channels=93, out_channels=out_channel))
        self.out_stage2 = nn.Sequential(EP(in_channels=98, out_channels=183, stride=1), Conv1x1BNReLU(in_channels=183, out_channels=out_channel))
        self.out_stage3 = nn.Sequential(EP(in_channels=189, out_channels=462, stride=1), Conv1x1BNReLU(in_channels=462, out_channels=out_channel))

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


class Residual(nn.Module):

    def __init__(self, nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels // 2
        self.conv1x1 = Conv1x1BNReLU(in_channels=nchannels, out_channels=mid_channels)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels, out_channels=nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x


class Darknet19(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.first_conv = Conv3x3BNReLU(in_channels=3, out_channels=32)
        self.block1 = self._make_layers(in_channels=32, out_channels=64, block_num=1)
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=2)
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=8)
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=8)
        self.block5 = self._make_layers(in_channels=512, out_channels=1024, block_num=4)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layers(self, in_channels, out_channels, block_num):
        _layers = []
        _layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=2))
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


class BatchNorm(nn.Module):

    def forward(self, x):
        return 2 * x - 1


class DynamicReLU_A(nn.Module):

    def __init__(self, channels, K=2, ratio=6):
        super(DynamicReLU_A, self).__init__()
        mid_channels = 2 * K
        self.K = K
        self.lambdas = torch.Tensor([1.0] * K + [0.5] * K).float()
        self.init_v = torch.Tensor([1.0] + [0.0] * (2 * K - 1)).float()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dynamic = nn.Sequential(nn.Linear(in_features=channels, out_features=channels // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channels // ratio, out_features=mid_channels), nn.Sigmoid(), BatchNorm())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.dynamic(y)
        relu_coefs = z.view(-1, 2 * self.K) * self.lambdas + self.init_v
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.K] + relu_coefs[:, self.K:]
        output = torch.max(output, dim=-1)[0].transpose(0, -1)
        return output


class DynamicReLU_B(nn.Module):

    def __init__(self, channels, K=2, ratio=6):
        super(DynamicReLU_B, self).__init__()
        mid_channels = 2 * K * channels
        self.K = K
        self.channels = channels
        self.lambdas = torch.Tensor([1.0] * K + [0.5] * K).float()
        self.init_v = torch.Tensor([1.0] + [0.0] * (2 * K - 1)).float()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.dynamic = nn.Sequential(nn.Linear(in_features=channels, out_features=channels // ratio), nn.ReLU(inplace=True), nn.Linear(in_features=channels // ratio, out_features=mid_channels), nn.Sigmoid(), BatchNorm())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = self.dynamic(y)
        relu_coefs = z.view(-1, self.channels, 2 * self.K) * self.lambdas + self.init_v
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.K] + relu_coefs[:, :, self.K:]
        output = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return output


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.depthwise1 = ConvBNReLU(in_channels, out_channels, 3, 1, 6, dilation=6)
        self.depthwise2 = ConvBNReLU(in_channels, out_channels, 3, 1, 12, dilation=12)
        self.depthwise3 = ConvBNReLU(in_channels, out_channels, 3, 1, 18, dilation=18)
        self.pointconv = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        x1 = self.depthwise1(x)
        x2 = self.depthwise2(x)
        x3 = self.depthwise3(x)
        x4 = self.pointconv(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class Decoder(nn.Module):

    def __init__(self, num_classes=2):
        super(Decoder, self).__init__()
        self.aspp = ASPP(320, 128)
        self.pconv1 = Conv1x1BN(128 * 4, 512)
        self.pconv2 = Conv1x1BN(512 + 32, 128)
        self.pconv3 = Conv1x1BN(128, num_classes)

    def forward(self, x, y):
        x = self.pconv1(self.aspp(x))
        x = F.interpolate(x, y.shape[2:], align_corners=True, mode='bilinear')
        x = torch.cat([x, y], dim=1)
        out = self.pconv3(self.pconv2(x))
        return out


class DeepLabv3Plus(nn.Module):

    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet('resnet50', None)
        self.aspp = ASPP(inplanes=self.backbone.final_out_channels)
        self.decoder = Decoder(self.num_classes, self.backbone.low_level_inplanes)

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        x, low_level_feat = self.backbone(imgs)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)
        return outputs


class InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


def AsymmetricConv(channels, stride, is_relu=False):
    return nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=[5, 1], stride=stride, padding=[2, 0], bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU(), nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=[1, 5], stride=stride, padding=[0, 2], bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU())


class RegularBottleneck(nn.Module):

    def __init__(self, in_places, places, stride=1, expansion=4, dilation=1, is_relu=False, asymmetric=False, p=0.01):
        super(RegularBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_places, mid_channels, False), AsymmetricConv(mid_channels, 1, is_relu) if asymmetric else Conv3x3BNReLU(mid_channels, mid_channels, 1, dilation, is_relu), Conv1x1BNReLU(mid_channels, places, is_relu), nn.Dropout2d(p=p))
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out += residual
        out = self.relu(out)
        return out


def Conv2x2BNReLU(in_channels, out_channels, is_relu=True):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU())


class DownBottleneck(nn.Module):

    def __init__(self, in_places, places, stride=2, expansion=4, is_relu=False, p=0.01):
        super(DownBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(Conv2x2BNReLU(in_places, mid_channels, is_relu), Conv3x3BNReLU(mid_channels, mid_channels, 1, 1, is_relu), Conv1x1BNReLU(mid_channels, places, is_relu), nn.Dropout2d(p=p))
        self.downsample = nn.MaxPool2d(3, stride=stride, padding=1, return_indices=True)
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x):
        out = self.bottleneck(x)
        residual, indices = self.downsample(x)
        n, ch, h, w = out.size()
        ch_res = residual.size()[1]
        padding = torch.zeros(n, ch - ch_res, h, w)
        residual = torch.cat((residual, padding), 1)
        out += residual
        out = self.relu(out)
        return out, indices


def TransposeConv3x3BNReLU(in_channels, out_channels, stride=2, is_relu=True):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True) if is_relu else nn.PReLU())


class UpBottleneck(nn.Module):

    def __init__(self, in_places, places, stride=2, expansion=4, is_relu=True, p=0.01):
        super(UpBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(Conv1x1BNReLU(in_places, mid_channels, is_relu), TransposeConv3x3BNReLU(mid_channels, mid_channels, stride, is_relu), Conv1x1BNReLU(mid_channels, places, is_relu), nn.Dropout2d(p=p))
        self.upsample_conv = Conv1x1BN(in_places, places)
        self.upsample_unpool = nn.MaxUnpool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x, indices):
        out = self.bottleneck(x)
        residual = self.upsample_conv(x)
        residual = self.upsample_unpool(residual, indices)
        out += residual
        out = self.relu(out)
        return out


class ENet(nn.Module):

    def __init__(self, num_classes):
        super(ENet, self).__init__()
        self.initialBlock = InitialBlock(3, 16)
        self.stage1_1 = DownBottleneck(16, 64, 2)
        self.stage1_2 = nn.Sequential(RegularBottleneck(64, 64, 1), RegularBottleneck(64, 64, 1), RegularBottleneck(64, 64, 1), RegularBottleneck(64, 64, 1))
        self.stage2_1 = DownBottleneck(64, 128, 2)
        self.stage2_2 = nn.Sequential(RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=2), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=4), RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=8), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=16))
        self.stage3 = nn.Sequential(RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=2), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=4), RegularBottleneck(128, 128, 1), RegularBottleneck(128, 128, 1, dilation=8), RegularBottleneck(128, 128, 1, asymmetric=True), RegularBottleneck(128, 128, 1, dilation=16))
        self.stage4_1 = UpBottleneck(128, 64, 2, is_relu=True)
        self.stage4_2 = nn.Sequential(RegularBottleneck(64, 64, 1, is_relu=True), RegularBottleneck(64, 64, 1, is_relu=True))
        self.stage5_1 = UpBottleneck(64, 16, 2, is_relu=True)
        self.stage5_2 = RegularBottleneck(16, 16, 1, is_relu=True)
        self.final_conv = nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initialBlock(x)
        x, indices1 = self.stage1_1(x)
        x = self.stage1_2(x)
        x, indices2 = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage3(x)
        x = self.stage4_1(x, indices2)
        x = self.stage4_2(x)
        x = self.stage5_1(x, indices1)
        x = self.stage5_2(x)
        out = self.final_conv(x)
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
        self.output5 = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, 4096, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout(), nn.Conv2d(4096, num_classes, kernel_size=1))
        self.up_pool3_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2)
        self.up_pool5_out = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2)

    def forward(self, x):
        _, _, w, h = x.size()
        x = self.padd(x)
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        output5 = self.up_pool5_out(self.output5(pool5))
        pool4_out = self.pool4_conv1x1(0.01 * pool4)
        output4 = self.up_pool4_out(pool4_out[:, :, 5:5 + output5.size()[2], 5:5 + output5.size()[3]] + output5)
        pool3_out = self.pool3_conv1x1(0.0001 * pool3)
        output3 = self.up_pool3_out(pool3_out[:, :, 9:9 + output4.size()[2], 9:9 + output4.size()[3]] + output4)
        out = self.up_pool3_out(output3)
        out = out[:, :, 31:31 + h, 31:31 + w].contiguous()
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        self.out = Conv3x3BNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


def DSConv(in_channels, out_channels, stride):
    return nn.Sequential(Conv3x3BN(in_channels=in_channels, out_channels=in_channels, stride=stride, groups=in_channels), Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels))


class LearningToDownsample(nn.Module):

    def __init__(self):
        super(LearningToDownsample, self).__init__()
        self.conv = Conv3x3BNReLU(in_channels=3, out_channels=32, stride=2)
        self.dsConv1 = DSConv(in_channels=32, out_channels=48, stride=2)
        self.dsConv2 = DSConv(in_channels=48, out_channels=64, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsConv1(x)
        out = self.dsConv2(x)
        return out


class GlobalFeatureExtractor(nn.Module):

    def __init__(self):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(inplanes=64, planes=64, blocks_num=3, stride=2)
        self.bottleneck2 = self._make_layer(inplanes=64, planes=96, blocks_num=3, stride=2)
        self.bottleneck3 = self._make_layer(inplanes=96, planes=128, blocks_num=3, stride=1)
        self.ppm = PyramidPooling(in_channels=128, out_channels=128)

    def _make_layer(self, inplanes, planes, blocks_num, stride=1):
        layers = []
        layers.append(InvertedResidual(inplanes, planes, stride))
        for i in range(1, blocks_num):
            layers.append(InvertedResidual(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        out = self.ppm(x)
        return out


class FeatureFusionModule(nn.Module):

    def __init__(self, num_classes=20):
        super(FeatureFusionModule, self).__init__()
        self.dsConv1 = nn.Sequential(DSConv(in_channels=128, out_channels=128, stride=1), Conv3x3BN(in_channels=128, out_channels=128, stride=1))
        self.dsConv2 = DSConv(in_channels=64, out_channels=128, stride=1)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.dsConv1(x)
        return x + self.dsConv2(y)


class Classifier(nn.Module):

    def __init__(self, num_classes=19):
        super(Classifier, self).__init__()
        self.dsConv = nn.Sequential(DSConv(in_channels=128, out_channels=128, stride=1), DSConv(in_channels=128, out_channels=128, stride=1))
        self.conv = Conv3x3BNReLU(in_channels=128, out_channels=num_classes, stride=1)

    def forward(self, x):
        x = self.dsConv(x)
        out = self.conv(x)
        return out


class FastSCNN(nn.Module):

    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        y = self.learning_to_downsample(x)
        x = self.global_feature_extractor(y)
        x = self.feature_fusion(x, y)
        out = self.classifier(x)
        return out


class CascadeFeatureFusion(nn.Module):

    def __init__(self, low_channels, high_channels, out_channels, num_classes):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = Conv3x3BNReLU(low_channels, out_channels, 1, dilation=2)
        self.conv_high = Conv3x3BNReLU(high_channels, out_channels, 1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_low_cls = nn.Conv2d(out_channels, num_classes, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        out = self.relu(x_low + x_high)
        x_low_cls = self.conv_low_cls(x_low)
        return out, x_low_cls


class Backbone(nn.Module):

    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(Backbone, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


class PyramidPoolingModule(nn.Module):

    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, x):
        feat = x
        height, width = x.shape[2:]
        for bin_size in self.pyramids:
            feat_x = F.adaptive_avg_pool2d(x, output_size=bin_size)
            feat_x = F.interpolate(feat_x, size=(height, width), mode='bilinear', align_corners=True)
            feat = feat + feat_x
        return feat


class ICNet(nn.Module):

    def __init__(self, num_classes):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.Sequential(Conv3x3BNReLU(3, 32, 2), Conv3x3BNReLU(32, 32, 2), Conv3x3BNReLU(32, 64, 2))
        self.backbone = Backbone()
        self.ppm = PyramidPoolingModule()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, num_classes)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, num_classes)
        self.conv_cls = nn.Conv2d(128, num_classes, 1, bias=False)

    def forward(self, x):
        x_sub1 = self.conv_sub1(x)
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        _, x_sub2, _, _ = self.backbone(x_sub2)
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        _, _, _, x_sub4 = self.backbone(x_sub4)
        x_sub4 = self.ppm(x_sub4)
        outs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outs.append(x_12_cls)
        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear')
        up_x2 = self.conv_cls(up_x2)
        outs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear')
        outs.append(up_x8)
        outs.reverse()
        return outs


def ConvReLU(in_channels, out_channels, kernel_size, stride, padding, dilation=[1, 1], groups=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False), nn.ReLU6(inplace=True))


class SS_nbt(nn.Module):

    def __init__(self, channels, dilation=1, groups=4):
        super(SS_nbt, self).__init__()
        mid_channels = channels // 2
        self.half_split = HalfSplit(dim=1)
        self.first_bottleneck = nn.Sequential(ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, padding=[1, 0]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, padding=[0, 1]), ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, dilation=[dilation, 1], padding=[dilation, 0]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, dilation=[1, dilation], padding=[0, dilation]))
        self.second_bottleneck = nn.Sequential(ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, padding=[0, 1]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, padding=[1, 0]), ConvReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[1, 3], stride=1, dilation=[1, dilation], padding=[0, dilation]), ConvBNReLU(in_channels=mid_channels, out_channels=mid_channels, kernel_size=[3, 1], stride=1, dilation=[dilation, 1], padding=[dilation, 0]))
        self.channelShuffle = ChannelShuffle(groups)

    def forward(self, x):
        x1, x2 = self.half_split(x)
        x1 = self.first_bottleneck(x1)
        x2 = self.second_bottleneck(x2)
        out = torch.cat([x1, x2], dim=1)
        return self.channelShuffle(out + x)


class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        mid_channels = out_channels - in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.maxpool(x)
        output = torch.cat([x1, x2], 1)
        return self.relu(self.bn(output))


class LWbottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(LWbottleneck, self).__init__()
        self.stride = stride
        self.pyramid_list = nn.ModuleList()
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[5, 1], stride=stride, padding=[2, 0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1, 5], stride=stride, padding=[0, 2]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[3, 1], stride=stride, padding=[1, 0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1, 3], stride=stride, padding=[0, 1]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[2, 1], stride=stride, padding=[1, 0]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=[1, 2], stride=stride, padding=[0, 1]))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=2, stride=stride, padding=1))
        self.pyramid_list.append(ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=stride, padding=1))
        self.shrink = Conv1x1BN(in_channels * 8, out_channels)

    def forward(self, x):
        b, c, w, h = x.shape
        if self.stride > 1:
            w, h = w // self.stride, h // self.stride
        outputs = []
        for pyconv in self.pyramid_list:
            pyconv_x = pyconv(x)
            if x.shape[2:] != pyconv_x.shape[2:]:
                pyconv_x = pyconv_x[:, :, :w, :h]
            outputs.append(pyconv_x)
        out = torch.cat(outputs, 1)
        return self.shrink(out)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.stage1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), Conv1x1BN(in_channels=32, out_channels=16))
        self.stage2 = nn.Sequential(LWbottleneck(in_channels=16, out_channels=24, stride=2), LWbottleneck(in_channels=24, out_channels=24, stride=1))
        self.stage3 = nn.Sequential(LWbottleneck(in_channels=24, out_channels=32, stride=2), LWbottleneck(in_channels=32, out_channels=32, stride=1))
        self.stage4 = nn.Sequential(LWbottleneck(in_channels=32, out_channels=32, stride=2))
        self.stage5 = nn.Sequential(LWbottleneck(in_channels=32, out_channels=64, stride=2), LWbottleneck(in_channels=64, out_channels=64, stride=1), LWbottleneck(in_channels=64, out_channels=64, stride=1), LWbottleneck(in_channels=64, out_channels=64, stride=1))
        self.conv1 = Conv1x1BN(in_channels=64, out_channels=320)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0)
        out1 = x = self.stage3(x)
        x = self.stage4(x)
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0)
        x = self.stage5(x)
        out2 = self.conv1(x)
        return out1, out2


class APN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(APN, self).__init__()
        self.conv1_1 = ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)
        self.conv2_1 = ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=2, padding=2)
        self.conv2_2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)
        self.conv3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2, padding=3), Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels))
        self.conv1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1), Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels))
        self.branch2 = Conv1x1BNReLU(in_channels=in_channels, out_channels=out_channels)
        self.branch3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        _, _, h, w = x.shape
        x1 = self.conv1_1(x)
        x2 = self.conv2_1(x1)
        x3 = self.conv3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        x2 = self.conv2_2(x2) + x3
        x2 = F.interpolate(x2, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x1 = self.conv1_2(x1) + x2
        out1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=True)
        return out1 * out2 + out3


class LEDnet(nn.Module):

    def __init__(self, num_classes=20):
        super(LEDnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(in_channels=128, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out


class LW_Network(nn.Module):

    def __init__(self, num_classes=2):
        super(LW_Network, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x1, x2 = self.encoder(x)
        out = self.decoder(x2, x1)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(Conv3x3BNReLU(in_channels, out_channels, stride=1), Conv3x3BNReLU(out_channels, out_channels, stride=1))

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, reverse=False):
        super().__init__()
        if reverse:
            self.triple_conv = nn.Sequential(Conv3x3BNReLU(in_channels, in_channels, stride=1), Conv3x3BNReLU(in_channels, in_channels, stride=1), Conv3x3BNReLU(in_channels, out_channels, stride=1))
        else:
            self.triple_conv = nn.Sequential(Conv3x3BNReLU(in_channels, out_channels, stride=1), Conv3x3BNReLU(out_channels, out_channels, stride=1), Conv3x3BNReLU(out_channels, out_channels, stride=1))

    def forward(self, x):
        return self.triple_conv(x)


class SegNet(nn.Module):
    """
        SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
        https://arxiv.org/pdf/1511.00561.pdf
    """

    def __init__(self, classes=19):
        super(SegNet, self).__init__()
        self.conv_down1 = DoubleConv(3, 64)
        self.conv_down2 = DoubleConv(64, 128)
        self.conv_down3 = TripleConv(128, 256)
        self.conv_down4 = TripleConv(256, 512)
        self.conv_down5 = TripleConv(512, 512)
        self.conv_up5 = TripleConv(512, 512, reverse=True)
        self.conv_up4 = TripleConv(512, 256, reverse=True)
        self.conv_up3 = TripleConv(256, 128, reverse=True)
        self.conv_up2 = DoubleConv(128, 64, reverse=True)
        self.conv_up1 = Conv3x3BNReLU(64, 64, stride=1)
        self.outconv = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv_down1(x)
        x1_size = x1.size()
        x1p, id1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.conv_down2(x1p)
        x2_size = x2.size()
        x2p, id2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.conv_down3(x2p)
        x3_size = x3.size()
        x3p, id3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.conv_down4(x3p)
        x4_size = x4.size()
        x4p, id4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.conv_down5(x4p)
        x5_size = x5.size()
        x5p, id5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x5d = self.conv_up5(x5d)
        x4d = F.max_unpool2d(x5d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x4d = self.conv_up4(x4d)
        x3d = F.max_unpool2d(x4d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x3d = self.conv_up3(x3d)
        x2d = F.max_unpool2d(x3d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x2d = self.conv_up2(x2d)
        x1d = F.max_unpool2d(x2d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x1d = self.conv_up1(x1d)
        out = self.outconv(x1d)
        return out


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.pool(self.double_conv(x))


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.reduce = Conv1x1BNReLU(in_channels, in_channels // 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(self.reduce(x1))
        _, channel1, height1, width1 = x1.size()
        _, channel2, height2, width2 = x2.size()
        diffY = height2 - height1
        diffX = width2 - width1
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        bilinear = True
        self.conv = DoubleConv(3, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024)
        self.up1 = UpConv(1024, 512, bilinear)
        self.up2 = UpConv(512, 256, bilinear)
        self.up3 = UpConv(256, 128, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xx = self.up1(x5, x4)
        xx = self.up2(xx, x3)
        xx = self.up3(xx, x2)
        xx = self.up4(xx, x1)
        outputs = self.outconv(xx)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AFF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (APN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (APNB,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (BatchNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelShuffle,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dCReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseBlock,
     lambda: ([], {'num_layers': 1, 'inplances': 4, 'growth_rate': 4, 'bn_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DynamicReLU_A,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DynamicReLU_B,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ECA_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EP,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EfficientNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (FCA,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FCN8s,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (FaceBoxes,
     lambda: ([], {'num_classes': 4, 'phase': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (FireModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GAM,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (HalfSplit,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionModules,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (LBwithGCBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LFFD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (LFFDBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LPN,
     lambda: ([], {'nJoints': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (LWbottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LossBranch,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MDConv,
     lambda: ([], {'nchannels': 4, 'kernel_sizes': [4, 4], 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MiddleBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NAM,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NonLocalBlock,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OSA_module,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OSAv2_module,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PEP,
     lambda: ([], {'in_channels': 4, 'proj_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PyramidPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PyramidPoolingModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNeXtBlock,
     lambda: ([], {'in_places': 64, 'places': 32}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     False),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SAG_Mask,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 512, 512])], {}),
     False),
    (SimpleBaseline,
     lambda: ([], {'nJoints': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SpatialAttentionModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialPyramidPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeAndExcite,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'se_kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TripleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TripletAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (YOLO_Nano,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (_ASPPModule,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_DenseLayer,
     lambda: ([], {'inplace': 4, 'growth_rate': 4, 'bn_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_TransitionLayer,
     lambda: ([], {'inplace': 4, 'plance': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (cSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (eSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (sSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (scSE_Module,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_shanglianlm0525_PyTorch_Networks(_paritybench_base):
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

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

    def test_060(self):
        self._check(*TESTCASES[60])

    def test_061(self):
        self._check(*TESTCASES[61])

    def test_062(self):
        self._check(*TESTCASES[62])

    def test_063(self):
        self._check(*TESTCASES[63])

    def test_064(self):
        self._check(*TESTCASES[64])

    def test_065(self):
        self._check(*TESTCASES[65])

    def test_066(self):
        self._check(*TESTCASES[66])

    def test_067(self):
        self._check(*TESTCASES[67])

    def test_068(self):
        self._check(*TESTCASES[68])

