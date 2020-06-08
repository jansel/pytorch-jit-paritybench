import sys
_module = sys.modules[__name__]
del sys
camvid_test = _module
camvid_train = _module
cityscapes_eval = _module
cityscapes_test = _module
cityscapes_train = _module
dataset = _module
camvid = _module
cityscapes = _module
BiSeNet_resnet = _module
BiSeNet_xception = _module
CGNet = _module
DFN = _module
DeepLabV3plus_resnet = _module
DeepLabV3plus_xception = _module
DenseASPP = _module
ENet = _module
ESPNet = _module
PSPNet = _module
SegNet = _module
model = _module
resnet = _module
xception = _module
colorize_mask = _module
compute_iou = _module
convert_state = _module
loss = _module
metric = _module
modelsize = _module
modeltools = _module
summary = _module
trainID2labelID = _module
vis_net = _module

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


import random


import numpy as np


import torch.nn as nn


from torch.utils import data


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


from torch import nn


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


from collections import OrderedDict


from torch.nn import BatchNorm2d as bn


from torch.nn import init


class SpatialPath(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = self.downsample_block(3, 64)
        self.layer2 = self.downsample_block(64, 128)
        self.layer3 = self.downsample_block(128, 256)

    def downsample_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(
            out_channels), nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class ARM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.pool = nn.AvgPool2d((input_h, input_w))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        feature_map = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.sigmoid(x)
        return x.expand_as(feature_map) * feature_map


class FFM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(channels, channels,
            kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels),
            nn.ReLU())
        self.pool = nn.AvgPool2d((input_h // 8, input_w // 8))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        feature = torch.cat([x1, x2], dim=1)
        feature = self.feature(feature)
        x = self.pool(feature)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return feature + x.expand_as(feature) * feature


model_urls = {'xception':
    'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class ContextPath(nn.Module):

    def __init__(self, input_h, input_w):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.backbone = resnet18(pretrained=True)
        self.x8_arm = ARM(input_h // 8, input_w // 8, 128)
        self.x16_arm = ARM(input_h // 16, input_w // 16, 256)
        self.x32_arm = ARM(input_h // 32, input_w // 32, 512)
        self.global_pool = nn.AvgPool2d((input_h // 32, input_w // 32))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        feature_x8 = self.backbone.layer2(x)
        feature_x16 = self.backbone.layer3(feature_x8)
        feature_x32 = self.backbone.layer4(feature_x16)
        center = self.global_pool(feature_x32)
        feature_x8 = self.x8_arm(feature_x8)
        feature_x16 = self.x16_arm(feature_x16)
        feature_x32 = self.x32_arm(feature_x32)
        up_feature_x32 = F.upsample(center, size=(self.input_h // 32, self.
            input_w // 32), mode='bilinear', align_corners=False)
        ensemble_feature_x32 = feature_x32 + up_feature_x32
        up_feature_x16 = F.upsample(ensemble_feature_x32, scale_factor=2,
            mode='bilinear', align_corners=False)
        ensemble_feature_x16 = torch.cat((feature_x16, up_feature_x16), dim=1)
        up_feature_x8 = F.upsample(ensemble_feature_x16, scale_factor=2,
            mode='bilinear', align_corners=False)
        ensemble_feature_x8 = torch.cat((feature_x8, up_feature_x8), dim=1)
        return ensemble_feature_x8


class BiSeNet_res18(nn.Module):

    def __init__(self, input_h, input_w, n_classes=19):
        super().__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(input_h, input_w)
        self.ffm = FFM(input_h, input_w, 1152)
        self.pred = nn.Conv2d(1152, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.spatial_path(x)
        x2 = self.context_path(x)
        feature = self.ffm(x1, x2)
        seg = self.pred(feature)
        return F.upsample(seg, x.size()[2:], mode='bilinear', align_corners
            =False)


class SpatialPath(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = self.downsample_block(3, 64)
        self.layer2 = self.downsample_block(64, 128)
        self.layer3 = self.downsample_block(128, 256)

    def downsample_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(
            out_channels), nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class ARM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.pool = nn.AvgPool2d((input_h, input_w))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        feature_map = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.sigmoid(x)
        return x.expand_as(feature_map) * feature_map


class FFM(nn.Module):

    def __init__(self, input_h, input_w, channels):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(channels, channels,
            kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels),
            nn.ReLU())
        self.pool = nn.AvgPool2d((input_h // 8, input_w // 8))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        feature = torch.cat([x1, x2], dim=1)
        feature = self.feature(feature)
        x = self.pool(feature)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return feature + x.expand_as(feature) * feature


def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """
    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model


class ContextPath(nn.Module):

    def __init__(self, input_h, input_w):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.backbone = xception()
        self.x8_arm = ARM(input_h // 8, input_w // 8, 256)
        self.x16_arm = ARM(input_h // 16, input_w // 16, 728)
        self.x32_arm = ARM(input_h // 32, input_w // 32, 2048)
        self.global_pool = nn.AvgPool2d((input_h // 32, input_w // 32))

    def forward(self, x):
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feature_x8 = self.backbone.layer3(x)
        feature_x16 = self.backbone.layer4(feature_x8)
        feature_x32 = self.backbone.layer5(feature_x16)
        center = self.global_pool(feature_x32)
        feature_x8 = self.x8_arm(feature_x8)
        feature_x16 = self.x16_arm(feature_x16)
        feature_x32 = self.x32_arm(feature_x32)
        up_feature_x32 = F.upsample(center, size=(self.input_h // 32, self.
            input_w // 32), mode='bilinear', align_corners=False)
        ensemble_feature_x32 = feature_x32 + up_feature_x32
        up_feature_x16 = F.upsample(ensemble_feature_x32, scale_factor=2,
            mode='bilinear', align_corners=False)
        ensemble_feature_x16 = torch.cat((feature_x16, up_feature_x16), dim=1)
        up_feature_x8 = F.upsample(ensemble_feature_x16, scale_factor=2,
            mode='bilinear', align_corners=False)
        ensemble_feature_x8 = torch.cat((feature_x8, up_feature_x8), dim=1)
        return ensemble_feature_x8


class BiSeNet_Xception34(nn.Module):

    def __init__(self, input_h, input_w, n_classes=19):
        super().__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath(input_h, input_w)
        self.ffm = FFM(input_h, input_w, 3288)
        self.pred = nn.Conv2d(3288, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.spatial_path(x)
        x2 = self.context_path(x)
        feature = self.ffm(x1, x2)
        seg = self.pred(feature)
        return F.upsample(seg, x.size()[2:], mode='bilinear', align_corners
            =False)


class ConvBNPReLU(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class ConvBN(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), groups=nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)
        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        self.bn = nn.BatchNorm2d(2 * nOut, eps=0.001)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)
        output = self.F_glo(joi_feat)
        return output


class ContextGuidedBlock(nn.Module):

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_prelu(joi_feat)
        output = self.F_glo(joi_feat)
        if self.add:
            output = input + output
        return output


class InputInjection(nn.Module):

    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class Context_Guided_Network(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """

    def __init__(self, classes=19, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(3, 32, 3, 2)
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)
        self.sample1 = InputInjection(1)
        self.sample2 = InputInjection(2)
        self.b1 = BNPReLU(32 + 3)
        self.level2_0 = ContextGuidedBlock_Down(32 + 3, 64, dilation_rate=2,
            reduction=8)
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2,
                reduction=8))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        self.level3_0 = ContextGuidedBlock_Down(128 + 3, 128, dilation_rate
            =4, reduction=16)
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4,
                reduction=16))
        self.bn_prelu_3 = BNPReLU(256)
        if dropout_flag:
            None
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(
                256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d') != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        out = F.upsample(classifier, input.size()[2:], mode='bilinear',
            align_corners=False)
        return out


class CAB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1,
            stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class RRB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


class DFN(nn.Module):

    def __init__(self, num_class=19):
        super(DFN, self).__init__()
        self.num_class = num_class
        self.resnet_features = resnet101(pretrained=False)
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.
            resnet_features.bn1, self.resnet_features.relu)
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.
            resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4
        self.out_conv = nn.Conv2d(2048, self.num_class, kernel_size=1, stride=1
            )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cab1 = CAB(self.num_class * 2, self.num_class)
        self.cab2 = CAB(self.num_class * 2, self.num_class)
        self.cab3 = CAB(self.num_class * 2, self.num_class)
        self.cab4 = CAB(self.num_class * 2, self.num_class)
        self.rrb_d_1 = RRB(256, self.num_class)
        self.rrb_d_2 = RRB(512, self.num_class)
        self.rrb_d_3 = RRB(1024, self.num_class)
        self.rrb_d_4 = RRB(2048, self.num_class)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.rrb_u_1 = RRB(self.num_class, self.num_class)
        self.rrb_u_2 = RRB(self.num_class, self.num_class)
        self.rrb_u_3 = RRB(self.num_class, self.num_class)
        self.rrb_u_4 = RRB(self.num_class, self.num_class)
        self.rrb_db_1 = RRB(256, self.num_class)
        self.rrb_db_2 = RRB(512, self.num_class)
        self.rrb_db_3 = RRB(1024, self.num_class)
        self.rrb_db_4 = RRB(2048, self.num_class)
        self.rrb_trans_1 = RRB(self.num_class, self.num_class)
        self.rrb_trans_2 = RRB(self.num_class, self.num_class)
        self.rrb_trans_3 = RRB(self.num_class, self.num_class)

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        res1 = self.rrb_db_1(f1)
        res1 = self.rrb_trans_1(res1 + self.upsample(self.rrb_db_2(f2)))
        res1 = self.rrb_trans_2(res1 + self.upsample_4(self.rrb_db_3(f3)))
        res1 = self.rrb_trans_3(res1 + self.upsample_8(self.rrb_db_4(f4)))
        res2 = self.out_conv(f4)
        res2 = self.global_pool(res2)
        res2 = nn.Upsample(size=f4.size()[2:], mode='nearest')(res2)
        f4 = self.rrb_d_4(f4)
        res2 = self.cab4([res2, f4])
        res2 = self.rrb_u_1(res2)
        f3 = self.rrb_d_3(f3)
        res2 = self.cab3([self.upsample(res2), f3])
        res2 = self.rrb_u_2(res2)
        f2 = self.rrb_d_2(f2)
        res2 = self.cab2([self.upsample(res2), f2])
        res2 = self.rrb_u_3(res2)
        f1 = self.rrb_d_1(f1)
        res2 = self.cab1([self.upsample(res2), f1])
        res2 = self.rrb_u_4(res2)
        return res1, res2

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides
            [0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=
            strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=
            strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=
            strides[3], rate=rates[3])
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0] *
            rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=
                blocks[i] * rate))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=
            kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os,
        pretrained=pretrained)
    return model


class DeepLabv3_plus(nn.Module):

    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=
        False, _print=True):
        if _print:
            None
            None
            None
            None
        super(DeepLabv3_plus, self).__init__()
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=
            pretrained)
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False), nn.BatchNorm2d(
            256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3,
            stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(
            ), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias
            =False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,
            n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)), int(
            math.ceil(input.size()[-1] / 4))), mode='bilinear',
            align_corners=True)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear',
            align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0,
        dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride,
            padding, dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, bias=False):
        super(SeparableConv2d_same, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0,
            dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.
            dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
        start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=
                False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1,
                dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1,
                dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1,
                dilation=dilation))
            rep.append(nn.BatchNorm2d(planes))
        if not start_with_relu:
            rep = rep[1:]
        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))
        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()
        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = 1, 2
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=
            True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
            start_with_relu=True, grow_first=True, is_last=True)
        self.block4 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=
            middle_block_rate, start_with_relu=True, grow_first=True)
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=
            exit_block_rates[0], start_with_relu=True, grow_first=False,
            is_last=True)
        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation
            =exit_block_rates[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation
            =exit_block_rates[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation
            =exit_block_rates[1])
        self.bn5 = nn.BatchNorm2d(2048)
        self.__init_weight()
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url(
            'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
            )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            None
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP_module(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=
            kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):

    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=
        False, _print=True):
        if _print:
            None
            None
            None
            None
        super(DeepLabv3_plus, self).__init__()
        self.xception_features = Xception(nInputChannels, os, pretrained)
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False), nn.BatchNorm2d(
            256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3,
            stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(
            ), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias
            =False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256,
            n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear',
            align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)), int(
            math.ceil(input.size()[-1] / 4))), mode='bilinear',
            align_corners=True)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear',
            align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """

    def __init__(self, n_class=19, output_stride=8):
        super(DenseASPP, self).__init__()
        bn_size = 4
        drop_rate = 0
        growth_rate = 32
        num_init_features = 64
        block_config = 6, 12, 48, 32
        dropout0 = 0.1
        dropout1 = 0.1
        d_feature0 = 480
        d_feature1 = 240
        feature_size = int(output_stride / 8)
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3,
            num_init_features, kernel_size=7, stride=2, padding=3, bias=
            False)), ('norm0', bn(num_init_features)), ('relu0', nn.ReLU(
            inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1))]))
        num_features = num_init_features
        block = _DenseBlock(num_layers=block_config[0], num_input_features=
            num_features, bn_size=bn_size, growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate
        trans = _Transition(num_input_features=num_features,
            num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2
        block = _DenseBlock(num_layers=block_config[1], num_input_features=
            num_features, bn_size=bn_size, growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate
        trans = _Transition(num_input_features=num_features,
            num_output_features=num_features // 2, stride=feature_size)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2
        block = _DenseBlock(num_layers=block_config[2], num_input_features=
            num_features, bn_size=bn_size, growth_rate=growth_rate,
            drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate
        trans = _Transition(num_input_features=num_features,
            num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2
        block = _DenseBlock(num_layers=block_config[3], num_input_features=
            num_features, bn_size=bn_size, growth_rate=growth_rate,
            drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate
        trans = _Transition(num_input_features=num_features,
            num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2
        self.features.add_module('norm5', bn(num_features))
        if feature_size > 1:
            self.features.add_module('upsample', nn.Upsample(scale_factor=2,
                mode='bilinear'))
        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=
            d_feature0, num2=d_feature1, dilation_rate=3, drop_out=dropout0,
            bn_start=False)
        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 *
            1, num1=d_feature0, num2=d_feature1, dilation_rate=6, drop_out=
            dropout0, bn_start=True)
        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 *
            2, num1=d_feature0, num2=d_feature1, dilation_rate=12, drop_out
            =dropout0, bn_start=True)
        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 *
            3, num1=d_feature0, num2=d_feature1, dilation_rate=18, drop_out
            =dropout0, bn_start=True)
        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 *
            4, num1=d_feature0, num2=d_feature1, dilation_rate=24, drop_out
            =dropout0, bn_start=True)
        num_features = num_features + 5 * d_feature1
        self.classification = nn.Sequential(nn.Dropout2d(p=dropout1), nn.
            Conv2d(in_channels=num_features, out_channels=n_class,
            kernel_size=1, padding=0), nn.Upsample(scale_factor=8, mode=
            'bilinear'))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        feature = self.features(_input)
        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)
        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)
        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)
        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)
        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)
        cls = self.classification(feature)
        return cls


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out,
        bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', bn(input_num, momentum=0.0003)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num,
            out_channels=num1, kernel_size=1)),
        self.add_module('norm2', bn(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=
            num2, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
            ),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.
                training)
        return feature


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
        dilation_rate=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', bn(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', bn(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate,
            growth_rate, kernel_size=3, stride=1, dilation=dilation_rate,
            padding=dilation_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
        drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                growth_rate, bn_size, drop_rate, dilation_rate=dilation_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features,
            num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))


class InitialBlock(nn.Module):
    """
    The initial block for Enet has 2 branches: The convolution branch and
    maxpool branch.
    The conv branch has 13 layers, while the maxpool branch gives 3 layers
    corresponding to the RBG channels.
    Both output layers are then concatenated to give an output of 16 layers.
    INPUTS:
    - input(Tensor): A 4D tensor of shape [batch_size, channel, height, width]
    """

    def __init__(self):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 13, (3, 3), stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(13, 0.001)
        self.prelu = nn.PReLU(13)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, input):
        output = torch.cat([self.prelu(self.batch_norm(self.conv(input))),
            self.pool(input)], 1)
        return output


class BottleNeck(nn.Module):
    """
    The bottle module has three different kinds of variants:
    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution which requires you to have a dilation factor.
    3. An asymetric convolution that has a decomposed filter size of 5x1 and
    1x5 separately.
    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape
    [batch_size, channel, height, widht].
    - output_channels(int): an integer indicating the output depth of the
    output convolutional block.
    - regularlizer_prob(float): the float p that represents the prob of
    dropping a layer for spatial dropout regularlization.
    - downsampling(bool): if True, a max-pool2D layer is added to downsample
    the spatial sizes.
    - upsampling(bool): if True, the upsampling bottleneck is activated but
    requires pooling indices to upsample.
    - dilated(bool): if True, then dilated convolution is done, but requires
    a dilation rate to be given.
    - dilation_rate(int): the dilation factor for performing atrous
    convolution/dilated convolution
    - asymmetric(bool): if True, then asymmetric convolution is done, and
    the only filter size used here is 5.
    - use_relu(bool): if True, then all the prelus become relus according to
    Enet author.
    """

    def __init__(self, input_channels=None, output_channels=None,
        regularlizer_prob=0.1, downsampling=False, upsampling=False,
        dilated=False, dilation_rate=None, asymmetric=False, use_relu=False):
        super(BottleNeck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.use_relu = use_relu
        internal = output_channels // 4
        input_stride = 2 if downsampling else 1
        conv1x1_1 = nn.Conv2d(input_channels, internal, input_stride,
            input_stride, bias=False)
        batch_norm1 = nn.BatchNorm2d(internal, 0.001)
        prelu1 = self._prelu(internal, use_relu)
        self.block1x1_1 = nn.Sequential(conv1x1_1, batch_norm1, prelu1)
        conv = None
        if downsampling:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
            conv = nn.Conv2d(internal, internal, 3, stride=1, padding=1)
        elif upsampling:
            spatial_conv = nn.Conv2d(input_channels, output_channels, 1,
                bias=False)
            batch_norm = nn.BatchNorm2d(output_channels, 0.001)
            self.conv_before_unpool = nn.Sequential(spatial_conv, batch_norm)
            self.unpool = nn.MaxUnpool2d(2)
            conv = nn.ConvTranspose2d(internal, internal, 3, stride=2,
                padding=1, output_padding=1)
        elif dilated:
            conv = nn.Conv2d(internal, internal, 3, padding=dilation_rate,
                dilation=dilation_rate)
        elif asymmetric:
            conv1 = nn.Conv2d(internal, internal, [5, 1], padding=(2, 0),
                bias=False)
            conv2 = nn.Conv2d(internal, internal, [1, 5], padding=(0, 2))
            conv = nn.Sequential(conv1, conv2)
        else:
            conv = nn.Conv2d(internal, internal, 3, padding=1)
        batch_norm = nn.BatchNorm2d(internal, 0.001)
        prelu = self._prelu(internal, use_relu)
        self.middle_block = nn.Sequential(conv, batch_norm, prelu)
        conv1x1_2 = nn.Conv2d(internal, output_channels, 1, bias=False)
        batch_norm2 = nn.BatchNorm2d(output_channels, 0.001)
        prelu2 = self._prelu(output_channels, use_relu)
        self.block1x1_2 = nn.Sequential(conv1x1_2, batch_norm2, prelu2)
        self.dropout = nn.Dropout2d(regularlizer_prob)

    def _prelu(self, channels, use_relu):
        return nn.PReLU(channels) if use_relu is False else nn.ReLU()

    def forward(self, input, pooling_indices=None):
        main = None
        input_shape = input.size()
        if self.downsampling:
            main, indices = self.pool(input)
            if self.output_channels != self.input_channels:
                pad = Variable(torch.Tensor(input_shape[0], self.
                    output_channels - self.input_channels, input_shape[2] //
                    2, input_shape[3] // 2).zero_(), requires_grad=False)
                if torch.cuda.is_available:
                    pad = pad
                main = torch.cat((main, pad), 1)
        elif self.upsampling:
            main = self.unpool(self.conv_before_unpool(input), pooling_indices)
        else:
            main = input
        other_net = nn.Sequential(self.block1x1_1, self.middle_block, self.
            block1x1_2)
        other = other_net(input)
        output = F.relu(main + other)
        if self.downsampling:
            return output, indices
        return output


ENCODER_LAYER_NAMES = ['initial', 'bottleneck_1_0', 'bottleneck_1_1',
    'bottleneck_1_2', 'bottleneck_1_3', 'bottleneck_1_4', 'bottleneck_2_0',
    'bottleneck_2_1', 'bottleneck_2_2', 'bottleneck_2_3', 'bottleneck_2_4',
    'bottleneck_2_5', 'bottleneck_2_6', 'bottleneck_2_7', 'bottleneck_2_8',
    'bottleneck_3_1', 'bottleneck_3_2', 'bottleneck_3_3', 'bottleneck_3_4',
    'bottleneck_3_5', 'bottleneck_3_6', 'bottleneck_3_7', 'bottleneck_3_8',
    'classifier']


_global_config['TRAIN'] = 4


class Encoder(nn.Module):

    def __init__(self, num_classes, only_encode=True):
        super(Encoder, self).__init__()
        self.state = only_encode
        layers = []
        layers.append(InitialBlock())
        layers.append(BottleNeck(16, 64, regularlizer_prob=0.01,
            downsampling=True))
        for i in range(4):
            layers.append(BottleNeck(64, 64, regularlizer_prob=0.01))
        layers.append(BottleNeck(64, 128, downsampling=True))
        for i in range(2):
            layers.append(BottleNeck(128, 128))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=2))
            layers.append(BottleNeck(128, 128, asymmetric=True))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=4))
            layers.append(BottleNeck(128, 128))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=8))
            layers.append(BottleNeck(128, 128, asymmetric=True))
            layers.append(BottleNeck(128, 128, dilated=True, dilation_rate=16))
        if only_encode:
            layers.append(nn.Conv2d(128, num_classes, 1))
        for layer, layer_name in zip(layers, ENCODER_LAYER_NAMES):
            super(Encoder, self).__setattr__(layer_name, layer)
        self.layers = layers

    def forward(self, input):
        pooling_stack = []
        output = input
        for layer in self.layers:
            if hasattr(layer, 'downsampling') and layer.downsampling:
                output, pooling_indices = layer(output)
                pooling_stack.append(pooling_indices)
            else:
                output = layer(output)
        if self.state:
            output = F.upsample(output, cfg.TRAIN.IMG_SIZE, None, 'bilinear')
        return output, pooling_stack


class Decoder(nn.Module):

    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        layers = []
        layers.append(BottleNeck(128, 64, upsampling=True, use_relu=True))
        layers.append(BottleNeck(64, 64, use_relu=True))
        layers.append(BottleNeck(64, 64, use_relu=True))
        layers.append(BottleNeck(64, 16, upsampling=True, use_relu=True))
        layers.append(BottleNeck(16, 16, use_relu=True))
        layers.append(nn.ConvTranspose2d(16, num_classes, 2, stride=2))
        self.layers = nn.ModuleList([layer for layer in layers])

    def forward(self, input, pooling_stack):
        output = input
        for layer in self.layers:
            if hasattr(layer, 'upsampling') and layer.upsampling:
                pooling_indices = pooling_stack.pop()
                output = layer(output, pooling_indices)
            else:
                output = layer(output)
        return output


class ENet(nn.Module):

    def __init__(self, n_classes=19, only_encode=False):
        super(ENet, self).__init__()
        self.state = only_encode
        self.encoder = Encoder(n_classes, only_encode=only_encode)
        self.decoder = Decoder(n_classes)

    def forward(self, input):
        output, pooling_stack = self.encoder(input)
        if not self.state:
            output = self.decoder(output, pooling_stack)
        return output


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution, which can maintain feature map size
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride,
            padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):

    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        output = self.bn(combine)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3, for input reinforcement, which establishes a direct link between 
    the input image and encoding stage, improving the flow of information.    
    """

    def __init__(self, samplingTimes):
        """
        :param samplingTimes: The rate at which you want to down-sample the image
        """
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        """
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        """
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes=20, p=5, q=3):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)
        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.b3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        out = F.upsample(classifier, input.size()[2:], mode='bilinear')
        return out


class ESPNet(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, classes=19, p=2, q=3, encoderFile=None):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        """
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            None
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=0.001)
        self.conv = CBR(19 + classes, classes, 3, 1)
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2,
            stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2 * classes),
            DilatedParllelResidualBlockB(2 * classes, classes, add=False))
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2,
            stride=2, padding=0, output_padding=0, bias=False), BR(classes))
        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2,
            padding=0, output_padding=0, bias=False)

    def forward(self, input):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)
        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)
        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.modules[7](output1_cat)
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.modules[9](torch.cat([output2_0, output2], 1))
        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))
        output1_C = self.level3_C(output1_cat)
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C,
            output2_c], 1)))
        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))
        classifier = self.classifier(concat_features)
        return classifier


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


affine_par = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes,
                kernel_size=3, stride=1, padding=padding, dilation=dilation,
                bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class Residual_Covolution(nn.Module):

    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding
            =12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1,
            padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1,
            padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding
            =0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg


class PSPModule(nn.Module):
    """Ref: Pyramid Scene Parsing Network,CVPR2017, http://arxiv.org/abs/1612.01105 """

    def __init__(self, inChannel, midReduction=4, outChannel=512, sizes=(1,
        2, 3, 6)):
        super(PSPModule, self).__init__()
        self.midChannel = int(inChannel / midReduction)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(inChannel, self.
            midChannel, size) for size in sizes])
        self.bottleneck = nn.Conv2d(inChannel + self.midChannel * 4,
            outChannel, kernel_size=3)
        self.bn = nn.BatchNorm2d(outChannel)
        self.prelu = nn.PReLU()

    def _make_stage(self, inChannel, midChannel, size):
        pooling = nn.AdaptiveAvgPool2d(output_size=(size, size))
        Conv = nn.Conv2d(inChannel, midChannel, kernel_size=1, bias=False)
        return nn.Sequential(pooling, Conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        mulBranches = [F.upsample(input=stage(feats), size=(h, w), mode=
            'bilinear') for stage in self.stages] + [feats]
        out = self.bottleneck(torch.cat((mulBranches[0], mulBranches[1],
            mulBranches[2], mulBranches[3], feats), 1))
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, midReduction=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=4)
        self.pspmodule = PSPModule(inChannel=512 * block.expansion,
            midReduction=midReduction, outChannel=512, sizes=(1, 2, 3, 6))
        self.spatial_drop = nn.Dropout2d(p=0.1)
        self.main_classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation == 2 or dilation == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pspmodule(x)
        x = self.spatial_drop(x)
        x = self.main_classifier(x)
        x = F.upsample(x, input_size, mode='bilinear')
        return x


class SegNet(nn.Module):

    def __init__(self, input_nbr=3, label_nbr=19):
        super(SegNet, self).__init__()
        batchNorm_momentum = 0.1
        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2,
            return_indices=True)
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2,
            return_indices=True)
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2,
            return_indices=True)
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2,
            return_indices=True)
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2,
            return_indices=True)
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()
        th = torch.load(model_path).state_dict()
        self.load_state_dict(th)


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, num_classes=1000, dilated=True,
        norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                downsample=downsample, previous_dilation=dilation,
                norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                previous_dilation=dilation, norm_layer=norm_layer))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1,
        start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=
                strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1,
                padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x
