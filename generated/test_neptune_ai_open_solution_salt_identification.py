import sys
_module = sys.modules[__name__]
del sys
common_blocks = _module
architectures = _module
base = _module
deprecated = _module
encoders = _module
large_kernel_matters = _module
misc = _module
models_with_depth = _module
pspnet = _module
unet = _module
augmentation = _module
callbacks = _module
loaders = _module
lovash_losses = _module
lovasz_losses = _module
metrics = _module
models = _module
pipelines = _module
postprocessing = _module
unet_models = _module
utils = _module
empty_vs_non_empty = _module
main = _module
prepare_metadata = _module

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


import numpy as np


from torch import nn


from torch.nn import functional as F


import torch


from torchvision import models


import torchvision


from functools import partial


from torch.autograd import Variable


from torch.optim.lr_scheduler import ExponentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.metrics import roc_auc_score


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from itertools import product


from scipy.stats import gmean


import torch.nn.functional as F


import torch.nn as nn


import torch.optim as optim


import logging


import random


import time


from itertools import chain


from collections import Iterable


from sklearn.model_selection import BaseCrossValidator


class Conv2dBnRelu(nn.Module):
    PADDING_METHODS = {'replication': nn.ReplicationPad2d, 'reflection': nn.ReflectionPad2d, 'zero': nn.ZeroPad2d}

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), use_relu=True, use_batch_norm=True, use_padding=True, padding_method='replication'):
        super().__init__()
        self.use_relu = use_relu
        self.use_batch_norm = use_batch_norm
        self.use_padding = use_padding
        self.kernel_w = kernel_size[0]
        self.kernel_h = kernel_size[1]
        self.padding_w = kernel_size[0] - 1
        self.padding_h = kernel_size[1] - 1
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.padding = Conv2dBnRelu.PADDING_METHODS[padding_method](padding=(0, self.padding_h, self.padding_w, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        if self.use_padding:
            x = self.padding(x)
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class DeconvConv2dBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, use_relu=True, use_batch_norm=True):
        super().__init__()
        self.use_relu = use_relu
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.deconv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class NoOperation(nn.Module):

    def forward(self, x):
        return x


class ChannelSELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialSELayer(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc(x)
        x = self.sigmoid(x)
        return module_input * x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, middle_channels)
        self.conv2 = Conv2dBnRelu(middle_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(inplace=True)
        self.channel_se = ChannelSELayer(out_channels, reduction=16)
        self.spatial_se = SpatialSELayer(out_channels)

    def forward(self, x, e=None):
        x = self.upsample(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        channel_se = self.channel_se(x)
        spatial_se = self.spatial_se(x)
        x = self.relu(channel_se + spatial_se)
        return x


class DepthChannelExcitation(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, channels), nn.Sigmoid())

    def forward(self, x, d=None):
        b, c, _, _ = x.size()
        y = self.fc(d).view(b, c, 1, 1)
        return x * y


class DepthSpatialExcitation(nn.Module):

    def __init__(self, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.grid_size_sqrt = int(np.sqrt(grid_size))
        self.fc = nn.Sequential(nn.Linear(1, grid_size), nn.Sigmoid())

    def forward(self, x, d=None):
        b, _, h, w = x.size()
        y = self.fc(d).view(b, 1, self.grid_size_sqrt, self.grid_size_sqrt)
        scale_factor = h // self.grid_size_sqrt
        y = F.upsample(y, scale_factor=scale_factor, mode='bilinear')
        return x * y


class GlobalConvolutionalNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, use_relu=False):
        super().__init__()
        self.conv1 = nn.Sequential(Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1), use_relu=use_relu, use_padding=True), Conv2dBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size), use_relu=use_relu, use_padding=True))
        self.conv2 = nn.Sequential(Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size), use_relu=use_relu, use_padding=True), Conv2dBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, 1), use_relu=use_relu, use_padding=True))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        return conv1 + conv2


class BoundaryRefinement(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), use_relu=True, use_padding=True), Conv2dBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), use_relu=False, use_padding=True))

    def forward(self, x):
        conv = self.conv(x)
        return x + conv


class ResNetEncoders(nn.Module):

    def __init__(self, encoder_depth, pretrained=False, pool0=False):
        super().__init__()
        if encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
        elif encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')
        if pool0:
            self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool)
        else:
            self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        return encoder2, encoder3, encoder4, encoder5


class ConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DecoderBlockV2(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.is_deconv = is_deconv
        self.deconv = nn.Sequential(ConvBnRelu(in_channels, middle_channels), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.upsample = nn.Sequential(ConvBnRelu(in_channels, out_channels), nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.input_adjust = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4
        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)


class UNetResNetWithDepth(nn.Module):

    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn
        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained)
        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')
        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr), Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2), nn.AvgPool2d(kernel_size=2, stride=2))
        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2, bottom_channel_nr, bottom_channel_nr // 8)
        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8, bottom_channel_nr // 2, bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8, bottom_channel_nr // 4, bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8, bottom_channel_nr // 8, bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8, bottom_channel_nr // 16, bottom_channel_nr // 8)
        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
            self.depth_channel_excitation = DepthChannelExcitation(5 * bottom_channel_nr // 8)
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
            self.depth_channel_excitation = DepthChannelExcitation(bottom_channel_nr // 8)

    def forward(self, x, d=None):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)
        center = self.center(encoder5)
        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)
        if self.use_hypercolumn:
            dec1 = torch.cat([dec1, F.upsample(dec2, scale_factor=2, mode='bilinear'), F.upsample(dec3, scale_factor=4, mode='bilinear'), F.upsample(dec4, scale_factor=8, mode='bilinear'), F.upsample(dec5, scale_factor=16, mode='bilinear')], 1)
        depth_channel_excitation = self.depth_channel_excitation(dec1, d)
        return self.final(depth_channel_excitation)


class LargeKernelMatters(nn.Module):
    """PyTorch LKM model using ResNet(18, 34, 50, 101 or 152) encoder.

        https://arxiv.org/pdf/1703.02719.pdf
    """

    def __init__(self, encoder_depth, num_classes, kernel_size=9, internal_channels=21, use_relu=False, pretrained=False, dropout_2d=0.0, pool0=False):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')
        self.gcn2 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr // 8, out_channels=internal_channels, kernel_size=kernel_size, use_relu=use_relu)
        self.gcn3 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr // 4, out_channels=internal_channels, kernel_size=kernel_size, use_relu=use_relu)
        self.gcn4 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr // 2, out_channels=internal_channels, kernel_size=kernel_size, use_relu=use_relu)
        self.gcn5 = GlobalConvolutionalNetwork(in_channels=bottom_channel_nr, out_channels=internal_channels, kernel_size=kernel_size, use_relu=use_relu)
        self.enc_br2 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.enc_br3 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.enc_br4 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.enc_br5 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.dec_br1 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.dec_br2 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.dec_br3 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.dec_br4 = BoundaryRefinement(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3)
        self.deconv5 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv4 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv3 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.deconv2 = DeconvConv2dBnRelu(in_channels=internal_channels, out_channels=internal_channels)
        self.final = nn.Conv2d(internal_channels, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)
        gcn2 = self.enc_br2(self.gcn2(encoder2))
        gcn3 = self.enc_br3(self.gcn3(encoder3))
        gcn4 = self.enc_br4(self.gcn4(encoder4))
        gcn5 = self.enc_br5(self.gcn5(encoder5))
        decoder5 = self.deconv5(gcn5)
        decoder4 = self.deconv4(self.dec_br4(decoder5 + gcn4))
        decoder3 = self.deconv3(self.dec_br3(decoder4 + gcn3))
        decoder2 = self.dec_br1(self.deconv2(self.dec_br2(decoder3 + gcn2)))
        return self.final(decoder2)


class StackingUnet(nn.Module):

    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)), Conv2dBnRelu(filter_nr, filter_nr * 2, kernel_size=(3, 3)))
        self.encoder2 = nn.Sequential(Conv2dBnRelu(filter_nr * 2, filter_nr * 2, kernel_size=(3, 3)), Conv2dBnRelu(filter_nr * 2, filter_nr * 4, kernel_size=(3, 3)), nn.MaxPool2d(2))
        self.encoder3 = nn.Sequential(Conv2dBnRelu(filter_nr * 4, filter_nr * 4, kernel_size=(3, 3)), Conv2dBnRelu(filter_nr * 4, filter_nr * 8, kernel_size=(3, 3)), nn.MaxPool2d(2))
        self.encoder4 = nn.Sequential(Conv2dBnRelu(filter_nr * 8, filter_nr * 8, kernel_size=(3, 3)), Conv2dBnRelu(filter_nr * 8, filter_nr * 16, kernel_size=(3, 3)), nn.MaxPool2d(2))
        self.center = nn.Sequential(Conv2dBnRelu(filter_nr * 16, filter_nr * 16), Conv2dBnRelu(filter_nr * 16, filter_nr * 8), nn.MaxPool2d(2))
        self.dec4 = DecoderBlock(filter_nr * 16 + filter_nr * 8, filter_nr * 16, filter_nr * 8)
        self.dec3 = DecoderBlock(filter_nr * 8 + filter_nr * 8, filter_nr * 8, filter_nr * 8)
        self.dec2 = DecoderBlock(filter_nr * 4 + filter_nr * 8, filter_nr * 8, filter_nr * 8)
        self.dec1 = DecoderBlock(filter_nr * 8, filter_nr * 8, filter_nr * 8)
        self.final = nn.Sequential(Conv2dBnRelu(filter_nr * 8, filter_nr * 4), nn.Conv2d(filter_nr * 4, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        conv = self.conv(x)
        encoder2 = self.encoder2(conv)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder4 = F.dropout2d(encoder4, p=self.dropout_2d)
        center = self.center(encoder4)
        dec4 = self.dec4(center, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)
        return self.final(dec1)


class StackingFCN(nn.Module):

    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)))
        self.final = nn.Sequential(nn.Conv2d(filter_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        x = F.dropout2d(self.conv(x), p=self.dropout_2d)
        return self.final(x)


class StackingFCNWithDepth(nn.Module):

    def __init__(self, input_model_nr, num_classes, filter_nr=32, dropout_2d=0.0):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.conv = nn.Sequential(Conv2dBnRelu(input_model_nr, filter_nr, kernel_size=(3, 3)))
        self.depth_channel_excitation = DepthChannelExcitation(filter_nr)
        self.final = nn.Sequential(nn.Conv2d(filter_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x, d=None):
        x = F.dropout2d(self.conv(x), p=self.dropout_2d)
        x = self.depth_channel_excitation(x, d)
        return self.final(x)


class EmptinessClassifier(nn.Module):

    def __init__(self, num_classes=2, encoder_depth=18, pretrained=True):
        super().__init__()
        if encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4
        self.classifier = nn.Sequential(nn.AvgPool2d(8), nn.Conv2d(bottom_channel_nr, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        pred = self.classifier(encoder5)
        return pred


class SeResNetEncoders(nn.Module):

    def __init__(self, encoder_depth, pretrained='imagenet', pool0=False):
        super().__init__()
        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained=pretrained)
        else:
            raise NotImplementedError('only 50, 101, 152 version of Resnet are implemented')
        if pool0:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1, self.encoder.layer0.pool0)
        else:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1)
        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        return encoder2, encoder3, encoder4, encoder5


class SeResNetXtEncoders(nn.Module):

    def __init__(self, encoder_depth, pretrained='imagenet', pool0=False):
        super().__init__()
        if encoder_depth == 50:
            self.encoder = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=pretrained)
        else:
            raise NotImplementedError('only 50, 101 version of Resnet are implemented')
        if pool0:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1, self.encoder.layer0.pool0)
        else:
            self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1)
        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        return encoder2, encoder3, encoder4, encoder5


class DenseNetEncoders(nn.Module):

    def __init__(self, encoder_depth, pretrained='imagenet', pool0=False):
        super().__init__()
        if encoder_depth == 121:
            self.encoder = pretrainedmodels.__dict__['densenet121'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 161:
            self.encoder = pretrainedmodels.__dict__['densenet161'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 169:
            self.encoder = pretrainedmodels.__dict__['densenet169'](num_classes=1000, pretrained=pretrained)
        elif encoder_depth == 201:
            self.encoder = pretrainedmodels.__dict__['densenet201'](num_classes=1000, pretrained=pretrained)
        else:
            raise NotImplementedError('only 121, 161, 169, 201 version of Densenet are implemented')
        if pool0:
            self.conv1 = nn.Sequential(self.encoder.features.conv0, self.encoder.features.norm0, self.encoder.features.relu0, self.encoder.features.pool0)
        else:
            self.conv1 = nn.Sequential(self.encoder.features.conv0, self.encoder.features.norm0, self.encoder.features.relu0)
        self.encoder2 = self.encoder.features.denseblock1
        self.transition1 = self.encoder.features.transition1
        self.encoder3 = self.encoder.features.denseblock2
        self.transition2 = self.encoder.features.transition2
        self.encoder4 = self.encoder.features.denseblock3
        self.transition3 = self.encoder.features.transition3
        self.encoder5 = self.encoder.features.denseblock4

    def forward(self, x):
        conv1 = self.conv1(x)
        encoder2 = self.encoder2(conv1)
        transition1 = self.transition1(encoder2)
        encoder3 = self.encoder3(transition1)
        transition2 = self.transition2(encoder3)
        encoder4 = self.encoder4(transition2)
        transition3 = self.transition3(encoder4)
        encoder5 = self.encoder5(transition3)
        return encoder2, encoder3, encoder4, encoder5


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.PReLU())

    def forward(self, x):
        p = F.upsample(input=x, scale_factor=2, mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):

    def __init__(self, encoder_depth, num_classes=2, sizes=(1, 2, 3, 6), deep_features_size=1024, dropout_2d=0.2, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn
        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')
        self.psp = PSPModule(bottom_channel_nr, deep_features_size, sizes)
        self.up4 = PSPUpsample(deep_features_size, deep_features_size // 2)
        self.up3 = PSPUpsample(deep_features_size // 2, deep_features_size // 4)
        self.up2 = PSPUpsample(deep_features_size // 4, deep_features_size // 8)
        self.up1 = PSPUpsample(deep_features_size // 8, deep_features_size // 16)
        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(15 * bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)
        psp = self.psp(encoder5)
        up4 = self.up4(psp)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        if self.use_hypercolumn:
            hypercolumn = torch.cat([up1, F.upsample(up2, scale_factor=2, mode='bilinear'), F.upsample(up3, scale_factor=4, mode='bilinear'), F.upsample(up4, scale_factor=8, mode='bilinear')], 1)
            drop = F.dropout2d(hypercolumn, p=self.dropout_2d)
        else:
            drop = F.dropout2d(up4, p=self.dropout_2d)
        return self.final(drop)


class UNetSeResNet(nn.Module):

    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn
        self.encoders = SeResNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        bottom_channel_nr = 2048
        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr), Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2), nn.AvgPool2d(kernel_size=2, stride=2))
        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2, bottom_channel_nr, bottom_channel_nr // 8)
        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8, bottom_channel_nr // 2, bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8, bottom_channel_nr // 4, bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8, bottom_channel_nr // 8, bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8, bottom_channel_nr // 16, bottom_channel_nr // 8)
        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)
        center = self.center(encoder5)
        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)
        if self.use_hypercolumn:
            dec1 = torch.cat([dec1, F.upsample(dec2, scale_factor=2, mode='bilinear'), F.upsample(dec3, scale_factor=4, mode='bilinear'), F.upsample(dec4, scale_factor=8, mode='bilinear'), F.upsample(dec5, scale_factor=16, mode='bilinear')], 1)
        return self.final(dec1)


class UNetSeResNetXt(nn.Module):

    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn
        self.encoders = SeResNetXtEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        bottom_channel_nr = 2048
        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr), Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2), nn.AvgPool2d(kernel_size=2, stride=2))
        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2, bottom_channel_nr, bottom_channel_nr // 8)
        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8, bottom_channel_nr // 2, bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8, bottom_channel_nr // 4, bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8, bottom_channel_nr // 8, bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8, bottom_channel_nr // 16, bottom_channel_nr // 8)
        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8), nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)
        center = self.center(encoder5)
        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)
        if self.use_hypercolumn:
            dec1 = torch.cat([dec1, F.upsample(dec2, scale_factor=2, mode='bilinear'), F.upsample(dec3, scale_factor=4, mode='bilinear'), F.upsample(dec4, scale_factor=8, mode='bilinear'), F.upsample(dec5, scale_factor=16, mode='bilinear')], 1)
        return self.final(dec1)


class UNetDenseNet(nn.Module):

    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn
        self.encoders = DenseNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        if encoder_depth == 121:
            encoder_channel_nr = [256, 512, 1024, 1024]
        elif encoder_depth == 161:
            encoder_channel_nr = [384, 768, 2112, 2208]
        elif encoder_depth == 169:
            encoder_channel_nr = [256, 512, 1280, 1664]
        elif encoder_depth == 201:
            encoder_channel_nr = [256, 512, 1792, 1920]
        else:
            raise NotImplementedError('only 121, 161, 169, 201 version of Densenet are implemented')
        self.center = nn.Sequential(Conv2dBnRelu(encoder_channel_nr[3], encoder_channel_nr[3]), Conv2dBnRelu(encoder_channel_nr[3], encoder_channel_nr[2]), nn.AvgPool2d(kernel_size=2, stride=2))
        self.dec5 = DecoderBlock(encoder_channel_nr[3] + encoder_channel_nr[2], encoder_channel_nr[3], encoder_channel_nr[3] // 8)
        self.dec4 = DecoderBlock(encoder_channel_nr[2] + encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 2, encoder_channel_nr[3] // 8)
        self.dec3 = DecoderBlock(encoder_channel_nr[1] + encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 4, encoder_channel_nr[3] // 8)
        self.dec2 = DecoderBlock(encoder_channel_nr[0] + encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 8)
        self.dec1 = DecoderBlock(encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 16, encoder_channel_nr[3] // 8)
        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 8), nn.Conv2d(encoder_channel_nr[3] // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 8), nn.Conv2d(encoder_channel_nr[3] // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)
        center = self.center(encoder5)
        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)
        if self.use_hypercolumn:
            dec1 = torch.cat([dec1, F.upsample(dec2, scale_factor=2, mode='bilinear'), F.upsample(dec3, scale_factor=4, mode='bilinear'), F.upsample(dec4, scale_factor=8, mode='bilinear'), F.upsample(dec5, scale_factor=16, mode='bilinear')], 1)
        return self.final(dec1)


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class DiceLoss(nn.Module):

    def __init__(self, smooth=0, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class DecoderBlockV1(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(ConvBnRelu(in_channels, middle_channels), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class SaltUNet(nn.Module):

    def __init__(self, num_classes, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.input_adjust = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.conv1 = list(self.encoder.layer1.children())[1]
        self.conv2 = list(self.encoder.layer1.children())[2]
        self.conv3 = list(self.encoder.layer2.children())[0]
        self.conv4 = list(self.encoder.layer2.children())[1]
        self.dec3 = DecoderBlockV2(256, 512, 256, is_deconv)
        self.dec2 = ConvBnRelu(256 + 64, 256)
        self.dec1 = DecoderBlockV2(256 + 64, (256 + 64) * 2, 256, is_deconv)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec3 = self.dec3(torch.cat([center, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)


class SaltLinkNet(nn.Module):

    def __init__(self, num_classes, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.input_adjust = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.conv1_1 = list(self.encoder.layer1.children())[1]
        self.conv1_2 = list(self.encoder.layer1.children())[2]
        self.conv2_0 = list(self.encoder.layer2.children())[0]
        self.conv2_1 = list(self.encoder.layer2.children())[1]
        self.conv2_2 = list(self.encoder.layer2.children())[2]
        self.conv2_3 = list(self.encoder.layer2.children())[3]
        self.dec2 = DecoderBlockV2(128, 256, 256, is_deconv=is_deconv)
        self.dec1 = DecoderBlockV2(256 + 64, 512, 256, is_deconv=is_deconv)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1_1 = self.conv1_1(input_adjust)
        conv1_2 = self.conv1_2(conv1_1)
        conv2_0 = self.conv2_0(conv1_2)
        conv2_1 = self.conv2_1(conv2_0)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_3 = self.conv2_3(conv2_2)
        conv1_sum = conv1_1 + conv1_2
        conv2_sum = conv2_0 + conv2_1 + conv2_2 + conv2_3
        dec2 = self.dec2(conv2_sum)
        dec1 = self.dec1(torch.cat([dec2, conv1_sum], 1))
        return self.final(F.dropout2d(dec1, p=self.dropout_2d))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BoundaryRefinement,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelSELayer,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (Conv2dBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBlock,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 16}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DecoderBlockV1,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBlockV2,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeconvConv2dBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EmptinessClassifier,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (GlobalConvolutionalNetwork,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LargeKernelMatters,
     lambda: ([], {'encoder_depth': 18, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (NoOperation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSPModule,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PSPUpsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetEncoders,
     lambda: ([], {'encoder_depth': 18}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SaltLinkNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SaltUNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SpatialSELayer,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (StackingFCN,
     lambda: ([], {'input_model_nr': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StackingUnet,
     lambda: ([], {'input_model_nr': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (UNetResNet,
     lambda: ([], {'encoder_depth': 34, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_neptune_ai_open_solution_salt_identification(_paritybench_base):
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

