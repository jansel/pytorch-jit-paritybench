import sys
_module = sys.modules[__name__]
del sys
base = _module
base_dataloader = _module
base_dataset = _module
base_model = _module
base_trainer = _module
dataloaders = _module
ade20k = _module
cityscapes = _module
coco = _module
voc = _module
inference = _module
models = _module
deeplabv3_plus = _module
duc_hdc = _module
enet = _module
fcn = _module
gcn = _module
pspnet = _module
resnet = _module
segnet = _module
unet = _module
upernet = _module
train = _module
trainer = _module
utils = _module
helpers = _module
logger = _module
losses = _module
lovasz_losses = _module
lr_scheduler = _module
metrics = _module
palette = _module
sync_batchnorm = _module
batchnorm = _module
batchnorm_reimpl = _module
comm = _module
replicate = _module
unittest = _module
torchsummary = _module
transforms = _module

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


import logging


import torch.nn as nn


import numpy as np


import math


import torch


from torch.utils import tensorboard


import scipy


import torch.nn.functional as F


from scipy import ndimage


from math import ceil


import torch.utils.model_zoo as model_zoo


from itertools import chain


from torch import nn


import time


from torch.autograd import Variable


import collections


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as init


import functools


from torch.nn.parallel.data_parallel import DataParallel


from collections import OrderedDict


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__(
            ) + f'\nNbr of trainable parameters: {nbr_params}'


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0001)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class ResNet(nn.Module):

    def __init__(self, in_channels=3, output_stride=16, backbone=
        'resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 64, 7,
                stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.
                ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2,
                padding=1))
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        if output_stride == 16:
            s3, s4, d3, d4 = 2, 1, 1, 2
        elif output_stride == 8:
            s3, s4, d3, d4 = 1, 1, 2, 4
        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone ==
                    'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3,
                        s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3,
                        s3)
                elif 'downsample.0' in n:
                    m.stride = s3, s3
        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone ==
                'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = s4, s4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_features


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
            stride, padding=padding, dilation=dilation, groups=in_channels,
            bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
        exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=
                stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        rep = []
        self.relu = nn.ReLU(inplace=True)
        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1,
            dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1,
            dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=
            stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [self.relu, SeparableConv2d(in_channels, in_channels,
                3, 1, dilation), nn.BatchNorm2d(in_channels)]
        if not use_1st_relu:
            rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        x = output + skip
        return x


class Xception(nn.Module):

    def __init__(self, output_stride=16, in_channels=3, pretrained=True):
        super(Xception, self).__init__()
        if output_stride == 16:
            b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8:
            b3_s, mf_d, ef_d = 1, 2, (2, 4)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, stride=2, dilation=1, use_1st_relu=False)
        self.block2 = Block(128, 256, stride=2, dilation=1)
        self.block3 = Block(256, 728, stride=b3_s, dilation=1)
        for i in range(16):
            exec(
                f'self.block{i + 4} = Block(728, 728, stride=1, dilation=mf_d)'
                )
        self.block20 = Block(728, 1024, stride=1, dilation=ef_d[0],
            exit_flow=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=ef_d[1])
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=ef_d[1])
        self.bn5 = nn.BatchNorm2d(2048)
        initialize_weights(self)
        if pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        url = (
            'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
            )
        pretrained_weights = model_zoo.load_url(url)
        state_dict = self.state_dict()
        model_dict = {}
        for k, v in pretrained_weights.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    for i in range(8):
                        model_dict[k.replace('block11', f'block{i + 12}')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
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
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, low_level_features


def assp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(nn.Conv2d(in_channels, out_channles, kernel_size,
        padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(
        out_channles), nn.ReLU(inplace=True))


class ASSP(nn.Module):

    def __init__(self, in_channels, output_stride):
        super(ASSP, self).__init__()
        assert output_stride in [8, 16
            ], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.
            Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)),
            mode='bilinear', align_corners=True)
        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        return x


class Decoder(nn.Module):

    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(nn.Conv2d(48 + 256, 256, 3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=
            True), nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn
            .Conv2d(256, num_classes, 1, stride=1))
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


class DUC(nn.Module):

    def __init__(self, in_channels, out_channles, upscale):
        super(DUC, self).__init__()
        out_channles = out_channles * upscale ** 2
        self.conv = nn.Conv2d(in_channels, out_channles, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.pixl_shf = nn.PixelShuffle(upscale_factor=upscale)
        initialize_weights(self)
        kernel = self.icnr(self.conv.weight, scale=upscale)
        self.conv.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pixl_shf(x)
        return x

    def icnr(self, x, scale=2, init=nn.init.kaiming_normal):
        """
        Even with pixel shuffle we still have check board artifacts,
        the solution is to initialize the d**2 feature maps with the same
        radom weights: https://arxiv.org/pdf/1707.02937.pdf
        """
        new_shape = [int(x.shape[0] / scale ** 2)] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
            subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel


class ResNet_HDC_DUC(nn.Module):

    def __init__(self, in_channels, output_stride, pretrained=True,
        dilation_bigger=False):
        super(ResNet_HDC_DUC, self).__init__()
        model = models.resnet101(pretrained=pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 64, 7,
                stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.
                ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2,
                padding=1))
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        if output_stride == 4:
            list(self.layer0.children())[0].stride = 1, 1
        d_res4b = []
        if dilation_bigger:
            d_res4b.extend([1, 2, 5, 9] * 5 + [1, 2, 5])
            d_res5b = [5, 9, 17]
        else:
            d_res4b.extend([1, 2, 3] * 7 + [2, 2])
            d_res5b = [3, 4, 5]
        l_index = 0
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                d = d_res4b[l_index]
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                l_index += 1
            elif 'downsample.0' in n:
                m.stride = 1, 1
        l_index = 0
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                d = d_res5b[l_index]
                m.dilation, m.padding, m.stride = (d, d), (d, d), (1, 1)
                l_index += 1
            elif 'downsample.0' in n:
                m.stride = 1, 1

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_features


class ASSP(nn.Module):

    def __init__(self, in_channels, output_stride, assp_channels=6):
        super(ASSP, self).__init__()
        assert output_stride in [4, 8
            ], 'Only output strides of 8 or 16 are suported'
        assert assp_channels in [4, 6
            ], 'Number of suported ASSP branches are 4 or 6'
        dilations = [1, 6, 12, 18, 24, 36]
        dilations = dilations[:assp_channels]
        self.assp_channels = assp_channels
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, 256, 3, dilation=dilations[3])
        if self.assp_channels == 6:
            self.aspp5 = assp_branch(in_channels, 256, 3, dilation=dilations[4]
                )
            self.aspp6 = assp_branch(in_channels, 256, 3, dilation=dilations[5]
                )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.
            Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(256 * (self.assp_channels + 1), 256, 1, bias
            =False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        if self.assp_channels == 6:
            x5 = self.aspp5(x)
            x6 = self.aspp6(x)
        x_avg_pool = F.interpolate(self.avg_pool(x), size=(x.size(2), x.
            size(3)), mode='bilinear', align_corners=True)
        if self.assp_channels == 6:
            x = self.conv1(torch.cat((x1, x2, x3, x4, x5, x6, x_avg_pool),
                dim=1))
        else:
            x = self.conv1(torch.cat((x1, x2, x3, x4, x_avg_pool), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        return x


class Decoder(nn.Module):

    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.DUC = DUC(256, 256, upscale=2)
        self.output = nn.Sequential(nn.Conv2d(48 + 256, 256, 3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=
            True), nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1), nn
            .Conv2d(256, num_classes, 1, stride=1))
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        x = self.DUC(x)
        if x.size() != low_level_features.size():
            x = x[:, :, :low_level_features.size(2), :low_level_features.
                size(3)]
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x


class InitalBlock(nn.Module):

    def __init__(self, in_channels, use_prelu=True):
        super(InitalBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels, 16 - in_channels, 3, padding=1,
            stride=2)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16) if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat((self.pool(x), self.conv(x)), dim=1)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels=None, activation=None,
        dilation=1, downsample=False, proj_ratio=4, upsample=False,
        asymetric=False, regularize=True, p_drop=None, use_prelu=True):
        super(BottleNeck, self).__init__()
        self.pad = 0
        self.upsample = upsample
        self.downsample = downsample
        if out_channels is None:
            out_channels = in_channels
        else:
            self.pad = out_channels - in_channels
        if regularize:
            assert p_drop is not None
        if downsample:
            assert not upsample
        elif upsample:
            assert not downsample
        inter_channels = in_channels // proj_ratio
        if upsample:
            self.spatil_conv = nn.Conv2d(in_channels, out_channels, 1, bias
                =False)
            self.bn_up = nn.BatchNorm2d(out_channels)
            self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        elif downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2,
                return_indices=True)
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 2, stride=2,
                bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        if asymetric:
            self.conv2 = nn.Sequential(nn.Conv2d(inter_channels,
                inter_channels, kernel_size=(1, 5), padding=(0, 2)), nn.
                BatchNorm2d(inter_channels), nn.PReLU() if use_prelu else
                nn.ReLU(inplace=True), nn.Conv2d(inter_channels,
                inter_channels, kernel_size=(5, 1), padding=(2, 0)))
        elif upsample:
            self.conv2 = nn.ConvTranspose2d(inter_channels, inter_channels,
                kernel_size=3, padding=1, output_padding=1, stride=2, bias=
                False)
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3,
                padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.prelu3 = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)
        self.regularizer = nn.Dropout2d(p_drop) if regularize else None
        self.prelu_out = nn.PReLU() if use_prelu else nn.ReLU(inplace=True)

    def forward(self, x, indices=None, output_size=None):
        identity = x
        if self.upsample:
            assert indices is not None and output_size is not None
            identity = self.bn_up(self.spatil_conv(identity))
            if identity.size() != indices.size():
                pad = indices.size(3) - identity.size(3), 0, indices.size(2
                    ) - identity.size(2), 0
                identity = F.pad(identity, pad, 'constant', 0)
            identity = self.unpool(identity, indices=indices)
        elif self.downsample:
            identity, idx = self.pool(identity)
        """
        if self.pad > 0:
            if self.pad % 2 == 0 : pad = (0, 0, 0, 0, self.pad//2, self.pad//2)
            else: pad = (0, 0, 0, 0, self.pad//2, self.pad//2+1)
            identity = F.pad(identity, pad, "constant", 0)
        """
        if self.pad > 0:
            extras = torch.zeros((identity.size(0), self.pad, identity.size
                (2), identity.size(3)))
            if torch.cuda.is_available():
                extras = extras
            identity = torch.cat((identity, extras), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        if self.regularizer is not None:
            x = self.regularizer(x)
        if identity.size() != x.size():
            pad = identity.size(3) - x.size(3), 0, identity.size(2) - x.size(2
                ), 0
            x = F.pad(x, pad, 'constant', 0)
        x += identity
        x = self.prelu_out(x)
        if self.downsample:
            return x, idx
        return x


class Block_Resnet_GCN(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(Block_Resnet_GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, bias=False,
            stride=stride, kernel_size=(kernel_size, 1), padding=(
            kernel_size // 2, 0))
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(out_channels, out_channels, bias=False,
            stride=stride, kernel_size=(1, kernel_size), padding=(0, 
            kernel_size // 2))
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv21 = nn.Conv2d(in_channels, out_channels, bias=False,
            stride=stride, kernel_size=(1, kernel_size), padding=(0, 
            kernel_size // 2))
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(out_channels, out_channels, bias=False,
            stride=stride, kernel_size=(kernel_size, 1), padding=(
            kernel_size // 2, 0))
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.relu22 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)
        x = x1 + x2
        return x


class BottleneckGCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
        out_channels_gcn, stride=1):
        super(BottleneckGCN, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d
                (out_channels))
        else:
            self.downsample = None
        self.gcn = Block_Resnet_GCN(kernel_size, in_channels, out_channels_gcn)
        self.conv1x1 = nn.Conv2d(out_channels_gcn, out_channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.gcn(x)
        x = self.conv1x1(x)
        x = self.bn1x1(x)
        x += identity
        return x


class ResnetGCN(nn.Module):

    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128),
        kernel_sizes=(5, 7)):
        super(ResnetGCN, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=False)
        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                padding=3, bias=False)
        self.initial = nn.Sequential(conv1, resnet.bn1, resnet.relu, resnet
            .maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = nn.Sequential(BottleneckGCN(512, 1024, kernel_sizes[0
            ], out_channels_gcn[0], stride=2), *([BottleneckGCN(1024, 1024,
            kernel_sizes[0], out_channels_gcn[0])] * 5))
        self.layer4 = nn.Sequential(BottleneckGCN(1024, 2048, kernel_sizes[
            1], out_channels_gcn[1], stride=2), *([BottleneckGCN(1024, 1024,
            kernel_sizes[1], out_channels_gcn[1])] * 5))
        initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = x.size(2), x.size(3)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz


class Resnet(nn.Module):

    def __init__(self, in_channels, backbone, out_channels_gcn=(85, 128),
        pretrained=True, kernel_sizes=(5, 7)):
        super(Resnet, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained)
        if in_channels == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                padding=3, bias=False)
        self.initial = nn.Sequential(conv1, resnet.bn1, resnet.relu, resnet
            .maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if not pretrained:
            initialize_weights(self)

    def forward(self, x):
        x = self.initial(x)
        conv1_sz = x.size(2), x.size(3)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4, conv1_sz


class GCN_Block(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels):
        super(GCN_Block, self).__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd'
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=(
            kernel_size, 1), padding=(kernel_size // 2, 0))
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,
            kernel_size), padding=(0, kernel_size // 2))
        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,
            kernel_size), padding=(0, kernel_size // 2))
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=(
            kernel_size, 1), padding=(kernel_size // 2, 0))
        initialize_weights(self)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.conv21(x)
        x2 = self.conv22(x2)
        x = x1 + x2
        return x


class BR_Block(nn.Module):

    def __init__(self, num_channels):
        super(BR_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        initialize_weights(self)

    def forward(self, x):
        identity = x
        x = self.conv2(self.relu2(self.conv1(x)))
        x += identity
        return x


class _PSPModule(nn.Module):

    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels,
            out_channels, b_s, norm_layer) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + 
            out_channels * len(bin_sizes), out_channels, kernel_size=3,
            padding=1, bias=False), norm_layer(out_channels), nn.ReLU(
            inplace=True), nn.Dropout2d(0.1))

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode=
            'bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


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

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, num_classes=1000, dilated=True,
        multi_grid=False, deep_base=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                stride=2, padding=1, bias=False), norm_layer(64), nn.ReLU(
                inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1,
                padding=1, bias=False), norm_layer(64), nn.ReLU(inplace=
                True), nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=
                1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=
                3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3],
                    stride=1, dilation=4, norm_layer=norm_layer, multi_grid
                    =True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3],
                    stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=
                multi_dilations[0], downsample=downsample,
                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
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
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=
                    multi_dilations[i], previous_dilation=dilation,
                    norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=
                    dilation, previous_dilation=dilation, norm_layer=
                    norm_layer))
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


class DecoderBottleneck(nn.Module):

    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.ConvTranspose2d(inchannels // 4, inchannels // 4,
            kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.ConvTranspose2d(inchannels, 
            inchannels // 2, kernel_size=2, stride=2, bias=False), nn.
            BatchNorm2d(inchannels // 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LastBottleneck(nn.Module):

    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.Conv2d(inchannels // 4, inchannels // 4,
            kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(inchannels, inchannels //
            4, kernel_size=1, bias=False), nn.BatchNorm2d(inchannels // 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size
            =2, stride=2)
        self.up_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True), nn.Conv2d(out_channels, out_channels,
            kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.
            ReLU(inplace=True))

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        if interpolate:
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                mode='bilinear', align_corners=True)
        else:
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
                diffY // 2))
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class PSPModule(nn.Module):

    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels,
            out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + 
            out_channels * len(bin_sizes), in_channels, kernel_size=3,
            padding=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(
            inplace=True), nn.Dropout2d(0.1))

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode=
            'bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):

    def __init__(self, in_channels=3, output_stride=16, backbone=
        'resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(nn.Conv2d(in_channels, 64, 7,
                stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.
                ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2,
                padding=1))
            initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        if output_stride == 16:
            s3, s4, d3, d4 = 2, 1, 1, 2
        elif output_stride == 8:
            s3, s4, d3, d4 = 1, 1, 2, 4
        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone ==
                    'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3,
                        s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3,
                        s3)
                elif 'downsample.0' in n:
                    m.stride = s3, s3
        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone ==
                'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = s4, s4

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear',
        align_corners=True) + y


class FPN_fuse(nn.Module):

    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out,
            kernel_size=1) for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out,
            kernel_size=3, padding=1)] * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(nn.Conv2d(len(feature_channels) *
            fpn_out, fpn_out, kernel_size=3, padding=1, bias=False), nn.
            BatchNorm2d(fpn_out), nn.ReLU(inplace=True))

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(
            features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(
            range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear',
            align_corners=True) for feature in P[1:]]
        x = self.conv_fusion(torch.cat(P, dim=1))
        return x


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=
            ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size
        ()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1]
            )
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - (2.0 * intersection + self.smooth) / (output_flat.sum() +
            target_flat.sum() + self.smooth)
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True
        ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=
            ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):

    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight
        =None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=
            reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes is 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, (0)]
        else:
            class_pred = probas[:, (c)]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(
            fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, classes='present', per_image=False,
    ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0),
            lab.unsqueeze(0), ignore), classes=classes) for prob, lab in
            zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore),
            classes=classes)
    return loss


class LovaszSoftmax(nn.Module):

    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(
                input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(
                input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std *
                self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum
                    ) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum
                    ) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum
                ) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum
                ) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1
            ) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0,
            2, 3).contiguous()


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module,
            device_ids)
        execute_replication_callbacks(modules)
        return modules

