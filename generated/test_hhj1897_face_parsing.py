import sys
_module = sys.modules[__name__]
del sys
face_parsing_test = _module
face_parsing = _module
parser = _module
resnet = _module
decoder = _module
convert_weights = _module
rtnet = _module
rtnet = _module
utils = _module
setup = _module

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


import time


import numpy as np


import torch


from torch import nn


import torch.nn.functional as F


import torchvision.transforms as T


from torch.nn.functional import softmax


import logging


import torchvision


from torchvision.models._utils import IntermediateLayerGetter


from typing import Dict


from typing import List


import torch.nn as nn


from torch.nn import BatchNorm2d


import math


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        return_layers = {'layer1': 'c1', 'layer2': 'c2', 'layer3': 'c3', 'layer4': 'c4'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, images, rois=None):
        return self.body(images)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str):
        if 'resnet18' in name or 'resnet34' in name:
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = [False, True, True]
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=replace_stride_with_dilation)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels)


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, bias=True)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class ASPP(nn.Module):

    def __init__(self, in_chan=2048, out_chan=256, with_gp=True, *args, **kwargs):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Decoder(nn.Module):

    def __init__(self, n_classes, low_chan=256, *args, **kwargs):
        super(Decoder, self).__init__()
        self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(ConvBNReLU(304, 256, ks=3, padding=1), ConvBNReLU(256, 256, ks=3, padding=1))
        self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear', align_corners=True)
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        logits = self.conv_out(feat_out)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class DeepLabV3Plus(nn.Module):

    def __init__(self, in_channels, num_classes, aspp_global_feature=False):
        super(DeepLabV3Plus, self).__init__()
        self.aspp = ASPP(in_chan=in_channels, out_chan=256, with_gp=aspp_global_feature)
        self.decoder = Decoder(num_classes, low_chan=256)
        self.low_level = True
        self.init_weight()

    def forward(self, x, low):
        feat_aspp = self.aspp(x)
        logits = self.decoder(low, feat_aspp)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class FCN(nn.Sequential):

    def __init__(self, in_channels, num_classes, **kwargs):
        inter_channels = in_channels // 4
        layers = [nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(inter_channels, num_classes, 1)]
        super(FCN, self).__init__(*layers)


DECODER_MAP = {'fcn': FCN, 'deeplabv3plus': DeepLabV3Plus}


class MixPad2d(nn.Module):
    """Mixed padding modes for H and W dimensions

    Args:
        padding (tuple): the size of the padding for x and y, ie (pad_x, pad_y)
        modes (tuple): the padding modes for x and y, the values of each can be
            ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``['replicate', 'circular']``

    """
    __constants__ = ['modes', 'padding']

    def __init__(self, padding=[1, 1], modes=['replicate', 'circular']):
        super(MixPad2d, self).__init__()
        assert len(padding) == 2
        self.padding = padding
        self.modes = modes

    def forward(self, x):
        x = nn.functional.pad(x, (0, 0, self.padding[1], self.padding[1]), self.modes[1])
        x = nn.functional.pad(x, (self.padding[0], self.padding[0], 0, 0), self.modes[0])
        return x

    def extra_repr(self):
        repr_ = 'Mixed Padding: \t x axis: mode: {}, padding: {},\n\t y axis mode: {}, padding: {}'.format(self.modes[0], self.padding[0], self.modes[1], self.padding[1])
        return repr_


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, mix_padding=False, padding_modes=['replicate', 'circular']):
    """3x3 convolution with padding"""
    if not mix_padding:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.Sequential(MixPad2d([dilation, dilation], padding_modes), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False, groups=groups, dilation=dilation))


class HybridBlock(nn.Module):
    expansion = 4
    pooling_r = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, bottleneck_width=32, avd=False, dilation=1, is_first=False, norm_layer=None, hybrid=False):
        super(HybridBlock, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        self.hybrid = hybrid
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        self.conv_polar, self.conv_cart = None, None
        self.conv_polar = nn.Sequential(conv3x3(group_width, group_width, stride=stride, groups=cardinality, dilation=dilation, mix_padding=True), norm_layer(group_width))
        self.conv_cart = nn.Sequential(conv3x3(group_width, group_width, stride=stride, groups=cardinality, dilation=dilation, mix_padding=False), norm_layer(group_width))
        self.conv3 = nn.Conv2d(group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        x, rois = x['x'], x['rois']
        residual = x
        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)
        out_a = self.conv_polar(out_a)
        if self.hybrid:
            _, _, h1, w1 = out_b.size()
            out_b = roi_tanh_polar_to_roi_tanh(out_b, rois, w1, h1, keep_aspect_ratio=True)
            out_b = self.conv_cart(out_b)
            _, _, h2, w2 = out_b.size()
            out_b = roi_tanh_to_roi_tanh_polar(out_b, rois / (h1 / h2), w2, h2, keep_aspect_ratio=True)
        else:
            out_b = self.conv_cart(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)
        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return dict(x=out, rois=rois)


class Bottleneck(nn.Module):
    """
    from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)
        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(first_planes, width, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None
        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def forward(self, x):
        x, rois = x['x'], x['rois']
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act3(x)
        return dict(x=x, rois=rois)


class RTNet(nn.Module):
    """ RTNet Variants Definations
    Parameters
    ----------
    block : Block
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block.
    dilated : bool, default False
        Applying dilation strategy to pretrained RTNet yielding a stride-8 model.
    deep_stem : bool, default False
        Replace 7x7 conv in input stem with 3 3x3 conv.
    avg_down : bool, default False
        Use AvgPool instead of stride conv when
        downsampling in the bottleneck.
    norm_layer : object
        Normalization layer used (default: :class:`torch.nn.BatchNorm2d`).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, groups=1, bottleneck_width=32, dilated=True, dilation=1, deep_stem=False, stem_width=64, avg_down=False, hybrid_stages=[True, True, True], avd=False, norm_layer=nn.BatchNorm2d, zero_init_residual=True, **kwargs):
        super(RTNet, self).__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd
        if hybrid_stages is None:
            hybrid_stages = [False, False, False]
        if len(hybrid_stages) != 3:
            raise ValueError('hybrid_stages should be None or a 3-element tuple, got {}'.format(hybrid_stages))
        None
        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, hybrid=hybrid_stages[0])
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, hybrid=hybrid_stages[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, hybrid=hybrid_stages[2])
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer, hybrid=hybrid_stages[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, norm_layer=norm_layer, hybrid=hybrid_stages[2])
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, hybrid=hybrid_stages[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, hybrid=hybrid_stages[2])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (HybridBlock, Bottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True, hybrid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, dilation=1, is_first=is_first, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, dilation=2, is_first=is_first, norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not hybrid:
                layers.append(Bottleneck(self.inplanes, planes, cardinality=self.cardinality))
            else:
                layers.append(block(self.inplanes, planes, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, dilation=dilation, norm_layer=norm_layer, hybrid=hybrid))
        return nn.Sequential(*layers)

    def forward(self, x, rois, *args, **kwargs):
        _, _, H, _ = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        _, _, H_stem, _ = x.shape
        s1 = H / H_stem
        c1 = self.layer1(dict(x=x, rois=rois / s1))['x']
        s2 = H / c1.shape[2]
        c2 = self.layer2(dict(x=c1, rois=rois / s2))['x']
        s3 = H / c2.shape[2]
        c3 = self.layer3(dict(x=c2, rois=rois / s3))['x']
        s4 = H / c3.shape[2]
        c4 = self.layer4(dict(x=c3, rois=rois / s4))['x']
        return dict(c1=c1, c2=c2, c3=c3, c4=c4)


def rtnet101(pretrained=False, **kwargs):
    """Constructs a RTNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RTNet(HybridBlock, [3, 4, 23, 3], deep_stem=False, stem_width=64, avg_down=False, avd=False, **kwargs)
    model.num_channels = 2048
    return model


def rtnet50(pretrained=False, **kwargs):
    """Constructs a RTNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RTNet(HybridBlock, [3, 4, 6, 3], deep_stem=False, stem_width=32, avg_down=False, avd=False, **kwargs)
    model.num_channels = 2048
    return model


ENCODER_MAP = {'rtnet50': [rtnet50, 2048], 'rtnet101': [rtnet101, 2048]}


class SegmentationModel(nn.Module):

    def __init__(self, encoder='rtnet50', decoder='fcn', num_classes=14):
        super().__init__()
        if 'rtnet' in encoder:
            encoder_func, in_channels = ENCODER_MAP[encoder.lower()]
            self.encoder = encoder_func()
        else:
            self.encoder = Backbone(encoder)
            in_channels = self.encoder.num_channels
        self.decoder = DECODER_MAP[decoder.lower()](in_channels=in_channels, num_classes=num_classes)
        self.low_level = getattr(self.decoder, 'low_level', False)

    def forward(self, x, rois):
        input_shape = x.shape[-2:]
        features = self.encoder(x, rois)
        low = features['c1']
        high = features['c4']
        if self.low_level:
            x = self.decoder(high, low)
        else:
            x = self.decoder(high)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_chan': 4, 'out_chan': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {'n_classes': 4}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 256, 64, 64])], {}),
     True),
    (DeepLabV3Plus,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 256, 64, 64])], {}),
     False),
    (FCN,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixPad2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hhj1897_face_parsing(_paritybench_base):
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

