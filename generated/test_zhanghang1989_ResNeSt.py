import sys
_module = sys.modules[__name__]
del sys
hubconf = _module
resnest = _module
gluon = _module
ablation = _module
data_utils = _module
dropblock = _module
model_store = _module
model_zoo = _module
resnet = _module
splat = _module
torch = _module
resnet = _module
splat = _module
transforms = _module
utils = _module
prepare_imagenet = _module
train = _module
verify = _module
verify = _module
setup = _module
test_radix_major = _module
test_torch = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import torch


import torch.nn as nn


from torch import nn


import torch.nn.functional as F


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Linear


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn.modules.utils import _pair


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import warnings


import numpy as np


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.
            size(0), -1)


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1,
        cardinality=1, bottleneck_width=64, avd=False, avd_first=False,
        dilation=1, is_first=False, rectified_conv=False, rectify_avg=False,
        norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False
            )
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)
        if radix >= 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=
                3, stride=stride, padding=dilation, dilation=dilation,
                groups=cardinality, bias=False, radix=radix, rectify=
                rectified_conv, rectify_avg=rectify_avg, norm_layer=
                norm_layer, dropblock_prob=dropblock_prob)
        elif rectified_conv:
            self.conv2 = RFConv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False, average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias
            =False)
        self.bn3 = norm_layer(planes * 4)
        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet Variants

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

    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=
        64, num_classes=1000, dilated=False, dilation=1, deep_stem=False,
        stem_width=64, avg_down=False, rectified_conv=False, rectify_avg=
        False, avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0,
        last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width,
                kernel_size=3, stride=2, padding=1, bias=False, **
                conv_kwargs), norm_layer(stem_width), nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1,
                padding=1, bias=False, **conv_kwargs), norm_layer(
                stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, 
                stem_width * 2, kernel_size=3, stride=1, padding=1, bias=
                False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding
                =3, bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=
            norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, norm_layer=norm_layer, dropblock_prob=
                dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, norm_layer=norm_layer, dropblock_prob=
                dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                dilation=1, norm_layer=norm_layer, dropblock_prob=
                dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=2, norm_layer=norm_layer, dropblock_prob=
                dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        norm_layer=None, dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride,
                        stride=stride, ceil_mode=True, count_include_pad=False)
                        )
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                        ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.
                    expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.
                    expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=
                downsample, radix=self.radix, cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width, avd=self.avd,
                avd_first=self.avd_first, dilation=1, is_first=is_first,
                rectified_conv=self.rectified_conv, rectify_avg=self.
                rectify_avg, norm_layer=norm_layer, dropblock_prob=
                dropblock_prob, last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=
                downsample, radix=self.radix, cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width, avd=self.avd,
                avd_first=self.avd_first, dilation=2, is_first=is_first,
                rectified_conv=self.rectified_conv, rectify_avg=self.
                rectify_avg, norm_layer=norm_layer, dropblock_prob=
                dropblock_prob, last_gamma=self.last_gamma))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix,
                cardinality=self.cardinality, bottleneck_width=self.
                bottleneck_width, avd=self.avd, avd_first=self.avd_first,
                dilation=dilation, rectified_conv=self.rectified_conv,
                rectify_avg=self.rectify_avg, norm_layer=norm_layer,
                dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
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
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        return x


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
        padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2,
        reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=
        None, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.
            cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class RadixMajorNaiveImp(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
        padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2,
        reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=
        None, dropblock_prob=0.0, **kwargs):
        super(RadixMajorNaiveImp, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                **kwargs)
        self.use_bn = norm_layer is not None
        assert not self.use_bn
        self.relu = ReLU(inplace=True)
        cardinal_group_width = channels // groups
        cardinal_inter_channels = inter_channels // groups
        self.fc1 = nn.ModuleList([nn.Linear(cardinal_group_width,
            cardinal_inter_channels) for _ in range(groups)])
        self.fc2 = nn.ModuleList([nn.Linear(cardinal_inter_channels, 
            cardinal_group_width * radix) for _ in range(groups)])
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, channel = x.shape[:2]
        cardinality = self.cardinality
        radix = self.radix
        tiny_group_width = channel // radix // cardinality
        all_groups = torch.split(x, tiny_group_width, dim=1)
        out = []
        for k in range(cardinality):
            U_k = [all_groups[r * cardinality + k] for r in range(radix)]
            U_k = sum(U_k)
            gap_k = F.adaptive_avg_pool2d(U_k, 1).squeeze()
            atten_k = self.fc2[k](self.fc1[k](gap_k))
            if radix > 1:
                x_k = [all_groups[r * cardinality + k] for r in range(radix)]
                x_k = torch.cat(x_k, dim=1)
                atten_k = atten_k.view(batch, radix, -1)
                atten_k = F.softmax(atten_k, dim=1)
            else:
                x_k = all_groups[k]
                atten_k = F.sigmoid(atten_k)
            attended_k = x_k * atten_k.view(batch, -1, 1, 1)
            out_k = sum(torch.split(attended_k, attended_k.size(1) // self.
                radix, dim=1))
            out.append(out_k)
        return torch.cat(out, dim=1).contiguous()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhanghang1989_ResNeSt(_paritybench_base):
    pass
    def test_000(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(RadixMajorNaiveImp(*[], **{'in_channels': 4, 'channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(SplAtConv2d(*[], **{'in_channels': 4, 'channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(rSoftMax(*[], **{'radix': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

