import sys
_module = sys.modules[__name__]
del sys
conf = _module
custom_directives = _module
encoding = _module
datasets = _module
ade20k = _module
base = _module
cityscapes = _module
cityscapescoarse = _module
coco = _module
folder = _module
hpw18 = _module
imagenet = _module
minc = _module
pascal_aug = _module
pascal_voc = _module
pcontext = _module
functions = _module
customize = _module
dist_syncbn = _module
encoding = _module
rectify = _module
syncbn = _module
lib = _module
setup = _module
models = _module
backbone = _module
resnest = _module
resnet = _module
resnet_variants = _module
resnext = _module
wideresnet = _module
xception = _module
deepten = _module
model_store = _module
model_zoo = _module
sseg = _module
atten = _module
base = _module
deeplab = _module
encnet = _module
fcfpn = _module
fcn = _module
psp = _module
upernet = _module
nn = _module
attention = _module
customize = _module
dropblock = _module
encoding = _module
loss = _module
rectify = _module
splat = _module
syncbn = _module
parallel = _module
transforms = _module
autoaug = _module
get_transform = _module
utils = _module
dist_helper = _module
files = _module
lr_scheduler = _module
metrics = _module
misc = _module
pallete = _module
precise_bn = _module
presets = _module
train_helper = _module
test_flops = _module
train_dist = _module
verify = _module
demo = _module
test = _module
test_models = _module
train = _module
train_dist = _module
prepare_ade20k = _module
prepare_citys = _module
prepare_coco = _module
prepare_imagenet = _module
prepare_minc = _module
prepare_pascal = _module
prepare_pcontext = _module
lint = _module
test_dataset = _module
test_function = _module
test_model = _module
test_module = _module
test_utils = _module

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


import warnings


import torch


from torch.autograd import Function


from torch.autograd import Variable


import torch.nn.functional as F


import math


import torch.nn as nn


from collections import OrderedDict


from functools import partial


import numpy as np


from torch.nn.functional import interpolate


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.nn.parallel.scatter_gather import scatter


from torch.nn.functional import upsample


from torch.nn import functional as F


from torch import nn


from torch.nn import Module


from torch.nn import Parameter


from torch.nn.modules.utils import _pair


from torch.nn import Conv2d


from torch.nn import Linear


from torch.nn import BatchNorm2d


from torch.nn import ReLU


from torch.nn.modules.batchnorm import _BatchNorm


import functools


import torch.cuda.comm as comm


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


import itertools


from typing import Any


from typing import Iterable


from typing import List


from typing import Tuple


from typing import Type


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


from torch.utils import data


from torch.nn.parallel.scatter_gather import gather


import copy


import torch.backends.cudnn as cudnn


from torch.autograd import gradcheck


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


class Bottleneck(nn.Module):
    """WideResNet BottleneckV1b
    """

    def __init__(self, inplanes, planes, stride=1, dilation=1, expansion=4,
        dropout=0.0, downsample=None, previous_dilation=1, **kwargs):
        super(Bottleneck, self).__init__()
        self.bn1 = ABN(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = ABN(planes)
        self.conv2 = nn.Conv2d(planes, planes * expansion // 2, kernel_size
            =3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn3 = ABN(planes * expansion // 2)
        self.conv3 = nn.Conv2d(planes * expansion // 2, planes * expansion,
            kernel_size=1, bias=False)
        self.downsample = downsample
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.downsample:
            bn1 = self.bn1(x)
            residual = self.downsample(bn1)
        else:
            residual = x.clone()
            bn1 = self.bn1(x)
        out = self.conv1(bn1)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.drop:
            out = self.drop(out)
        out = self.conv3(out)
        out = out + residual
        return out


class WideResNet(nn.Module):
    """ Pre-trained WideResNet Model
    featuremaps at conv5.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.

    Reference:

        - Zifeng Wu, et al. "Wider or Deeper: Revisiting the ResNet Model for Visual Recognition"

        - Samuel Rota Bulò, et al. 
            "In-Place Activated BatchNorm for Memory-Optimized Training of DNNs"
    """

    def __init__(self, layers, classes=1000, dilated=False, **kwargs):
        self.inplanes = 64
        super(WideResNet, self).__init__()
        self.mod1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.mod2 = self._make_layer(2, BasicBlock, 128, layers[0])
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.mod3 = self._make_layer(3, BasicBlock, 256, layers[1], stride=1)
        self.mod4 = self._make_layer(4, BasicBlock, 512, layers[2], stride=2)
        if dilated:
            self.mod5 = self._make_layer(5, BasicBlock, 512, layers[3],
                stride=1, dilation=2, expansion=2)
            self.mod6 = self._make_layer(6, Bottleneck, 512, layers[4],
                stride=1, dilation=4, expansion=4, dropout=0.3)
            self.mod7 = self._make_layer(7, Bottleneck, 1024, layers[5],
                stride=1, dilation=4, expansion=4, dropout=0.5)
        else:
            self.mod5 = self._make_layer(5, BasicBlock, 512, layers[3],
                stride=2, expansion=2)
            self.mod6 = self._make_layer(6, Bottleneck, 512, layers[4],
                stride=2, expansion=4, dropout=0.3)
            self.mod7 = self._make_layer(7, Bottleneck, 1024, layers[5],
                stride=1, expansion=4, dropout=0.5)
        self.bn_out = ABN(4096)
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(4096, classes)

    def _make_layer(self, stage_index, block, planes, blocks, stride=1,
        dilation=1, expansion=1, dropout=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                expansion, kernel_size=1, stride=stride, bias=False))
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                expansion=expansion, dropout=dropout, downsample=downsample,
                previous_dilation=dilation))
        elif dilation == 4 and stage_index < 7:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                expansion=expansion, dropout=dropout, downsample=downsample,
                previous_dilation=dilation))
        else:
            assert stage_index == 7
            layers.append(block(self.inplanes, planes, stride, dilation=
                dilation, expansion=expansion, dropout=dropout, downsample=
                downsample, previous_dilation=dilation))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                expansion=expansion, dropout=dropout, previous_dilation=
                dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mod1(x)
        x = self.pool2(x)
        x = self.mod2(x)
        x = self.pool3(x)
        x = self.mod3(x)
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.bn_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=
        1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0,
            dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1
            .dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
        norm_layer=None, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=
                False)
            self.skipbn = norm_layer(planes)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation,
                norm_layer=norm_layer))
            rep.append(norm_layer(planes))
            filters = planes
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation,
                norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation,
                norm_layer=norm_layer))
            rep.append(norm_layer(planes))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, norm_layer=
                norm_layer))
            rep.append(norm_layer(planes))
        elif is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, dilation,
                norm_layer=norm_layer))
            rep.append(norm_layer(planes))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x


class Xception65(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception65, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 2
            exit_block_dilations = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block20_stride = 1
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
            norm_layer=norm_layer, start_with_relu=True, grow_first=True,
            is_last=True)
        midflowblocks = []
        for i in range(4, 20):
            midflowblocks.append(('block%d' % i, Block(728, 728, reps=3,
                stride=1, dilation=middle_block_dilation, norm_layer=
                norm_layer, start_with_relu=True, grow_first=True)))
        self.midflow = nn.Sequential(OrderedDict(midflowblocks))
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride,
            dilation=exit_block_dilations[0], norm_layer=norm_layer,
            start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.midflow(x)
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Xception71(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception71, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 2
            exit_block_dilations = 1, 1
        elif output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block20_stride = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block20_stride = 1
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False)
        block2 = []
        block2.append(Block(128, 256, reps=2, stride=1, norm_layer=
            norm_layer, start_with_relu=False, grow_first=True))
        block2.append(Block(256, 256, reps=2, stride=2, norm_layer=
            norm_layer, start_with_relu=False, grow_first=True))
        block2.append(Block(256, 728, reps=2, stride=1, norm_layer=
            norm_layer, start_with_relu=False, grow_first=True))
        self.block2 = nn.Sequential(*block2)
        self.block3 = Block(728, 728, reps=2, stride=entry_block3_stride,
            norm_layer=norm_layer, start_with_relu=True, grow_first=True,
            is_last=True)
        midflowblocks = []
        for i in range(4, 20):
            midflowblocks.append(('block%d' % i, Block(728, 728, reps=3,
                stride=1, dilation=middle_block_dilation, norm_layer=
                norm_layer, start_with_relu=True, grow_first=True)))
        self.midflow = nn.Sequential(OrderedDict(midflowblocks))
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride,
            dilation=exit_block_dilations[0], norm_layer=norm_layer,
            start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=
            exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.midflow(x)
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


_model_sha1 = {name: checksum for checksum, name in [(
    'fb9de5b360976e3e8bd3679d3e93c5409a5eff3c', 'resnest50'), (
    '966fb78c22323b0c68097c5c1242bd16d3e07fd5', 'resnest101'), (
    'd7fd712f5a1fcee5b3ce176026fbb6d0d278454a', 'resnest200'), (
    '51ae5f19032e22af4ec08e695496547acdba5ce5', 'resnest269'), (
    'a75c83cfc89a56a4e8ba71b14f1ec67e923787b3', 'resnet50s'), (
    '03a0f310d6447880f1b22a83bd7d1aa7fc702c6e', 'resnet101s'), (
    '36670e8bc2428ecd5b7db1578538e2dd23872813', 'resnet152s'), (
    'da4785cfc837bf00ef95b52fb218feefe703011f', 'wideresnet38'), (
    'b41562160173ee2e979b795c551d3c7143b1e5b5', 'wideresnet50'), (
    '1225f149519c7a0113c43a056153c1bb15468ac0', 'deepten_resnet50_minc'), (
    '662e979de25a389f11c65e9f1df7e06c2c356381', 'fcn_resnet50s_ade'), (
    '4de91d5922d4d3264f678b663f874da72e82db00', 'encnet_resnet50s_pcontext'
    ), ('9f27ea13d514d7010e59988341bcbd4140fcc33d',
    'encnet_resnet101s_pcontext'), (
    '07ac287cd77e53ea583f37454e17d30ce1509a4a', 'encnet_resnet50s_ade'), (
    '3f54fa3b67bac7619cd9b3673f5c8227cf8f4718', 'encnet_resnet101s_ade'), (
    '4aba491aaf8e4866a9c9981b210e3e3266ac1f2a', 'fcn_resnest50_ade'), (
    '2225f09d0f40b9a168d9091652194bc35ec2a5a9', 'deeplab_resnest50_ade'), (
    '06ca799c8cc148fe0fafb5b6d052052935aa3cc8', 'deeplab_resnest101_ade'),
    ('7b9e7d3e6f0e2c763c7d77cad14d306c0a31fe05', 'deeplab_resnest200_ade'),
    ('0074dd10a6e6696f6f521653fb98224e75955496', 'deeplab_resnest269_ade'),
    ('77a2161deeb1564e8b9c41a4bb7a3f33998b00ad', 'fcn_resnest50_pcontext'),
    ('08dccbc4f4694baab631e037a374d76d8108c61f',
    'deeplab_resnest50_pcontext'), (
    'faf5841853aae64bd965a7bdc2cdc6e7a2b5d898',
    'deeplab_resnest101_pcontext'), (
    'fe76a26551dd5dcf2d474fd37cba99d43f6e984e',
    'deeplab_resnest200_pcontext'), (
    'b661fd26c49656e01e9487cd9245babb12f37449', 'deeplab_resnest269_pcontext')]
    }


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest() == sha1_hash


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.
            format(name=name))
    return _model_sha1[name][:8]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    if overwrite or not os.path.exists(fname) or sha1_hash and not check_sha1(
        fname, sha1_hash):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError('Failed downloading url %s' % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=
                    int(total_length / 1024.0 + 0.5), unit='KB', unit_scale
                    =False, dynamic_ncols=True):
                    f.write(chunk)
        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. The repo may be outdated or download may be incomplete. If the "repo_url" is overridden, consider switching to the default repo.'
                .format(fname))
    return fname


_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'


encoding_repo_url = 'https://hangzh.s3-us-west-1.amazonaws.com/'


class GlobalPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


class ATTENHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs,
        nheads, nmixs, with_global, with_enc, se_loss, lateral):
        super(ATTENHead, self).__init__()
        self.with_enc = with_enc
        self.se_loss = se_loss
        self._up_kwargs = up_kwargs
        inter_channels = in_channels // 4
        self.lateral = lateral
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        if lateral:
            self.connect = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512,
                kernel_size=1, bias=False), norm_layer(512), nn.ReLU(
                inplace=True)), nn.Sequential(nn.Conv2d(1024, 512,
                kernel_size=1, bias=False), norm_layer(512), nn.ReLU(
                inplace=True))])
            self.fusion = nn.Sequential(nn.Conv2d(3 * 512, 512, kernel_size
                =3, padding=1, bias=False), norm_layer(512), nn.ReLU(
                inplace=True))
        extended_channels = 0
        self.atten = ACFModule(nheads, nmixs, inter_channels, 
            inter_channels // nheads * nmixs, inter_channels // nheads,
            norm_layer)
        if with_global:
            extended_channels = inter_channels
            self.atten_layers = ConcurrentModule([GlobalPooling(
                inter_channels, extended_channels, norm_layer, self.
                _up_kwargs), self.atten])
        else:
            self.atten_layers = nn.Sequential(*atten)
        if with_enc:
            self.encmodule = EncModule(inter_channels + extended_channels,
                out_channels, ncodes=32, se_loss=se_loss, norm_layer=norm_layer
                )
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv2d(
            inter_channels + extended_channels, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        feat = self.atten_layers(feat)
        if self.with_enc:
            outs = list(self.encmodule(feat))
        else:
            outs = [feat]
        outs[0] = self.conv6(outs[0])
        return tuple(outs)

    def demo(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        attn = self.atten.demo(feat)
        return attn


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1
    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(
        ), 'Intersection area should be smaller than Union area'
    return area_inter, area_union


up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, 'Correct area should be smaller than Labeled'
    return pixel_correct, pixel_labeled


def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnet152', root=
            root)), strict=False)
    return model


def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnet50', root=
            root)), strict=False)
    return model


def resnet50d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, stem_width=32,
        avg_down=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnet50d', root=
            root)), strict=False)
    return model


def resnext50_32x4d(pretrained=False, root='~/.encoding/models', **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnext50_32x4d',
            root=root)), strict=False)
    return model


def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], radix=2, groups=1,
        bottleneck_width=64, deep_stem=True, stem_width=32, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnest50', root=
            root)), strict=True)
    return model


def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], radix=2, groups=1,
        bottleneck_width=64, deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnest101', root=
            root)), strict=True)
    return model


def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnet101', root=
            root)), strict=False)
    return model


def xception65(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Xception65(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('xception65', root=
            root)))
    return model


def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], radix=2, groups=1,
        bottleneck_width=64, deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnest200', root=
            root)), strict=False)
    return model


def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8], radix=2, groups=1,
        bottleneck_width=64, deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnest269', root=
            root)), strict=True)
    return model


def resnet101s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-101 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnet101s', root=
            root)), strict=False)
    return model


def wideresnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a WideResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WideResNet([3, 3, 6, 6, 3, 1], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('wideresnet50',
            root=root)), strict=False)
    return model


def resnet152s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-152 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnet152s', root=
            root)), strict=False)
    return model


def resnext101_32x8d(pretrained=False, root='~/.encoding/models', **kwargs):
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['bottleneck_width'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('resnext101_32x8d',
            root=root)), strict=False)
    return model


def wideresnet38(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a WideResNet-38 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WideResNet([3, 3, 6, 3, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(get_model_file('wideresnet38',
            root=root)), strict=False)
    return model


def get_backbone(name, **kwargs):
    models = {'resnet50': resnet50, 'resnet101': resnet101, 'resnet152':
        resnet152, 'resnest50': resnest50, 'resnest101': resnest101,
        'resnest200': resnest200, 'resnest269': resnest269, 'resnet50s':
        resnet50s, 'resnet101s': resnet101s, 'resnet152s': resnet152s,
        'resnet50d': resnet50d, 'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d, 'xception65': xception65,
        'wideresnet38': wideresnet38, 'wideresnet50': wideresnet50}
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models
            .keys()))))
    net = models[name](**kwargs)
    return net


class BaseNet(nn.Module):

    def __init__(self, nclass, backbone, aux, se_loss, dilated=True,
        norm_layer=None, base_size=520, crop_size=480, mean=[0.485, 0.456, 
        0.406], std=[0.229, 0.224, 0.225], root='~/.encoding/models', *args,
        **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        self.backbone = backbone
        self.pretrained = get_backbone(backbone, *args, pretrained=True,
            dilated=dilated, norm_layer=norm_layer, root=root, **kwargs)
        self.pretrained.fc = None
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone.startswith('wideresnet'):
            x = self.pretrained.mod1(x)
            x = self.pretrained.pool2(x)
            x = self.pretrained.mod2(x)
            x = self.pretrained.pool3(x)
            x = self.pretrained.mod3(x)
            x = self.pretrained.mod4(x)
            x = self.pretrained.mod5(x)
            c3 = x.clone()
            x = self.pretrained.mod6(x)
            x = self.pretrained.mod7(x)
            x = self.pretrained.bn_out(x)
            return None, None, c3, x
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data,
            self.nclass)
        return correct, labeled, inter, union


def crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.size()
    assert c == 3
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        img_pad[:, (i), :, :] = F.pad(img[:, (i), :, :], (0, padw, 0, padh),
            value=pad_values[i])
    assert img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size
    return img_pad


def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def flip_image(img):
    assert img.dim() == 4
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3) - 1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        output += flip_image(foutput)
    return output.exp()


class MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""

    def __init__(self, module, nclass, device_ids=None, flip=True, scales=[
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(MultiEvalModule, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        None

    def parallel_forward(self, inputs, **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        inputs = [(input.unsqueeze(0),) for input, device in zip(inputs,
            self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    def forward(self, image):
        """Mult-size Evaluation"""
        batch, _, h, w = image.size()
        assert batch == 1
        stride_rate = 2.0 / 3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, self.nclass, h, w).zero_()
        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            cur_img = resize_image(image, height, width, **self.module.
                _up_kwargs)
            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.module.mean, self.module.
                    std, crop_size)
                outputs = module_inference(self.module, pad_img, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    pad_img = pad_image(cur_img, self.module.mean, self.
                        module.std, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.size()
                assert ph >= height and pw >= width
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch, self.nclass, ph, pw
                        ).zero_()
                    count_norm = image.new().resize_(batch, 1, ph, pw).zero_()
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        pad_crop_img = pad_image(crop_img, self.module.mean,
                            self.module.std, crop_size)
                        output = module_inference(self.module, pad_crop_img,
                            self.flip)
                        outputs[:, :, h0:h1, w0:w1] += crop_image(output, 0,
                            h1 - h0, 0, w1 - w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert (count_norm == 0).sum() == 0
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]
            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            scores += score
        return scores


class DeepLabV3Head(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs,
        atrous_rates=[12, 24, 36], **kwargs):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer,
            up_kwargs, **kwargs)
        self.block = nn.Sequential(nn.Conv2d(inter_channels, inter_channels,
            3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            True), nn.Dropout(0.1, False), nn.Conv2d(inter_channels,
            out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


class AsppPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=
        atrous_rate, dilation=atrous_rate, bias=False), norm_layer(
        out_channels), nn.ReLU(True))
    return block


class ASPP_Module(nn.Module):

    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels,
            out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU
            (True), nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class EncModule(nn.Module):

    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
        norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1,
            bias=False), norm_layer(in_channels), nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes), norm_layer(ncodes), nn.ReLU(
            inplace=True), Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.
            Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):

    def __init__(self, in_channels, out_channels, se_loss=True, lateral=
        True, norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, 512, 3, padding=1,
            bias=False), norm_layer(512), nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 512,
                kernel_size=1, bias=False), norm_layer(512), nn.ReLU(
                inplace=True)), nn.Sequential(nn.Conv2d(1024, 512,
                kernel_size=1, bias=False), norm_layer(512), nn.ReLU(
                inplace=True))])
            self.fusion = nn.Sequential(nn.Conv2d(3 * 512, 512, kernel_size
                =3, padding=1, bias=False), norm_layer(512), nn.ReLU(
                inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32, se_loss=
            se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv2d(512,
            out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


class FCFPNHead(nn.Module):

    def __init__(self, out_channels, norm_layer=None, fpn_inchannels=[256, 
        512, 1024, 2048], fpn_dim=256, up_kwargs=None):
        super(FCFPNHead, self).__init__()
        assert up_kwargs is not None
        self._up_kwargs = up_kwargs
        fpn_lateral = []
        for fpn_inchannel in fpn_inchannels[:-1]:
            fpn_lateral.append(nn.Sequential(nn.Conv2d(fpn_inchannel,
                fpn_dim, kernel_size=1, bias=False), norm_layer(fpn_dim),
                nn.ReLU(inplace=True)))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(fpn_inchannels) - 1):
            fpn_out.append(nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim,
                kernel_size=3, padding=1, bias=False), norm_layer(fpn_dim),
                nn.ReLU(inplace=True)))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.c4conv = nn.Sequential(nn.Conv2d(fpn_inchannels[-1], fpn_dim, 
            3, padding=1, bias=False), norm_layer(fpn_dim), nn.ReLU())
        inter_channels = len(fpn_inchannels) * fpn_dim
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, 512, 3,
            padding=1, bias=False), norm_layer(512), nn.ReLU(), nn.Dropout(
            0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        c4 = inputs[-1]
        if hasattr(self, 'extramodule'):
            c4 = self.extramodule(c4)
        feat = self.c4conv(c4)
        c1_size = inputs[0].size()[2:]
        feat_up = upsample(feat, c1_size, **self._up_kwargs)
        fpn_features = [feat_up]
        for i in reversed(range(len(inputs) - 1)):
            feat_i = self.fpn_lateral[i](inputs[i])
            feat = upsample(feat, feat_i.size()[2:], **self._up_kwargs)
            feat = feat + feat_i
            feat_up = upsample(self.fpn_out[i](feat), c1_size, **self.
                _up_kwargs)
            fpn_features.append(feat_up)
        fpn_features = torch.cat(fpn_features, 1)
        return self.conv5(fpn_features),


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GlobalPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            in_channels, out_channels, 1, bias=False), norm_layer(
            out_channels), nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return interpolate(pool, (h, w), **self._up_kwargs)


class FCNHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={},
        with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        if with_global:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels,
                inter_channels, 3, padding=1, bias=False), norm_layer(
                inter_channels), nn.ReLU(), ConcurrentModule([Identity(),
                GlobalPooling(inter_channels, inter_channels, norm_layer,
                self._up_kwargs)]), nn.Dropout(0.1, False), nn.Conv2d(2 *
                inter_channels, out_channels, 1))
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels,
                inter_channels, 3, padding=1, bias=False), norm_layer(
                inter_channels), nn.ReLU(), nn.Dropout(0.1, False), nn.
                Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class PSPHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer,
            up_kwargs), nn.Conv2d(in_channels * 2, inter_channels, 3,
            padding=1, bias=False), norm_layer(inter_channels), nn.ReLU(
            True), nn.Dropout(0.1, False), nn.Conv2d(inter_channels,
            out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class MixtureOfSoftMaxACF(nn.Module):
    """"Mixture of SoftMax"""

    def __init__(self, n_mix, d_k, attn_dropout=0.1):
        super(MixtureOfSoftMaxACF, self).__init__()
        self.temperature = np.power(d_k, 0.5)
        self.n_mix = n_mix
        self.att_drop = attn_dropout
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)
        self.d_k = d_k
        if n_mix > 1:
            self.weight = nn.Parameter(torch.Tensor(n_mix, d_k))
            std = np.power(n_mix, -0.5)
            self.weight.data.uniform_(-std, std)

    def forward(self, qt, kt, vt):
        B, d_k, N = qt.size()
        m = self.n_mix
        assert d_k == self.d_k
        d = d_k // m
        if m > 1:
            bar_qt = torch.mean(qt, 2, True)
            pi = self.softmax1(torch.matmul(self.weight, bar_qt)).view(B *
                m, 1, 1)
        q = qt.view(B * m, d, N).transpose(1, 2)
        N2 = kt.size(2)
        kt = kt.view(B * m, d, N2)
        v = vt.transpose(1, 2)
        attn = torch.bmm(q, kt)
        attn = attn / self.temperature
        attn = self.softmax2(attn)
        attn = self.dropout(attn)
        if m > 1:
            attn = (attn * pi).view(B, m, N, N2).sum(1)
        output = torch.bmm(attn, v)
        return output, attn


class ConvBnAct(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, radix=0, groups=1, bias=True, padding_mode=
        'zeros', rectify=False, rectify_avg=False, act=True, norm_layer=nn.
        BatchNorm2d):
        super().__init__()
        if radix > 0:
            conv_layer = SplAtConv2d
            conv_kwargs = {'radix': radix, 'rectify': rectify,
                'rectify_avg': rectify_avg, 'norm_layer': norm_layer}
        else:
            conv_layer = RFConv2d if rectify else nn.Conv2d
            conv_kwargs = {'average_mode': rectify_avg} if rectify else {}
        self.add_module('conv', conv_layer(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=
            padding_mode, **conv_kwargs))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        if act:
            self.add_module('relu', nn.ReLU())


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class GramMatrix(nn.Module):
    """ Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \\mathcal{G} = \\sum_{h=1}^{H_i}\\sum_{w=1}^{W_i} \\mathcal{F}_{h,w}\\mathcal{F}_{h,w}^T
    """

    def forward(self, y):
        b, ch, h, w = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class View(nn.Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """

    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(nn.Module):

    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(nn.Module):

    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(nn.Module):
    """Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \\frac{v}{\\max(\\lVert v \\rVert_p, \\epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\\lVert v \\rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """

    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-08)


class ConcurrentModule(nn.ModuleList):
    """Feed to a list of modules concurrently. 
    The outputs of the layers are concatenated at channel dimension.

    Args:
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, modules=None):
        super(ConcurrentModule, self).__init__(modules)

    def forward(self, x):
        outputs = []
        for layer in self:
            outputs.append(layer(x))
        return torch.cat(outputs, 1)


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), norm_layer(out_channels), nn.ReLU(True))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self.
            _up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self.
            _up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self.
            _up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self.
            _up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class DropBlock2D(nn.Module):
    """Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size, share_channel=False):
        super(DropBlock2D, self).__init__()
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.register_buffer('drop_prob', drop_prob * torch.ones(1, dtype=
            torch.float32))
        self.inited = False
        self.step_size = 0.0
        self.start_step = 0
        self.nr_steps = 0
        self.block_size = block_size
        self.share_channel = share_channel

    def reset(self):
        """stop DropBlock"""
        self.inited = True
        self.i[0] = 0
        self.drop_prob = 0.0

    def reset_steps(self, start_step, nr_steps, start_value=0, stop_value=None
        ):
        self.inited = True
        stop_value = self.drop_prob.item(
            ) if stop_value is None else stop_value
        self.i[0] = 0
        self.drop_prob[0] = start_value
        self.step_size = (stop_value - start_value) / nr_steps
        self.nr_steps = nr_steps
        self.start_step = start_step

    def forward(self, x):
        if not self.training or self.drop_prob.item() == 0.0:
            return x
        else:
            self.step()
            gamma = self._compute_gamma(x)
            if self.share_channel:
                mask = (torch.rand(*x.shape[2:], device=x.device, dtype=x.
                    dtype) < gamma).unsqueeze(0).unsqueeze(0)
            else:
                mask = (torch.rand(*x.shape[1:], device=x.device, dtype=x.
                    dtype) < gamma).unsqueeze(0)
            block_mask, keeped = self._compute_block_mask(mask)
            out = x * block_mask
            out = out * (block_mask.numel() / keeped).to(out)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(mask, kernel_size=(self.block_size, self.
            block_size), stride=(1, 1), padding=self.block_size // 2)
        keeped = block_mask.numel() - block_mask.sum().to(torch.float32)
        block_mask = 1 - block_mask
        return block_mask, keeped

    def _compute_gamma(self, x):
        _, c, h, w = x.size()
        gamma = self.drop_prob.item() / self.block_size ** 2 * (h * w) / ((
            w - self.block_size + 1) * (h - self.block_size + 1))
        return gamma

    def step(self):
        assert self.inited
        idx = self.i.item()
        if idx > self.start_step and idx < self.start_step + self.nr_steps:
            self.drop_prob += self.step_size
        self.i += 1

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        idx_key = prefix + 'i'
        drop_prob_key = prefix + 'drop_prob'
        if idx_key not in state_dict:
            state_dict[idx_key] = torch.zeros(1, dtype=torch.int64)
        if idx_key not in drop_prob_key:
            state_dict[drop_prob_key] = torch.ones(1, dtype=torch.float32)
        super(DropBlock2D, self)._load_from_state_dict(state_dict, prefix,
            local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """overwrite save method"""
        pass

    def extra_repr(self):
        return 'drop_prob={}, step_size={}'.format(self.drop_prob, self.
            step_size)


class _aggregate(Function):

    @staticmethod
    def forward(ctx, A, X, C):
        ctx.save_for_backward(A, X, C)
        if A.is_cuda:
            E = lib.gpu.aggregate_forward(A, X, C)
        else:
            E = lib.cpu.aggregate_forward(A, X, C)
        return E

    @staticmethod
    def backward(ctx, gradE):
        A, X, C = ctx.saved_variables
        if A.is_cuda:
            gradA, gradX, gradC = lib.gpu.aggregate_backward(gradE, A, X, C)
        else:
            gradA, gradX, gradC = lib.cpu.aggregate_backward(gradE, A, X, C)
        return gradA, gradX, gradC


def aggregate(A, X, C):
    """ Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
    to the codewords (:math:`C`) with assignment weights (:math:`A`).

    .. math::

        e_{k} = \\sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\\in\\mathcal{R}^{B\\times N\\times K}`
          :math:`X\\in\\mathcal{R}^{B\\times N\\times D}` :math:`C\\in\\mathcal{R}^{K\\times D}`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\\in\\mathcal{R}^{B\\times K\\times D}`

    Examples:
        >>> B,N,K,D = 2,3,4,5
        >>> A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
        >>> X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> func = encoding.aggregate()
        >>> E = func(A, X, C)
    """
    return _aggregate.apply(A, X, C)


class _scaled_l2(Function):

    @staticmethod
    def forward(ctx, X, C, S):
        if X.is_cuda:
            SL = lib.gpu.scaled_l2_forward(X, C, S)
        else:
            SL = lib.cpu.scaled_l2_forward(X, C, S)
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, gradSL):
        X, C, S, SL = ctx.saved_variables
        if X.is_cuda:
            gradX, gradC, gradS = lib.gpu.scaled_l2_backward(gradSL, X, C,
                S, SL)
        else:
            gradX, gradC, gradS = lib.cpu.scaled_l2_backward(gradSL, X, C,
                S, SL)
        return gradX, gradC, gradS


def scaled_l2(X, C, S):
    """ scaled_l2 distance

    .. math::
        sl_{ik} = s_k \\|x_i-c_k\\|^2

    Shape:
        - Input: :math:`X\\in\\mathcal{R}^{B\\times N\\times D}`
          :math:`C\\in\\mathcal{R}^{K\\times D}` :math:`S\\in \\mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\\in\\mathcal{R}^{B\\times N\\times K}`
    """
    return _scaled_l2.apply(X, C, S)


class Encoding(Module):
    """
    Encoding Layer: a learnable residual encoder.

    .. image:: _static/img/cvpr17.svg
        :width: 30%
        :align: center

    Encoding Layer accpets 3D or 4D inputs.
    It considers an input featuremaps with the shape of :math:`C\\times H\\times W`
    as a set of C-dimentional input features :math:`X=\\{x_1, ...x_N\\}`, where N is total number
    of features given by :math:`H\\times W`, which learns an inherent codebook
    :math:`D=\\{d_1,...d_K\\}` and a set of smoothing factor of visual centers
    :math:`S=\\{s_1,...s_K\\}`. Encoding Layer outputs the residuals with soft-assignment weights
    :math:`e_k=\\sum_{i=1}^Ne_{ik}`, where

    .. math::

        e_{ik} = \\frac{exp(-s_k\\|r_{ik}\\|^2)}{\\sum_{j=1}^K exp(-s_j\\|r_{ij}\\|^2)} r_{ik}

    and the residuals are given by :math:`r_{ik} = x_i - d_k`. The output encoders are
    :math:`E=\\{e_1,...e_K\\}`.

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\\in\\mathcal{R}^{B\\times N\\times D}` or
          :math:`\\mathcal{R}^{B\\times D\\times H\\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\\times W`.)
        - Output: :math:`E\\in\\mathcal{R}^{B\\times K\\times D}`

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)
    """

    def __init__(self, D, K):
        super(Encoding, self).__init__()
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1.0 / (self.K * self.D) ** (1 / 2)
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        assert X.size(1) == self.D
        B, D = X.size(0), self.D
        if X.dim() == 3:
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.D
            ) + '=>' + str(self.K) + 'x' + str(self.D) + ')'


class EncodingDrop(Module):
    """Dropout regularized Encoding Layer.
    """

    def __init__(self, D, K):
        super(EncodingDrop, self).__init__()
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1.0 / (self.K * self.D) ** (1 / 2)
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def _drop(self):
        if self.training:
            self.scale.data.uniform_(-1, 0)
        else:
            self.scale.data.zero_().add_(-0.5)

    def forward(self, X):
        assert X.size(1) == self.D
        if X.dim() == 3:
            B, D = X.size(0), self.D
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            B, D = X.size(0), self.D
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        self._drop()
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        E = aggregate(A, X, self.codewords)
        self._drop()
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.D
            ) + '=>' + str(self.K) + 'x' + str(self.D) + ')'


class Inspiration(Module):
    """
    Inspiration Layer (CoMatch Layer) enables the multi-style transfer in feed-forward
    network, which learns to match the target feature statistics during the training.
    This module is differentialble and can be inserted in standard feed-forward network
    to be learned directly from the loss function without additional supervision.

    .. math::
        Y = \\phi^{-1}[\\phi(\\mathcal{F}^T)W\\mathcal{G}]

    Please see the `example of MSG-Net <./experiments/style.html>`_
    training multi-style generative network for real-time transfer.

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."
        *arXiv preprint arXiv:1703.06953 (2017)*
    """

    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        self.weight = Parameter(torch.Tensor(1, C, C), requires_grad=True)
        self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C,
            self.C), X.view(X.size(0), X.size(1), -1)).view_as(X)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.C) + ')'


class UpsampleConv2d(Module):
    """
    To avoid the checkerboard artifacts of standard Fractionally-strided Convolution,
    we adapt an integer stride convolution but producing a :math:`2\\times 2` outputs for
    each convolutional window.

    .. image:: _static/img/upconv.png
        :width: 50%
        :align: center

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."
        *arXiv preprint arXiv:1703.06953 (2017)*

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Zero-padding added to one side of the output.
          Default: 0
        groups (int, optional): Number of blocked connections from input channels to output
          channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        scale_factor (int): scaling factor for upsampling convolution. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = scale * (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\\_size[0] + output\\_padding[0]`
          :math:`W_{out} = scale * (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\\_size[1] + output\\_padding[1]`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, scale * scale * out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (scale * scale * out_channels)

    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.UpsampleCov2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.UpsampleCov2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.UpsampleCov2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, scale_factor=1, bias=True):
        super(UpsampleConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale_factor = scale_factor
        self.weight = Parameter(torch.Tensor(out_channels * scale_factor *
            scale_factor, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels * scale_factor *
                scale_factor))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)
        return F.pixel_shuffle(out, self.scale_factor)


def pairwise_cosine(X, C, normalize=False):
    """Pairwise Cosine Similarity or Dot-product Similarity
    Shape:
        - Input: :math:`X\\in\\mathcal{R}^{B\\times N\\times D}`
          :math:`C\\in\\mathcal{R}^{K\\times D}` :math:`S\\in \\mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\\in\\mathcal{R}^{B\\times N\\times K}`
    """
    if normalize:
        X = F.normalize(X, dim=2, eps=1e-08)
        C = F.normalize(C, dim=1, eps=1e-08)
    return torch.matmul(X, C.t())


class EncodingCosine(Module):

    def __init__(self, D, K):
        super(EncodingCosine, self).__init__()
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1.0 / (self.K * self.D) ** (1 / 2)
        self.codewords.data.uniform_(-std1, std1)

    def forward(self, X):
        assert X.size(1) == self.D
        if X.dim() == 3:
            B, D = X.size(0), self.D
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            B, D = X.size(0), self.D
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        L = pairwise_cosine(X, self.codewords)
        A = F.softmax(L, dim=2)
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.D
            ) + '=>' + str(self.K) + 'x' + str(self.D) + ')'


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class NLLMultiLabelSmooth(nn.Module):

    def __init__(self, smoothing=0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, aux=False,
        aux_weight=0.4, weight=None, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass
                ).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), bins=nclass,
                min=0, max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class _rectify(Function):

    @staticmethod
    def forward(ctx, y, x, kernel_size, stride, padding, dilation, average):
        ctx.save_for_backward(x)
        kernel_size = [(k + 2 * (d - 1)) for k, d in zip(kernel_size, dilation)
            ]
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.average = average
        if x.is_cuda:
            lib.gpu.conv_rectify(y, x, kernel_size, stride, padding,
                dilation, average)
        else:
            lib.cpu.conv_rectify(y, x, kernel_size, stride, padding,
                dilation, average)
        ctx.mark_dirty(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_variables
        if x.is_cuda:
            lib.gpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
                ctx.padding, ctx.dilation, ctx.average)
        else:
            lib.cpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
                ctx.padding, ctx.dilation, ctx.average)
        ctx.mark_dirty(grad_y)
        return grad_y, None, None, None, None, None, None


rectify = _rectify.apply


class RFConv2d(Conv2d):
    """Rectified Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
        average_mode=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.rectify = average_mode or (padding[0] > 0 or padding[1] > 0)
        self.average = average_mode
        super(RFConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode
                =self.padding_mode), weight, self.bias, self.stride, _pair(
                0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)

    def forward(self, input):
        output = self._conv_forward(input, self.weight)
        if self.rectify:
            output = rectify(output, input, self.kernel_size, self.stride,
                self.padding, self.dilation, self.average)
        return output

    def extra_repr(self):
        return super().extra_repr() + ', rectify={}, average_mode={}'.format(
            self.rectify, self.average)


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
        self.bn0 = norm_layer(channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
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
        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel // self.radix, dim=1)
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
            atten = torch.split(atten, channel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(atten, splited)])
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


class dist_syncbatchnorm_(Function):

    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var, eps,
        momentum, training, process_group):
        x = x.contiguous()
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.process_group = process_group
        if not ctx.training:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2
            if x.is_cuda:
                y = lib.gpu.batchnorm_forward(x, _ex, _exs, gamma, beta,
                    ctx.eps)
            else:
                y = lib.cpu.batchnorm_forward(x, _ex, _exs, gamma, beta,
                    ctx.eps)
            ctx.save_for_backward(x, _ex, _exs, gamma, beta)
            return y
        size = x.numel() // x.size(1)
        if size == 1:
            raise ValueError(
                'Expected more than 1 value per channel when training, got input size {}'
                .format(size))
        if x.is_cuda:
            _ex, _exs = lib.gpu.expectation_forward(x)
        else:
            raise NotImplemented
        count = torch.Tensor([1]).to(x.device)
        count_all_reduce = torch.distributed.all_reduce(count, group=
            process_group, async_op=True)
        _ex_all_reduce = torch.distributed.all_reduce(_ex, group=
            process_group, async_op=True)
        _exs_all_reduce = torch.distributed.all_reduce(_exs, group=
            process_group, async_op=True)
        count_all_reduce.wait()
        _ex_all_reduce.wait()
        _exs_all_reduce.wait()
        _ex = _ex / count
        _exs = _exs / count
        _var = _exs - _ex ** 2
        running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * _ex)
        running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * _var)
        ctx.mark_dirty(running_mean, running_var)
        if x.is_cuda:
            y = lib.gpu.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            y = lib.cpu.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = lib.gpu.batchnorm_backward(dz,
                x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented
        if ctx.training:
            process_group = ctx.process_group
            count = torch.Tensor([1]).to(x.device)
            count_all_reduce = torch.distributed.all_reduce(count, group=
                process_group, async_op=True)
            _dex_all_reduce = torch.distributed.all_reduce(_dex, group=
                process_group, async_op=True)
            _dexs_all_reduce = torch.distributed.all_reduce(_dexs, group=
                process_group, async_op=True)
            count_all_reduce.wait()
            _dex_all_reduce.wait()
            _dexs_all_reduce.wait()
            _dex = _dex / count
            _dexs = _dexs / count
            if x.is_cuda:
                dx_ = lib.gpu.expectation_backward(x, _dex, _dexs)
            else:
                raise NotImplemented
            dx = dx + dx_
        return dx, dgamma, dbeta, None, None, None, None, None, None


dist_syncbatchnorm = dist_syncbatchnorm_.apply


class DistSyncBatchNorm(_BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*

    Examples:
        >>> m = DistSyncBatchNorm(100)
        >>> net = torch.nn.parallel.DistributedDataParallel(m)
        >>> output = net(input)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, process_group
        =None):
        super(DistSyncBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=True, track_running_stats=True)
        self.process_group = process_group

    def forward(self, x):
        need_sync = self.training or not self.track_running_stats
        process_group = None
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        y = dist_syncbatchnorm(x, self.weight, self.bias, self.running_mean,
            self.running_var, self.eps, self.momentum, self.training,
            process_group)
        return y.view(input_shape)


def _act_backward(ctx, x, dx):
    if ctx.activation.lower() == 'leaky_relu':
        if x.is_cuda:
            lib.gpu.leaky_relu_backward(x, dx, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert activation == 'none'


def _act_forward(ctx, x):
    if ctx.activation.lower() == 'leaky_relu':
        if x.is_cuda:
            lib.gpu.leaky_relu_forward(x, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert activation == 'none'


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        return modules


torch_ver = torch.__version__[:3]


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    if torch_ver != '0.3':
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != '0.3':
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhanghang1989_PyTorch_Encoding(_paritybench_base):
    pass
    def test_000(self):
        self._check(ConvBnAct(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DropBlock2D(*[], **{'drop_prob': 4, 'block_size': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(GramMatrix(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Inspiration(*[], **{'C': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(LabelSmoothing(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_007(self):
        self._check(Mean(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(MixtureOfSoftMaxACF(*[], **{'n_mix': 4, 'd_k': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(Normalize(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(RFConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(Sum(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4, 4])], {})

    def test_012(self):
        self._check(UpsampleConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(rSoftMax(*[], **{'radix': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

