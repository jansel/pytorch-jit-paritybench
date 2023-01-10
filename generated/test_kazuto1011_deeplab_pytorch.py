import sys
_module = sys.modules[__name__]
del sys
convert = _module
demo = _module
hubconf = _module
libs = _module
caffe_pb2 = _module
datasets = _module
base = _module
cocostuff = _module
voc = _module
models = _module
deeplabv1 = _module
deeplabv2 = _module
deeplabv3 = _module
deeplabv3plus = _module
msc = _module
resnet = _module
utils = _module
crf = _module
lr_scheduler = _module
metric = _module
main = _module

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


import re


from collections import Counter


from collections import OrderedDict


import numpy as np


import torch


import matplotlib


import matplotlib.cm as cm


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


from torch.hub import load_state_dict_from_url


import random


from torch.utils import data


import scipy.io as sio


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.tensorboard import SummaryWriter


_BOTTLENECK_EXPANSION = 4


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False) if downsample else nn.Identity()

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [(1) for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)
        for i in range(n_layers):
            self.add_module('block{}'.format(i + 1), _Bottleneck(in_ch=in_ch if i == 0 else out_ch, out_ch=out_ch, stride=stride if i == 0 else 1, dilation=dilation * multi_grids[i], downsample=True if i == 0 else False))


class _Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(_Stem, self).__init__()
        self.add_module('conv1', _ConvBnReLU(3, out_ch, 7, 2, 3, 1))
        self.add_module('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class DeepLabV1(nn.Sequential):
    """
    DeepLab v1: Dilated ResNet + 1x1 Conv
    Note that this is just a container for loading the pretrained COCO model and not mentioned as "v1" in papers.
    """

    def __init__(self, n_classes, n_blocks):
        super(DeepLabV1, self).__init__()
        ch = [(64 * 2 ** p) for p in range(6)]
        self.add_module('layer1', _Stem(ch[0]))
        self.add_module('layer2', _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module('layer3', _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module('layer4', _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module('layer5', _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module('fc', nn.Conv2d(2048, n_classes, 1))


class _ImagePool(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode='bilinear', align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0', _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module('c{}'.format(i + 1), _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate))
        self.stages.add_module('imagepool', _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [(64 * 2 ** p) for p in range(6)]
        self.add_module('layer1', _Stem(ch[0]))
        self.add_module('layer2', _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module('layer3', _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module('layer4', _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module('layer5', _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module('aspp', _ASPP(ch[5], n_classes, atrous_rates))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


class DeepLabV3(nn.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3, self).__init__()
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        ch = [(64 * 2 ** p) for p in range(6)]
        self.add_module('layer1', _Stem(ch[0]))
        self.add_module('layer2', _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module('layer3', _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module('layer4', _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module('layer5', _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids))
        self.add_module('aspp', _ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module('fc1', _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module('fc2', nn.Conv2d(256, n_classes, kernel_size=1))


class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3Plus, self).__init__()
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        ch = [(64 * 2 ** p) for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[5], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module('fc1', _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.reduce = _ConvBnReLU(256, 48, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(OrderedDict([('conv1', _ConvBnReLU(304, 256, 3, 1, 1, 1)), ('conv2', _ConvBnReLU(256, 256, 3, 1, 1, 1)), ('conv3', nn.Conv2d(256, n_classes, kernel_size=1))]))

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h_ = self.reduce(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)
        h = self.fc1(h)
        h = F.interpolate(h, size=h_.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
        return h


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        logits = self.base(x)
        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(l, size=(H, W), mode='bilinear', align_corners=False)
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode='bilinear', align_corners=False)
            logits_pyramid.append(self.base(h))
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max


class ResNet(nn.Sequential):

    def __init__(self, n_classes, n_blocks):
        super(ResNet, self).__init__()
        ch = [(64 * 2 ** p) for p in range(6)]
        self.add_module('layer1', _Stem(ch[0]))
        self.add_module('layer2', _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module('layer3', _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module('layer4', _ResLayer(n_blocks[2], ch[3], ch[4], 2, 1))
        self.add_module('layer5', _ResLayer(n_blocks[3], ch[4], ch[5], 2, 1))
        self.add_module('pool5', nn.AdaptiveAvgPool2d(1))
        self.add_module('flatten', nn.Flatten())
        self.add_module('fc', nn.Linear(ch[5], n_classes))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MSC,
     lambda: ([], {'base': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kazuto1011_deeplab_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

