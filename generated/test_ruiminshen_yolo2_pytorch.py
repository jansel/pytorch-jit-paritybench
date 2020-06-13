import sys
_module = sys.modules[__name__]
del sys
benchmark_caffe2 = _module
cache = _module
coco = _module
voc = _module
checksum_caffe2 = _module
checksum_torch = _module
eval = _module
convert_darknet_torch = _module
convert_onnx_caffe2 = _module
convert_torch_onnx = _module
demo_data = _module
demo_graph = _module
demo_lr = _module
detect = _module
dimension_cluster = _module
disable_bad_images = _module
download_url = _module
eval = _module
model = _module
densenet = _module
inception3 = _module
inception4 = _module
mobilenet = _module
resnet = _module
vgg = _module
yolo2 = _module
pruner = _module
receptive_field_analyzer = _module
split_data = _module
train = _module
transform = _module
augmentation = _module
image = _module
resize = _module
label = _module
channel = _module
data = _module
iou = _module
numpy = _module
torch = _module
postprocess = _module
visualize = _module
variable_stat = _module
video2image = _module

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


import logging.config


import numpy as np


import torch.autograd


import torch.cuda


import torch.optim


import torch.utils.data


import torch.nn.functional as F


import inspect


import torch


import torch.nn as nn


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


import scipy.stats as stats


import torch.utils.model_zoo


import collections.abc


import collections


import re


def meshgrid(rows, cols, swap=False):
    i = torch.arange(0, rows).repeat(cols).view(-1, 1)
    j = torch.arange(0, cols).view(-1, 1).repeat(1, rows).view(-1, 1)
    return torch.cat([i, j], 1) if swap else torch.cat([j, i], 1)


class Inference(nn.Module):

    def __init__(self, config, dnn, anchors):
        nn.Module.__init__(self)
        self.config = config
        self.dnn = dnn
        self.anchors = anchors

    def forward(self, x):
        device_id = x.get_device() if torch.cuda.is_available() else None
        feature = self.dnn(x)
        rows, cols = feature.size()[-2:]
        cells = rows * cols
        _feature = feature.permute(0, 2, 3, 1).contiguous().view(feature.
            size(0), cells, self.anchors.size(0), -1)
        sigmoid = F.sigmoid(_feature[:, :, :, :3])
        iou = sigmoid[:, :, :, (0)]
        ij = torch.autograd.Variable(utils.ensure_device(meshgrid(rows,
            cols).view(1, -1, 1, 2), device_id))
        center_offset = sigmoid[:, :, :, 1:3]
        center = ij + center_offset
        size_norm = _feature[:, :, :, 3:5]
        anchors = torch.autograd.Variable(utils.ensure_device(self.anchors.
            view(1, 1, -1, 2), device_id))
        size = torch.exp(size_norm) * anchors
        size2 = size / 2
        yx_min = center - size2
        yx_max = center + size2
        logits = _feature[:, :, :, 5:] if _feature.size(-1) > 5 else None
        return feature, iou, center_offset, size_norm, yx_min, yx_max, logits


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
        stride=1, bn=True, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in
                    kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1,
            affine=True) if bn else lambda x: x
        self.act = nn.ReLU(inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = Conv2d(config_channels.channels, config_channels(int(96 *
            ratio), '%s.conv.conv.weight' % prefix), kernel_size=3, stride=
            2, bn=bn)
        config_channels.channels = channels + self.conv.conv.weight.size(0)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            64 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            96 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=3, stride=1, bn=bn))
        self.branch0 = nn.Sequential(*branch)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            96 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=(3, 3), stride=1, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        config_channels.channels = self.branch0[-1].conv.weight.size(0
            ) + self.branch1[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.conv = Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.conv.conv.weight' % prefix), kernel_size=3,
            stride=2, bn=bn)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        config_channels.channels = self.conv.conv.weight.size(0) + channels

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int
            (96 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=1,
            stride=1, bn=bn)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            64 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            96 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=3, stride=1, padding=1, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            64 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            96 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=3, stride=1, padding=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            96 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=3, stride=1, padding=1, bn=bn))
        self.branch2 = nn.Sequential(*branch)
        config_channels.channels = channels
        branch = []
        branch.append(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            96 * ratio), '%s.branch3.%d.conv.weight' % (prefix, len(branch)
            )), kernel_size=1, stride=1, bn=bn))
        self.branch3 = nn.Sequential(*branch)
        config_channels.channels = self.branch0.conv.weight.size(0
            ) + self.branch1[-1].conv.weight.size(0) + self.branch2[-1
            ].conv.weight.size(0) + self.branch3[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int
            (384 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=
            3, stride=2, bn=bn)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            224 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=3, stride=1, padding=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=3, stride=2, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        self.branch2 = nn.MaxPool2d(3, stride=2)
        config_channels.channels = self.branch0.conv.weight.size(0
            ) + self.branch1[-1].conv.weight.size(0) + channels

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int
            (384 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=
            1, stride=1, bn=bn)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            224 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        self.branch1 = nn.Sequential(*branch)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            224 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            224 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            256 * ratio), '%s.branch2.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        self.branch2 = nn.Sequential(*branch)
        config_channels.channels = channels
        branch = []
        branch.append(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            128 * ratio), '%s.branch3.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=1, stride=1, bn=bn))
        self.branch3 = nn.Sequential(*branch)
        config_channels.channels = self.branch0.conv.weight.size(0
            ) + self.branch1[-1].conv.weight.size(0) + self.branch2[-1
            ].conv.weight.size(0) + self.branch3[-1].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            192 * ratio), '%s.branch0.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=3, stride=2, bn=bn))
        self.branch0 = nn.Sequential(*branch)
        config_channels.channels = channels
        branch = []
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=1, stride=1, bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            256 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(1, 7), stride=1, padding=(0, 3), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            320 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=(7, 1), stride=1, padding=(3, 0), bn=bn))
        branch.append(Conv2d(config_channels.channels, config_channels(int(
            320 * ratio), '%s.branch1.%d.conv.weight' % (prefix, len(branch
            ))), kernel_size=3, stride=2, bn=bn))
        self.branch1 = nn.Sequential(*branch)
        self.branch2 = nn.MaxPool2d(3, stride=2)
        config_channels.channels = self.branch0[-1].conv.weight.size(0
            ) + self.branch1[-1].conv.weight.size(0) + channels

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self, config_channels, prefix, bn=True, ratio=1):
        nn.Module.__init__(self)
        channels = config_channels.channels
        self.branch0 = Conv2d(config_channels.channels, config_channels(int
            (256 * ratio), '%s.branch0.conv.weight' % prefix), kernel_size=
            1, stride=1, bn=bn)
        config_channels.channels = channels
        self.branch1_0 = Conv2d(config_channels.channels, config_channels(
            int(384 * ratio), '%s.branch1_0.conv.weight' % prefix),
            kernel_size=1, stride=1, bn=bn)
        _channels = config_channels.channels
        self.branch1_1a = Conv2d(_channels, config_channels(int(256 * ratio
            ), '%s.branch1_1a.conv.weight' % prefix), kernel_size=(1, 3),
            stride=1, padding=(0, 1), bn=bn)
        self.branch1_1b = Conv2d(_channels, config_channels(int(256 * ratio
            ), '%s.branch1_1b.conv.weight' % prefix), kernel_size=(3, 1),
            stride=1, padding=(1, 0), bn=bn)
        config_channels.channels = channels
        self.branch2_0 = Conv2d(config_channels.channels, config_channels(
            int(384 * ratio), '%s.branch2_0.conv.weight' % prefix),
            kernel_size=1, stride=1, bn=bn)
        self.branch2_1 = Conv2d(config_channels.channels, config_channels(
            int(448 * ratio), '%s.branch2_1.conv.weight' % prefix),
            kernel_size=(3, 1), stride=1, padding=(1, 0), bn=bn)
        self.branch2_2 = Conv2d(config_channels.channels, config_channels(
            int(512 * ratio), '%s.branch2_2.conv.weight' % prefix),
            kernel_size=(1, 3), stride=1, padding=(0, 1), bn=bn)
        _channels = config_channels.channels
        self.branch2_3a = Conv2d(_channels, config_channels(int(256 * ratio
            ), '%s.branch2_3a.conv.weight' % prefix), kernel_size=(1, 3),
            stride=1, padding=(0, 1), bn=bn)
        self.branch2_3b = Conv2d(_channels, config_channels(int(256 * ratio
            ), '%s.branch2_3b.conv.weight' % prefix), kernel_size=(3, 1),
            stride=1, padding=(1, 0), bn=bn)
        config_channels.channels = channels
        branch = []
        branch.append(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False))
        branch.append(Conv2d(config_channels.channels, int(256 * ratio),
            kernel_size=1, stride=1, bn=bn))
        self.branch3 = nn.Sequential(*branch)
        config_channels.channels = self.branch0.conv.weight.size(0
            ) + self.branch1_1a.conv.weight.size(0
            ) + self.branch1_1b.conv.weight.size(0
            ) + self.branch2_3a.conv.weight.size(0
            ) + self.branch2_3b.conv.weight.size(0) + self.branch3[-1
            ].conv.weight.size(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception4(nn.Module):

    def __init__(self, config_channels, anchors, num_cls, ratio=1):
        nn.Module.__init__(self)
        features = []
        bn = config_channels.config.getboolean('batch_norm', 'enable')
        features.append(Conv2d(config_channels.channels, config_channels(32,
            'features.%d.conv.weight' % len(features)), kernel_size=3,
            stride=2, bn=bn))
        features.append(Conv2d(config_channels.channels, config_channels(32,
            'features.%d.conv.weight' % len(features)), kernel_size=3,
            stride=1, bn=bn))
        features.append(Conv2d(config_channels.channels, config_channels(64,
            'features.%d.conv.weight' % len(features)), kernel_size=3,
            stride=1, padding=1, bn=bn))
        features.append(Mixed_3a(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Mixed_4a(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Mixed_5a(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_A(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Reduction_A(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Reduction_B(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_C(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_C(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(Inception_C(config_channels, 'features.%d' % len(
            features), bn=bn, ratio=ratio))
        features.append(nn.Conv2d(config_channels.channels, model.
            output_channels(len(anchors), num_cls), 1))
        self.features = nn.Sequential(*features)
        self.init(config_channels)

    def init(self, config_channels):
        try:
            gamma = config_channels.config.getboolean('batch_norm', 'gamma')
        except (configparser.NoSectionError, configparser.NoOptionError):
            gamma = True
        try:
            beta = config_channels.config.getboolean('batch_norm', 'beta')
        except (configparser.NoSectionError, configparser.NoOptionError):
            beta = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.weight.requires_grad = gamma
                m.bias.requires_grad = beta
        try:
            if config_channels.config.getboolean('model', 'pretrained'):
                settings = pretrained_settings['inceptionv4'][
                    config_channels.config.get('inception4', 'pretrained')]
                logging.info('use pretrained model: ' + str(settings))
                state_dict = self.state_dict()
                for key, value in torch.utils.model_zoo.load_url(settings[
                    'url']).items():
                    if key in state_dict:
                        state_dict[key] = value
                self.load_state_dict(state_dict)
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass

    def forward(self, x):
        return self.features(x)

    def scope(self, name):
        return '.'.join(name.split('.')[:-2])


def conv_dw(in_channels, stride):
    return nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(
        in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=
        False)), ('bn', nn.BatchNorm2d(in_channels)), ('act', nn.ReLU(
        inplace=True))]))


def conv_pw(in_channels, out_channels):
    return nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(
        in_channels, out_channels, 1, 1, 0, bias=False)), ('bn', nn.
        BatchNorm2d(out_channels)), ('act', nn.ReLU(inplace=True))]))


def conv_unit(in_channels, out_channels, stride):
    return nn.Sequential(collections.OrderedDict([('dw', conv_dw(
        in_channels, stride)), ('pw', conv_pw(in_channels, out_channels))]))


def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(collections.OrderedDict([('conv', nn.Conv2d(
        in_channels, out_channels, 3, stride, 1, bias=False)), ('bn', nn.
        BatchNorm2d(out_channels)), ('act', nn.ReLU(inplace=True))]))


class MobileNet(nn.Module):

    def __init__(self, config_channels, anchors, num_cls):
        nn.Module.__init__(self)
        layers = []
        layers.append(conv_bn(config_channels.channels, config_channels(32,
            'layers.%d.conv.weight' % len(layers)), 2))
        layers.append(conv_unit(config_channels.channels, config_channels(
            64, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            128, 'layers.%d.pw.conv.weight' % len(layers)), 2))
        layers.append(conv_unit(config_channels.channels, config_channels(
            128, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            256, 'layers.%d.pw.conv.weight' % len(layers)), 2))
        layers.append(conv_unit(config_channels.channels, config_channels(
            256, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            512, 'layers.%d.pw.conv.weight' % len(layers)), 2))
        layers.append(conv_unit(config_channels.channels, config_channels(
            512, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            512, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            512, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            512, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            512, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(conv_unit(config_channels.channels, config_channels(
            1024, 'layers.%d.pw.conv.weight' % len(layers)), 2))
        layers.append(conv_unit(config_channels.channels, config_channels(
            1024, 'layers.%d.pw.conv.weight' % len(layers)), 1))
        layers.append(nn.Conv2d(config_channels.channels, model.
            output_channels(len(anchors), num_cls), 1))
        self.layers = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Module):

    def __init__(self, config_channels, prefix, channels, stride=1):
        nn.Module.__init__(self)
        channels_in = config_channels.channels
        self.conv1 = conv3x3(config_channels.channels, config_channels(
            channels, '%s.conv1.weight' % prefix), stride)
        self.bn1 = nn.BatchNorm2d(config_channels.channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(config_channels.channels, config_channels(
            channels, '%s.conv2.weight' % prefix))
        self.bn2 = nn.BatchNorm2d(config_channels.channels)
        if stride > 1 or channels_in != config_channels.channels:
            downsample = []
            downsample.append(nn.Conv2d(channels_in, config_channels.
                channels, kernel_size=1, stride=stride, bias=False))
            downsample.append(nn.BatchNorm2d(config_channels.channels))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

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

    def __init__(self, config_channels, prefix, channels, stride=1):
        nn.Module.__init__(self)
        channels_in = config_channels.channels
        self.conv1 = nn.Conv2d(config_channels.channels, config_channels(
            channels, '%s.conv1.weight' % prefix), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(config_channels.channels)
        self.conv2 = nn.Conv2d(config_channels.channels, config_channels(
            channels, '%s.conv2.weight' % prefix), kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(config_channels.channels)
        self.conv3 = nn.Conv2d(config_channels.channels, config_channels(
            channels * 4, '%s.conv3.weight' % prefix), kernel_size=1, bias=
            False)
        self.bn3 = nn.BatchNorm2d(config_channels.channels)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or channels_in != config_channels.channels:
            downsample = []
            downsample.append(nn.Conv2d(channels_in, config_channels.
                channels, kernel_size=1, stride=stride, bias=False))
            downsample.append(nn.BatchNorm2d(config_channels.channels))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

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


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
        stride=1, bn=True, act=True):
        nn.Module.__init__(self)
        if isinstance(padding, bool):
            if isinstance(kernel_size, collections.abc.Iterable):
                padding = tuple((kernel_size - 1) // 2 for kernel_size in
                    kernel_size) if padding else 0
            else:
                padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01
            ) if bn else lambda x: x
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def reorg(x, stride_h=2, stride_w=2):
    batch_size, channels, height, width = x.size()
    _height, _width = height // stride_h, width // stride_w
    if 1:
        x = x.view(batch_size, channels, _height, stride_h, _width, stride_w
            ).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, stride_h * stride_w
            ).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, stride_h * stride_w, _height, _width
            ).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)
    else:
        x = x.view(batch_size, channels, _height, stride_h, _width, stride_w)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.contiguous()
        x = x.view(batch_size, -1, _height, _width)
    return x


class Darknet(nn.Module):

    def __init__(self, config_channels, anchors, num_cls, stride=2, ratio=1):
        nn.Module.__init__(self)
        self.stride = stride
        channels = int(32 * ratio)
        layers = []
        bn = config_channels.config.getboolean('batch_norm', 'enable')
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels *= 2
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels // 2, 'layers1.%d.conv.weight' % len(layers)), 1,
                bn=bn))
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels *= 2
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels // 2, 'layers1.%d.conv.weight' % len(layers)), 1,
                bn=bn))
        layers.append(Conv2d(config_channels.channels, config_channels(
            channels, 'layers1.%d.conv.weight' % len(layers)), 3, bn=bn,
            padding=True))
        self.layers1 = nn.Sequential(*layers)
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2))
        channels *= 2
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers2.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels // 2, 'layers2.%d.conv.weight' % len(layers)), 1,
                bn=bn))
        for _ in range(3):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers2.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
        self.layers2 = nn.Sequential(*layers)
        self.passthrough = Conv2d(self.layers1[-1].conv.weight.size(0),
            config_channels(int(64 * ratio), 'passthrough.conv.weight'), 1,
            bn=bn)
        layers = []
        layers.append(Conv2d(self.passthrough.conv.weight.size(0) * self.
            stride * self.stride + self.layers2[-1].conv.weight.size(0),
            config_channels(int(1024 * ratio), 'layers3.%d.conv.weight' %
            len(layers)), 3, bn=bn, padding=True))
        layers.append(Conv2d(config_channels.channels, model.
            output_channels(len(anchors), num_cls), 1, bn=False, act=False))
        self.layers3 = nn.Sequential(*layers)
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layers1(x)
        _x = reorg(self.passthrough(x), self.stride)
        x = self.layers2(x)
        x = torch.cat([_x, x], 1)
        return self.layers3(x)

    def scope(self, name):
        return '.'.join(name.split('.')[:-2])

    def get_mapper(self, index):
        if index == 94:
            return lambda indices, channels: torch.cat([(indices + i *
                channels) for i in range(self.stride * self.stride)])


class Tiny(nn.Module):

    def __init__(self, config_channels, anchors, num_cls, channels=16):
        nn.Module.__init__(self)
        layers = []
        bn = config_channels.config.getboolean('batch_norm', 'enable')
        for _ in range(5):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels *= 2
        layers.append(Conv2d(config_channels.channels, config_channels(
            channels, 'layers.%d.conv.weight' % len(layers)), 3, bn=bn,
            padding=True))
        layers.append(nn.ConstantPad2d((0, 1, 0, 1), float(np.finfo(np.
            float32).min)))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        channels *= 2
        for _ in range(2):
            layers.append(Conv2d(config_channels.channels, config_channels(
                channels, 'layers.%d.conv.weight' % len(layers)), 3, bn=bn,
                padding=True))
        layers.append(Conv2d(config_channels.channels, model.
            output_channels(len(anchors), num_cls), 1, bn=False, act=False))
        self.layers = nn.Sequential(*layers)
        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layers(x)

    def scope(self, name):
        return '.'.join(name.split('.')[:-2])


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ruiminshen_yolo2_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

