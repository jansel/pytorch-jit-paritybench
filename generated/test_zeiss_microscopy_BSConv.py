import sys
_module = sys.modules[__name__]
del sys
bsconv_pytorch_train = _module
bsconv = _module
datasets = _module
pytorch = _module
common = _module
mobilenet = _module
modules = _module
profile = _module
provider = _module
replacers = _module
resnet = _module
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


import torch


import torch.nn


import torch.optim


import torch.optim.lr_scheduler


import torch.utils.data


import torchvision.transforms


import torchvision.datasets


import re


import types


import torch.nn.init


import math


import collections


import warnings


import abc


class Swish(torch.nn.Module):

    def forward(self, x):
        return x * torch.nn.functional.sigmoid(x, inplace=True)


class HSwish(torch.nn.Module):

    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0


class HSigmoid(torch.nn.Module):

    def forward(self, x):
        return torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)


def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU(inplace=True)
    elif activation == 'relu6':
        return torch.nn.ReLU6(inplace=True)
    elif activation == 'swish':
        return Swish()
    elif activation == 'hswish':
        return HSwish()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid(inplace=True)
    elif activation == 'hsigmoid':
        return HSigmoid()
    else:
        raise NotImplementedError('Activation {} not implemented'.format(activation))


class SEUnit(torch.nn.Module):

    def __init__(self, channels, squeeze_factor=16, squeeze_activation='relu', excite_activation='sigmoid'):
        super().__init__()
        squeeze_channels = channels // squeeze_factor
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels, out_channels=squeeze_channels, bias=True)
        self.activation1 = get_activation(squeeze_activation)
        self.conv2 = conv1x1(in_channels=squeeze_channels, out_channels=channels, bias=True)
        self.activation2 = get_activation(excite_activation)

    def forward(self, x):
        s = self.pool(x)
        s = self.conv1(s)
        s = self.activation1(s)
        s = self.conv2(s)
        s = self.activation2(s)
        return x * s


class Classifier(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

    def init_params(self):
        torch.nn.init.xavier_normal_(self.conv.weight, gain=1.0)


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False, use_bn=True, activation='relu'):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = activation is not None
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


def conv1x1_block(in_channels, out_channels, stride=1, bias=False, use_bn=True, activation='relu'):
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias, use_bn=use_bn, activation=activation)


def conv3x3_dw_block(channels, stride=1, use_bn=True, activation='relu'):
    return ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, groups=channels, use_bn=use_bn, activation=activation)


class DepthwiseSeparableConvBlock(torch.nn.Module):
    """
    Depthwise-separable convolution (DSC) block internally used in MobileNets.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv_dw = conv3x3_dw_block(channels=in_channels, stride=stride)
        self.conv_pw = conv1x1_block(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


def conv5x5_dw_block(channels, stride=1, use_bn=True, activation='relu'):
    return ConvBlock(in_channels=channels, out_channels=channels, kernel_size=5, stride=stride, padding=2, groups=channels, use_bn=use_bn, activation=activation)


class LinearBottleneck(torch.nn.Module):
    """
    Linear bottleneck block internally used in MobileNets.
    """

    def __init__(self, in_channels, mid_channels, out_channels, stride, activation='relu6', kernel_size=3, use_se=False):
        super().__init__()
        self.use_res_skip = in_channels == out_channels and stride == 1
        self.use_se = use_se
        self.conv1 = conv1x1_block(in_channels=in_channels, out_channels=mid_channels, activation=activation)
        if kernel_size == 3:
            self.conv2 = conv3x3_dw_block(channels=mid_channels, stride=stride, activation=activation)
        elif kernel_size == 5:
            self.conv2 = conv5x5_dw_block(channels=mid_channels, stride=stride, activation=activation)
        else:
            raise ValueError
        if self.use_se:
            self.se_unit = SEUnit(channels=mid_channels, squeeze_factor=4, squeeze_activation='relu', excite_activation='hsigmoid')
        self.conv3 = conv1x1_block(in_channels=mid_channels, out_channels=out_channels, activation=None)

    def forward(self, x):
        if self.use_res_skip:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se_unit(x)
        x = self.conv3(x)
        if self.use_res_skip:
            x = x + residual
        return x


def conv3x3_block(in_channels, out_channels, stride=1, bias=False, use_bn=True, activation='relu'):
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=bias, use_bn=use_bn, activation=activation)


class MobileNetV1(torch.nn.Module):
    """
    Class for constructing MobileNetsV1.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """

    def __init__(self, num_classes, init_conv_channels, init_conv_stride, channels, strides, in_channels=3, in_size=(224, 224), use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size
        self.backbone = torch.nn.Sequential()
        if self.use_data_batchnorm:
            self.backbone.add_module('data_bn', torch.nn.BatchNorm2d(num_features=in_channels))
        self.backbone.add_module('init_conv', conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module('unit{}'.format(unit_id + 1), DepthwiseSeparableConvBlock(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module('stage{}'.format(stage_id + 1), stage)
        self.backbone.add_module('global_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class MobileNetV2(torch.nn.Module):
    """
    Class for constructing MobileNetsV2.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """

    def __init__(self, num_classes, init_conv_channels, init_conv_stride, channels, mid_channels, final_conv_channels, strides, in_channels=3, in_size=(224, 224), use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size
        self.backbone = torch.nn.Sequential()
        if self.use_data_batchnorm:
            self.backbone.add_module('data_bn', torch.nn.BatchNorm2d(num_features=in_channels))
        self.backbone.add_module('init_conv', conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride, activation='relu6'))
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                mid_channel = mid_channels[stage_id][unit_id]
                stage.add_module('unit{}'.format(unit_id + 1), LinearBottleneck(in_channels=in_channels, mid_channels=mid_channel, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module('stage{}'.format(stage_id + 1), stage)
        self.backbone.add_module('final_conv', conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels, activation='relu6'))
        self.backbone.add_module('global_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.classifier = Classifier(in_channels=final_conv_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class MobileNetV3(torch.nn.Module):
    """
    Class for constructing MobileNetsV3.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """

    def __init__(self, num_classes, init_conv_channels, init_conv_stride, final_conv_channels, final_conv_se, channels, mid_channels, strides, se_units, kernel_sizes, activations, dropout_rate=0.0, in_channels=3, in_size=(224, 224), use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size
        self.dropout_rate = dropout_rate
        self.backbone = torch.nn.Sequential()
        if self.use_data_batchnorm:
            self.backbone.add_module('data_bn', torch.nn.BatchNorm2d(num_features=in_channels))
        self.backbone.add_module('init_conv', conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride, activation='hswish'))
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                mid_channel = mid_channels[stage_id][unit_id]
                use_se = se_units[stage_id][unit_id] == 1
                kernel_size = kernel_sizes[stage_id]
                activation = activations[stage_id]
                stage.add_module('unit{}'.format(unit_id + 1), LinearBottleneck(in_channels=in_channels, mid_channels=mid_channel, out_channels=unit_channels, stride=stride, activation=activation, use_se=use_se, kernel_size=kernel_size))
                in_channels = unit_channels
            self.backbone.add_module('stage{}'.format(stage_id + 1), stage)
        self.backbone.add_module('final_conv1', conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels[0], activation='hswish'))
        in_channels = final_conv_channels[0]
        if final_conv_se:
            self.backbone.add_module('final_se', SEUnit(channels=in_channels, squeeze_factor=4, squeeze_activation='relu', excite_activation='hsigmoid'))
        self.backbone.add_module('final_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))
        if len(final_conv_channels) > 1:
            self.backbone.add_module('final_conv2', conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels[1], activation='hswish', use_bn=False))
            in_channels = final_conv_channels[1]
        if self.dropout_rate != 0.0:
            self.backbone.add_module('final_dropout', torch.nn.Dropout(dropout_rate))
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class BSConvU(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', with_bn=False, bn_kwargs=None):
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        self.add_module('pw', torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))
        self.add_module('dw', torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels, bias=bias, padding_mode=padding_mode))


class BSConvS(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', p=0.25, min_mid_channels=4, with_bn=False, bn_kwargs=None):
        super().__init__()
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}
        self.add_module('pw1', torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False))
        if with_bn:
            self.add_module('bn1', torch.nn.BatchNorm2d(num_features=mid_channels, **bn_kwargs))
        self.add_module('pw2', torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False))
        if with_bn:
            self.add_module('bn2', torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))
        self.add_module('dw', torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels, bias=bias, padding_mode=padding_mode))

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p='fro')


def conv7x7_block(in_channels, out_channels, stride=1, bias=False, use_bn=True, activation='relu'):
    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3, bias=bias, use_bn=use_bn, activation=activation)


class InitUnitLarge(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv7x7_block(in_channels=in_channels, out_channels=out_channels, stride=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class InitUnitSmall(torch.nn.Module):

    def __init__(self, in_channels, out_channels, preact=False):
        super().__init__()
        self.conv = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=1, use_bn=not preact, activation=None if preact else 'relu')

    def forward(self, x):
        x = self.conv(x)
        return x


class PostActivation(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class StandardUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.use_projection = in_channels != out_channels or stride != 1
        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, stride=1, activation=None)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride, activation=None)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_projection:
            residual = self.projection(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x


class PreactUnit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.use_projection = in_channels != out_channels or stride != 1
        self.bn = torch.nn.BatchNorm2d(num_features=in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, use_bn=False, activation=None)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride, use_bn=False, activation=None)

    def forward(self, x):
        if self.use_projection:
            x = self.bn(x)
            x = self.relu(x)
            residual = self.projection(x)
        else:
            residual = x
            x = self.bn(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class ResNet(torch.nn.Module):

    def __init__(self, channels, num_classes, preact=False, init_unit_channels=64, use_init_unit_large=True, in_channels=3, in_size=(224, 224), use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.use_init_unit_large = use_init_unit_large
        self.in_size = in_size
        self.backbone = torch.nn.Sequential()
        if self.use_data_batchnorm:
            self.backbone.add_module('data_bn', torch.nn.BatchNorm2d(num_features=in_channels))
        if self.use_init_unit_large:
            self.backbone.add_module('init_unit', InitUnitLarge(in_channels=in_channels, out_channels=init_unit_channels))
        else:
            self.backbone.add_module('init_unit', InitUnitSmall(in_channels=in_channels, out_channels=init_unit_channels, preact=preact))
        in_channels = init_unit_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = 2 if unit_id == 0 and stage_id != 0 else 1
                if preact:
                    stage.add_module('unit{}'.format(unit_id + 1), PreactUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                else:
                    stage.add_module('unit{}'.format(unit_id + 1), StandardUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module('stage{}'.format(stage_id + 1), stage)
        if preact:
            self.backbone.add_module('final_activation', PostActivation(in_channels))
        self.backbone.add_module('global_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BSConvS,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BSConvU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classifier,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseSeparableConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InitUnitLarge,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InitUnitSmall,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBottleneck,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PostActivation,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreactUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (StandardUnit,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_zeiss_microscopy_BSConv(_paritybench_base):
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

