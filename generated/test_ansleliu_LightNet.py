import sys
_module = sys.modules[__name__]
del sys
augmentations = _module
calculate_class_weight = _module
cityscapes_loader = _module
cityscapesscripts = _module
annotation = _module
cityscapesLabelTool = _module
evaluation = _module
evalInstanceLevelSemanticLabeling = _module
evalPixelLevelSemanticLabeling = _module
instance = _module
instances2dict = _module
setup = _module
helpers = _module
csHelpers = _module
labels = _module
preparation = _module
createTrainIdInstanceImgs = _module
createTrainIdLabelImgs = _module
json2instanceImg = _module
json2labelImg = _module
viewer = _module
cityscapesViewer = _module
demo_cityscapes = _module
demo_mvd = _module
evaluation_cityscapes = _module
gan_augmention = _module
mapillary_vistas_loader = _module
models = _module
inceptionresnetv2 = _module
mixscaledensenet = _module
mobilenetv2aspp = _module
mobilenetv2exfuse = _module
mobilenetv2plus = _module
mobilenetv2share = _module
mobilenetv2vortex = _module
rfmobilenetv2context = _module
rfmobilenetv2plus = _module
rfshufflenetv2plus = _module
sedpshufflenet = _module
sewrnetv1 = _module
sewrnetv2 = _module
shufflenetv2plus = _module
modules = _module
_ext = _module
bn = _module
build = _module
context_encode = _module
dense = _module
dualpath = _module
exfuse = _module
functions = _module
group_norm = _module
misc = _module
residual = _module
rfblock = _module
net_viz = _module
guided_backprop = _module
layer_viz = _module
net_viz_pytorch = _module
visualize = _module
scripts = _module
cyclical_lr = _module
deploy_model = _module
loss = _module
metrics = _module
model_measure = _module
test_inplace = _module
train_auxiliary = _module
train_context = _module
train_inplace = _module
train_lovasz = _module
train_mixscale = _module
train_mobile = _module
train_mobile_mvd = _module
train_sedpnet = _module
train_share = _module
train_shuffle = _module
train_vortex = _module
utils = _module
yellowfin = _module

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


import numpy as np


import scipy.misc as misc


import torch


import torch.nn.functional as F


from torch.autograd import Variable


from functools import partial


import torch.nn as nn


from collections import OrderedDict


import math


from torch.nn import init


from collections import Iterable


from itertools import repeat


import torch.autograd as autograd


from torch.nn import ReLU


from functools import reduce


import random


from torch.utils import data


from itertools import filterfalse


import copy


import logging


class BasicConv2d(nn.Module):
    """
        Define the basic conv-bn-relu block
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=
        0, dilate=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilate,
            bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    """
        Define the 35x35 grid modules of Inception V4 network
        to replace later-half stem modules of InceptionResNet V2
    """

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1,
            stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1,
            stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding
            =1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1,
            count_include_pad=False), BasicConv2d(192, 64, kernel_size=1,
            stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    """
        The 35x35 grid modules of InceptionResNet V2
    """

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding
            =1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """
        The 35x35 to 17x17 reduction module
    """

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=1,
            padding=2, dilate=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=2, dilate=2), BasicConv2d(256, 384, kernel_size=3,
            stride=1, padding=2, dilate=2))
        self.branch2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1), nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    """
        The 17x17 grid modules of InceptionResNet V2
    """

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1,
            padding=(0, 6), dilate=(1, 2)), BasicConv2d(160, 192,
            kernel_size=(7, 1), stride=1, padding=(6, 0), dilate=(2, 1)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    """
        The 17x17 to 8x8 reduction module
    """

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=1,
            padding=4, dilate=4))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=4, dilate=4))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1,
            padding=4, dilate=4), BasicConv2d(288, 320, kernel_size=3,
            stride=1, padding=4, dilate=4))
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2,
            padding=1), nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    """
        The 8x8 grid modules of InceptionResNet V2
    """

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1,
            padding=(0, 4), dilate=(1, 4)), BasicConv2d(224, 256,
            kernel_size=(3, 1), stride=1, padding=(4, 0), dilate=(4, 1)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


def _check_contiguous(*args):
    if not all([(mod is None or mod.is_contiguous()) for mod in args]):
        raise ValueError('Non-contiguous input')


ACT_LEAKY_RELU = 'leaky_relu'


ACT_NONE = 'none'


ACT_ELU = 'elu'


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _ext.leaky_relu_backward_cuda(x, dx, ctx.slope)
        _ext.leaky_relu_cuda(x, 1.0 / ctx.slope)
    elif ctx.activation == ACT_ELU:
        _ext.elu_backward_cuda(x, dx)
        _ext.elu_inv_cuda(x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _ext.leaky_relu_cuda(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _ext.elu_cuda(x)
    elif ctx.activation == ACT_NONE:
        pass


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=oup,
        kernel_size=3, stride=stride, padding=1, bias=False), nn.
        BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True))


def conv3x3(in_channels, out_channels, stride=1, bias=True, groups=1, dilate=1
    ):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=dilate, bias=bias, groups=groups, dilation=dilate)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=
        groups, stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, dilate=1,
        grouped_conv=True, combine='add', up=False):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        if self.combine == 'add':
            self.depthwise_stride = 1
            self.dilate = dilate
            self.up = False
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 1 if up is True else 2
            self.dilate = dilate if up is True else 1
            self.up = up
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
        else:
            raise ValueError(
                'Cannot combine tensors with "{}"Only "add" and "concat" aresupported'
                .format(self.combine))
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.
            in_channels, self.bottleneck_channels, self.first_1x1_groups,
            batch_norm=True, relu=True)
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.
            bottleneck_channels, stride=self.depthwise_stride, groups=self.
            bottleneck_channels, dilate=self.dilate)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.
            bottleneck_channels, self.out_channels, self.groups, batch_norm
            =True, relu=False)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), dim=1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2,
                padding=1)
            if self.up is True:
                residual = F.upsample(residual, scale_factor=2, mode='bilinear'
                    )
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3, dilate=1,
        grouped_conv=True, combine='add', up=False):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        if self.combine == 'add':
            self.depthwise_stride = 1
            self.dilate = dilate
            self.up = False
            self._combine_func = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 1 if up is True else 2
            self.dilate = dilate if up is True else 1
            self.up = up
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
        else:
            raise ValueError(
                'Cannot combine tensors with "{}"Only "add" and "concat" aresupported'
                .format(self.combine))
        self.first_1x1_groups = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.
            in_channels, self.bottleneck_channels, self.first_1x1_groups,
            batch_norm=True, relu=True)
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.
            bottleneck_channels, stride=self.depthwise_stride, groups=self.
            bottleneck_channels, dilate=self.dilate)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.
            bottleneck_channels, self.out_channels, self.groups, batch_norm
            =True, relu=False)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), dim=1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2,
                padding=1)
            if self.up is True:
                residual = F.upsample(residual, scale_factor=2, mode='bilinear'
                    )
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)


class ABN(nn.Sequential):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, activation=nn.ReLU(inplace=True), **kwargs
        ):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        activation : nn.Module
            Module used as an activation function.
        kwargs
            All other arguments are forwarded to the `BatchNorm2d` constructor.
        """
        super(ABN, self).__init__(OrderedDict([('bn', nn.BatchNorm2d(
            num_features, **kwargs)), ('act', activation)]))


class InPlaceABNWrapper(nn.Module):
    """Wrapper module to make `InPlaceABN` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNWrapper, self).__init__()
        self.bn = InPlaceABN(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)


class InPlaceABNSyncWrapper(nn.Module):
    """Wrapper module to make `InPlaceABNSync` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNSyncWrapper, self).__init__()
        self.bn = InPlaceABNSync(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)


class DenseModule(nn.Module):

    def __init__(self, in_chns, squeeze_ratio, out_chns, n_layers,
        dilate_sec=(1, 2, 4, 8, 16), norm_act=ABN):
        super(DenseModule, self).__init__()
        self.n_layers = n_layers
        self.mid_out = int(in_chns * squeeze_ratio)
        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for idx in range(self.n_layers):
            dilate = dilate_sec[idx % len(dilate_sec)]
            self.last_channel = in_chns + idx * out_chns
            """
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.last_channel)),
                ("conv", nn.Conv2d(self.last_channel, self.mid_out, 1, bias=False))
            ])))
            """
            self.convs3.append(nn.Sequential(OrderedDict([('bn', norm_act(
                self.last_channel)), ('conv', nn.Conv2d(self.last_channel,
                out_chns, kernel_size=3, stride=1, padding=dilate, dilation
                =dilate, bias=False))])))

    @property
    def out_channels(self):
        return self.last_channel + 1

    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


class DPDenseModule(nn.Module):

    def __init__(self, in_chns, squeeze_ratio, out_chns, n_layers,
        dilate_sec=(1, 2, 4, 8, 16), norm_act=ABN):
        super(DPDenseModule, self).__init__()
        self.n_layers = n_layers
        self.convs3 = nn.ModuleList()
        for idx in range(self.n_layers):
            dilate = dilate_sec[idx % len(dilate_sec)]
            self.last_channel = in_chns + idx * out_chns
            mid_out = int(self.last_channel * squeeze_ratio)
            self.convs3.append(nn.Sequential(OrderedDict([('bn.1', norm_act
                (self.last_channel)), ('conv_up', nn.Conv2d(self.
                last_channel, mid_out, kernel_size=1, stride=1, padding=0,
                bias=False)), ('bn.2', norm_act(mid_out)), ('dconv', nn.
                Conv2d(mid_out, mid_out, kernel_size=3, stride=1, padding=
                dilate, groups=mid_out, dilation=dilate, bias=False)), (
                'pconv', nn.Conv2d(mid_out, out_chns, kernel_size=1, stride
                =1, padding=0, bias=False)), ('dropout', nn.Dropout2d(p=0.2,
                inplace=True))])))
            """
            self.convs3.append(nn.Sequential(OrderedDict([("bn.1", norm_act(self.last_channel)),
                                                          ("dconv", nn.Conv2d(self.last_channel, self.last_channel,
                                                                              kernel_size=3, stride=1, padding=dilate,
                                                                              groups=self.last_channel, dilation=dilate,
                                                                              bias=False)),
                                                          ("pconv", nn.Conv2d(self.last_channel, out_chns,
                                                                              kernel_size=1, stride=1, padding=0,
                                                                              bias=False)),
                                                          ("dropout", nn.Dropout2d(p=0.2, inplace=True))])))
            """

    @property
    def out_channels(self):
        return self.last_channel + 1

    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


class DualPathInPlaceABNBlock(nn.Module):

    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups
        =1, dilation=1, block_type='normal', norm_act=ABN):
        super(DualPathInPlaceABNBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.dilation = dilation
        self.groups = groups
        self.inc = inc
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False
        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = nn.Sequential(OrderedDict([(
                    'conv1x1_w_s2_bn', norm_act(in_chs)), ('conv1x1_w_s2',
                    nn.Conv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1,
                    stride=2, padding=0, groups=self.groups, dilation=1,
                    bias=False))]))
            else:
                self.c1x1_w_s1 = nn.Sequential(OrderedDict([(
                    'conv1x1_w_s1_bn', norm_act(in_chs)), ('conv1x1_w_s1',
                    nn.Conv2d(in_chs, num_1x1_c + 2 * inc, kernel_size=1,
                    stride=1, padding=0, groups=self.groups, dilation=1,
                    bias=False))]))
        self.c1x1_a = nn.Sequential(OrderedDict([('conv1x1_a_bn', norm_act(
            in_chs)), ('conv1x1_a', nn.Conv2d(in_chs, num_1x1_a,
            kernel_size=1, stride=1, padding=0, groups=self.groups,
            dilation=1, bias=False))]))
        if self.dilation > 1 and self.key_stride == 1:
            self.c3x3_b = nn.Sequential(OrderedDict([('conv3x3_b_bn',
                norm_act(num_1x1_a)), ('conv3x3_b', nn.Conv2d(num_1x1_a,
                num_3x3_b, kernel_size=3, stride=1, padding=dilation,
                groups=num_1x1_a, dilation=self.dilation, bias=False))]))
        else:
            self.c3x3_b = nn.Sequential(OrderedDict([('conv3x3_b_bn',
                norm_act(num_1x1_a)), ('conv3x3_b', nn.Conv2d(num_1x1_a,
                num_3x3_b, kernel_size=3, stride=self.key_stride, padding=1,
                groups=num_1x1_a, dilation=1, bias=False))]))
        self.c1x1_c = nn.Sequential(OrderedDict([('conv1x1_c_bn', norm_act(
            num_3x3_b)), ('conv1x1_c', nn.Conv2d(num_3x3_b, num_1x1_c + inc,
            kernel_size=1, stride=1, padding=0, groups=self.groups,
            dilation=1, bias=False)), ('se_block', SEBlock(num_1x1_c + inc,
            16)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))

    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1,
            height, width)
        return x

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in.clone())
            else:
                x_s = self.c1x1_w_s1(x_in.clone())
            x_s = self._channel_shuffle(x_s, self.groups)
            x_s = torch.split(x_s, self.num_1x1_c, dim=1)
        else:
            x_s = x
        x_r = self.c1x1_a(x_in)
        x_r = self._channel_shuffle(x_r, self.groups)
        x_r = self.c3x3_b(x_r)
        x_r = self.c1x1_c(x_r)
        x_r = self._channel_shuffle(x_r, self.groups)
        x_r = torch.split(x_r, self.num_1x1_c, dim=1)
        resid = torch.add(x_s[0], 1, x_r[0])
        dense = torch.cat([x_s[1], x_r[1]], dim=1)
        return resid, dense


class SemanticSupervision(nn.Module):

    def __init__(self, in_chns, out_chns):
        super(SemanticSupervision, self).__init__()
        self.out_chns = out_chns
        self.semantic = nn.Sequential(OrderedDict([('conv1x7', nn.Conv2d(
            in_chns, in_chns // 2 * 3, kernel_size=(1, 7), stride=1,
            padding=(0, 3), bias=False)), ('norm1', nn.BatchNorm2d(in_chns //
            2 * 3)), ('act1', nn.LeakyReLU(negative_slope=0.1, inplace=True
            )), ('conv7x1', nn.Conv2d(in_chns // 2 * 3, out_chns // 2 * 3,
            kernel_size=(7, 1), stride=1, padding=(3, 0), bias=False)), (
            'norm2', nn.BatchNorm2d(out_chns // 2 * 3)), ('act2', nn.
            LeakyReLU(negative_slope=0.1, inplace=True))]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_chns // 2 * 3, out_chns)

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        se = self.semantic(x)
        se = self.avg_pool(se).view(bahs, self.out_chns // 2 * 3)
        se = self.classifier(se)
        return se


class GroupNorm2D(nn.Module):
    """
        Group Normalization
        Reference: https://128.84.21.199/abs/1803.08494v1
    """

    def __init__(self, num_features, num_groups=32, eps=1e-05):
        """

        :param num_features:
        :param num_groups:
        :param eps:
        """
        super(GroupNorm2D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.num_groups == 0
        x = x.view(batch_size, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(batch_size, num_channels, height, width)
        return x * self.weight + self.bias


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        return inputs.view((in_size[0], in_size[1], 1, 1))


class CatInPlaceABN(nn.Module):
    """
    Block for concat the two output tensor of feature net
    """

    def __init__(self, in_chs, norm_act=ABN):
        super(CatInPlaceABN, self).__init__()
        self.norm_act = norm_act(in_chs)

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        x = self.norm_act(x)
        return x


class LightHeadBlock(nn.Module):

    def __init__(self, in_chs, mid_chs=256, out_chs=256, kernel_size=15,
        norm_act=ABN):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.abn = norm_act(in_chs)
        self.conv_l = nn.Sequential(OrderedDict([('conv_lu', nn.Conv2d(
            in_chs, mid_chs, kernel_size=(kernel_size, 1), padding=(pad, 0)
            )), ('conv_ld', nn.Conv2d(mid_chs, out_chs, kernel_size=(1,
            kernel_size), padding=(0, pad)))]))
        self.conv_r = nn.Sequential(OrderedDict([('conv_ru', nn.Conv2d(
            in_chs, mid_chs, kernel_size=(1, kernel_size), padding=(0, pad)
            )), ('conv_rd', nn.Conv2d(mid_chs, out_chs, kernel_size=(
            kernel_size, 1), padding=(pad, 0)))]))

    def forward(self, x):
        x = self.abn(x)
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.Sequential(nn.Linear(channel, int(channel / reduction
            )), nn.LeakyReLU(negative_slope=0.1, inplace=True), nn.Linear(
            int(channel / reduction), channel), nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        y = self.avg_pool(x).view(bahs, chs)
        y = self.fcs(y).view(bahs, chs, 1, 1)
        return torch.mul(x, y)


class SCSEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(
            channel // reduction)), nn.ReLU(inplace=True), nn.Linear(int(
            channel // reduction), channel), nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
            stride=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)
        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class ModifiedSCSEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(ModifiedSCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(
            channel // reduction)), nn.ReLU(inplace=True), nn.Linear(int(
            channel // reduction), channel), nn.Sigmoid())
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
            stride=1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        spa_se = self.spatial_se(x)
        return torch.mul(torch.mul(x, chn_se), spa_se)


class VortexPooling(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2,
        rate=(3, 9, 27)):
        super(VortexPooling, self).__init__()
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.
            AdaptiveAvgPool2d((1, 1))), ('conv1x1', nn.Conv2d(in_chs,
            out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=
            False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode=
            'bilinear')), ('bn0', nn.BatchNorm2d(num_features=out_chs))]))
        self.conv3x3 = nn.Sequential(OrderedDict([('conv3x3', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=False,
            groups=1, dilation=1)), ('bn3x3', nn.BatchNorm2d(num_features=
            out_chs))]))
        self.vortex_bra1 = nn.Sequential(OrderedDict([('avg_pool', nn.
            AvgPool2d(kernel_size=rate[0], stride=1, padding=int((rate[0] -
            1) / 2), ceil_mode=False)), ('conv3x3', nn.Conv2d(in_chs,
            out_chs, kernel_size=3, stride=1, padding=rate[0], bias=False,
            groups=1, dilation=rate[0])), ('bn3x3', nn.BatchNorm2d(
            num_features=out_chs))]))
        self.vortex_bra2 = nn.Sequential(OrderedDict([('avg_pool', nn.
            AvgPool2d(kernel_size=rate[1], stride=1, padding=int((rate[1] -
            1) / 2), ceil_mode=False)), ('conv3x3', nn.Conv2d(in_chs,
            out_chs, kernel_size=3, stride=1, padding=rate[1], bias=False,
            groups=1, dilation=rate[1])), ('bn3x3', nn.BatchNorm2d(
            num_features=out_chs))]))
        self.vortex_bra3 = nn.Sequential(OrderedDict([('avg_pool', nn.
            AvgPool2d(kernel_size=rate[2], stride=1, padding=int((rate[2] -
            1) / 2), ceil_mode=False)), ('conv3x3', nn.Conv2d(in_chs,
            out_chs, kernel_size=3, stride=1, padding=rate[2], bias=False,
            groups=1, dilation=rate[2])), ('bn3x3', nn.BatchNorm2d(
            num_features=out_chs))]))
        self.vortex_catdown = nn.Sequential(OrderedDict([('conv_down', nn.
            Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1,
            bias=False, groups=1, dilation=1)), ('bn_down', nn.BatchNorm2d(
            num_features=out_chs)), ('dropout', nn.Dropout2d(p=0.2, inplace
            =True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio),
            int(feat_res[1] * up_ratio)), mode='bilinear')

    def forward(self, x):
        out = torch.cat([self.gave_pool(x), self.conv3x3(x), self.
            vortex_bra1(x), self.vortex_bra2(x), self.vortex_bra3(x)], dim=1)
        out = self.vortex_catdown(out)
        return self.upsampling(out)


class ASPPBlock(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2,
        aspp_sec=(12, 24, 36)):
        super(ASPPBlock, self).__init__()
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.
            AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(in_chs,
            out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=
            False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode=
            'bilinear')), ('bn0', nn.BatchNorm2d(num_features=out_chs))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False,
            groups=1, dilation=1)), ('bn1_1', nn.BatchNorm2d(num_features=
            out_chs))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('conv2_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[0],
            bias=False, groups=1, dilation=aspp_sec[0])), ('bn2_1', nn.
            BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('conv2_2', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[1],
            bias=False, groups=1, dilation=aspp_sec[1])), ('bn2_2', nn.
            BatchNorm2d(num_features=out_chs))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('conv2_3', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[2],
            bias=False, groups=1, dilation=aspp_sec[2])), ('bn2_3', nn.
            BatchNorm2d(num_features=out_chs))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('conv_down', nn.
            Conv2d(5 * out_chs, out_chs, kernel_size=1, stride=1, padding=1,
            bias=False, groups=1, dilation=1)), ('bn_down', nn.BatchNorm2d(
            num_features=out_chs)), ('dropout', nn.Dropout2d(p=0.2, inplace
            =True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio),
            int(feat_res[1] * up_ratio)), mode='bilinear')

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1,
            height, width)
        return x

    def forward(self, x):
        out = torch.cat([self.gave_pool(x), self.conv1x1(x), self.aspp_bra1
            (x), self.aspp_bra2(x), self.aspp_bra3(x)], dim=1)
        out = self.aspp_catdown(out)
        return self.upsampling(out)


class ASPPInPlaceABNBlock(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2,
        aspp_sec=(12, 24, 36), norm_act=ABN):
        super(ASPPInPlaceABNBlock, self).__init__()
        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.
            AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(in_chs,
            out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=
            False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode=
            'bilinear'))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False,
            groups=1, dilation=1))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('conv2_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[0],
            bias=False, groups=1, dilation=aspp_sec[0]))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('conv2_2', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[1],
            bias=False, groups=1, dilation=aspp_sec[1]))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('conv2_3', nn.Conv2d(
            in_chs, out_chs, kernel_size=3, stride=1, padding=aspp_sec[2],
            bias=False, groups=1, dilation=aspp_sec[2]))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('norm_act',
            norm_act(5 * out_chs)), ('conv_down', nn.Conv2d(5 * out_chs,
            out_chs, kernel_size=1, stride=1, padding=1, bias=False, groups
            =1, dilation=1)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio),
            int(feat_res[1] * up_ratio)), mode='bilinear')

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1,
            height, width)
        return x

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([self.gave_pool(x), self.conv1x1(x), self.aspp_bra1(x
            ), self.aspp_bra2(x), self.aspp_bra3(x)], dim=1)
        out = self.aspp_catdown(x)
        return out, self.upsampling(out)


class SDASPPInPlaceABNBlock(nn.Module):

    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2,
        aspp_sec=(12, 24, 36), norm_act=ABN):
        super(SDASPPInPlaceABNBlock, self).__init__()
        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([('gavg', nn.
            AdaptiveAvgPool2d((1, 1))), ('conv1_0', nn.Conv2d(in_chs,
            out_chs, kernel_size=1, stride=1, padding=0, groups=1, bias=
            False, dilation=1)), ('up0', nn.Upsample(size=feat_res, mode=
            'bilinear'))]))
        self.conv1x1 = nn.Sequential(OrderedDict([('conv1_1', nn.Conv2d(
            in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=False,
            groups=1, dilation=1))]))
        self.aspp_bra1 = nn.Sequential(OrderedDict([('dconv2_1', nn.Conv2d(
            in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_sec[0],
            bias=False, groups=in_chs, dilation=aspp_sec[0])), ('pconv2_1',
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0,
            bias=False, groups=1, dilation=1))]))
        self.aspp_bra2 = nn.Sequential(OrderedDict([('dconv2_2', nn.Conv2d(
            in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_sec[1],
            bias=False, groups=in_chs, dilation=aspp_sec[1])), ('pconv2_2',
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0,
            bias=False, groups=1, dilation=1))]))
        self.aspp_bra3 = nn.Sequential(OrderedDict([('dconv2_3', nn.Conv2d(
            in_chs, in_chs, kernel_size=3, stride=1, padding=aspp_sec[2],
            bias=False, groups=in_chs, dilation=aspp_sec[2])), ('pconv2_3',
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0,
            bias=False, groups=1, dilation=1))]))
        self.aspp_catdown = nn.Sequential(OrderedDict([('norm_act',
            norm_act(5 * out_chs)), ('conv_down', nn.Conv2d(5 * out_chs,
            out_chs, kernel_size=1, stride=1, padding=1, bias=False, groups
            =1, dilation=1)), ('dropout', nn.Dropout2d(p=0.2, inplace=True))]))
        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio),
            int(feat_res[1] * up_ratio)), mode='bilinear')

    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1,
            height, width)
        return x

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([self.gave_pool(x), self.conv1x1(x), self.aspp_bra1(x
            ), self.aspp_bra2(x), self.aspp_bra3(x)], dim=1)
        return self.upsampling(self.aspp_catdown(x))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=
            inp * expand_ratio, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=
            inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True), nn.
            ReLU6(inplace=True), nn.Conv2d(in_channels=inp * expand_ratio,
            out_channels=inp * expand_ratio, kernel_size=3, stride=stride,
            padding=dilate, dilation=dilate, groups=inp * expand_ratio,
            bias=False), nn.BatchNorm2d(num_features=inp * expand_ratio,
            eps=1e-05, momentum=0.1, affine=True), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
            kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=
            False), nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=
            0.1, affine=True))

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)


class SCSEInvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(SCSEInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(in_channels=inp, out_channels=
            inp * expand_ratio, kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False), nn.BatchNorm2d(num_features=
            inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True), nn.
            ReLU6(inplace=True), nn.Conv2d(in_channels=inp * expand_ratio,
            out_channels=inp * expand_ratio, kernel_size=3, stride=stride,
            padding=dilate, dilation=dilate, groups=inp * expand_ratio,
            bias=False), nn.BatchNorm2d(num_features=inp * expand_ratio,
            eps=1e-05, momentum=0.1, affine=True), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
            kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=
            False), nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=
            0.1, affine=True), SCSEBlock(channel=oup, reduction=2))

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=
        1, norm_act=ABN, is_se=False, dropout=None):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values'
                )
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3,
                stride=stride, padding=dilation, bias=False, dilation=
                dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.
                Conv2d(channels[0], channels[1], 3, stride=1, padding=
                dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            if not is_se:
                layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1,
                    stride=stride, padding=0, bias=False)), ('bn2',
                    norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0],
                    channels[1], 3, stride=1, padding=dilation, bias=False,
                    groups=groups, dilation=dilation)), ('bn3', norm_act(
                    channels[1])), ('conv3', nn.Conv2d(channels[1],
                    channels[2], 1, stride=1, padding=0, bias=False))]
            else:
                layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1,
                    stride=stride, padding=0, bias=False)), ('bn2',
                    norm_act(channels[0])), ('conv2', nn.Conv2d(channels[0],
                    channels[1], 3, stride=1, padding=dilation, bias=False,
                    groups=groups, dilation=dilation)), ('bn3', norm_act(
                    channels[1])), ('conv3', nn.Conv2d(channels[1],
                    channels[2], 1, stride=1, padding=0, bias=False)), (
                    'se_block', SEBlock(channels[2], 16))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride
                =stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)
            ].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average
            =self.size_average)
        return loss


class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """

    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25, gamma=
        2, size_average=True):
        """
        Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

        :param num_classes:   (int) num of the classes
        :param ignore_label:  (int) ignore label
        :param alpha:         (1D Tensor or Variable) the scalar factor
        :param gamma:         (float) gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        :param size_average:  (bool): By default, the losses are averaged over observations for each mini-batch.
                                      If the size_average is set to False, the losses are
                                      instead summed for each mini-batch.
        """
        super(FocalLoss2D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.size_average = size_average
        self.one_hot = Variable(torch.eye(self.num_classes))

    def forward(self, cls_preds, cls_targets):
        """

        :param cls_preds:    (n, c, h, w)
        :param cls_targets:  (n, h, w)
        :return:
        """
        assert not cls_targets.requires_grad
        assert cls_targets.dim() == 3
        assert cls_preds.size(0) == cls_targets.size(0), '{0} vs {1} '.format(
            cls_preds.size(0), cls_targets.size(0))
        assert cls_preds.size(2) == cls_targets.size(1), '{0} vs {1} '.format(
            cls_preds.size(2), cls_targets.size(1))
        assert cls_preds.size(3) == cls_targets.size(2), '{0} vs {1} '.format(
            cls_preds.size(3), cls_targets.size(3))
        if cls_preds.is_cuda:
            self.one_hot = self.one_hot
        n, c, h, w = cls_preds.size()
        cls_targets = cls_targets.view(n * h * w, 1)
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)
        cls_targets = cls_targets[target_mask]
        cls_targets = self.one_hot.index_select(dim=0, index=cls_targets)
        prob = F.softmax(cls_preds, dim=1)
        prob = prob.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        prob = prob[target_mask.repeat(1, c)]
        prob = prob.view(-1, c)
        probs = torch.clamp((prob * cls_targets).sum(1).view(-1, 1), min=
            1e-08, max=1.0)
        batch_loss = -self.alpha * torch.pow(1 - probs, self.gamma
            ) * probs.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class SemanticEncodingLoss(nn.Module):

    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25):
        super(SemanticEncodingLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def unique_encode(self, cls_targets):
        batch_size, _, _ = cls_targets.size()
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)
        cls_targets = [cls_targets[idx].masked_select(target_mask[idx]) for
            idx in np.arange(batch_size)]
        unique_cls = [np.unique(label.numpy()) for label in cls_targets]
        encode = np.zeros((batch_size, self.num_classes), dtype=np.uint8)
        for idx in np.arange(batch_size):
            np.put(encode[idx], unique_cls[idx], 1)
        return torch.from_numpy(encode).float()

    def forward(self, predicts, enc_cls_target, size_average=True):
        se_loss = F.binary_cross_entropy_with_logits(predicts,
            enc_cls_target, weight=None, size_average=size_average)
        return self.alpha * se_loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ansleliu_LightNet(_paritybench_base):
    pass
    def test_000(self):
        self._check(ABN(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ASPPBlock(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 56, 112])], {})

    def test_002(self):
        self._check(ASPPInPlaceABNBlock(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 56, 112])], {})

    def test_003(self):
        self._check(BasicConv2d(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Block17(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_005(self):
        self._check(Block35(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_006(self):
        self._check(Block8(*[], **{}), [torch.rand([4, 2080, 64, 64])], {})

    @_fails_compile()
    def test_007(self):
        self._check(CatInPlaceABN(*[], **{'in_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(DenseModule(*[], **{'in_chns': 4, 'squeeze_ratio': 4, 'out_chns': 4, 'n_layers': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(IdentityResidualBlock(*[], **{'in_channels': 4, 'channels': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'dilate': 4, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(LightHeadBlock(*[], **{'in_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Mixed_5b(*[], **{}), [torch.rand([4, 192, 64, 64])], {})

    def test_014(self):
        self._check(Mixed_6a(*[], **{}), [torch.rand([4, 320, 64, 64])], {})

    def test_015(self):
        self._check(Mixed_7a(*[], **{}), [torch.rand([4, 1088, 64, 64])], {})

    def test_016(self):
        self._check(SCSEInvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'dilate': 4, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(SDASPPInPlaceABNBlock(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 56, 112])], {})

    @_fails_compile()
    def test_018(self):
        self._check(SemanticEncodingLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(SemanticSupervision(*[], **{'in_chns': 4, 'out_chns': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(VortexPooling(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 56, 112])], {})

