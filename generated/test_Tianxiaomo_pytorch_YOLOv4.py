import sys
_module = sys.modules[__name__]
del sys
cfg = _module
dataset = _module
demo = _module
demo_onnx = _module
demo_tensorflow = _module
evaluate_on_coco = _module
models = _module
camera = _module
coco_annotation = _module
config = _module
darknet2onnx = _module
darknet2pytorch = _module
onnx2tensorflow = _module
region_loss = _module
utils = _module
yolo_layer = _module
train = _module

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


import logging


import time


import numpy as np


import torch


from torch import nn


import torch.nn.functional as F


import torch.nn as nn


import math


from torch.autograd import Variable


import itertools


from torch.utils.data import DataLoader


from torch import optim


from torch.nn import functional as F


class Mish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.tanh(torch.nn.functional.softplus(x))
        return x


class Upsample(nn.Module):

    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert x.data.dim() == 4
        _, _, H, W = target_size
        return F.interpolate(x, size=(H, W), mode='nearest')


class Conv_Bn_Activation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(Mish())
        elif activation == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'linear':
            pass
        else:
            None

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of     two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4
        x7 = self.conv7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        self.resblock = ResBlock(ch=64, nblocks=2)
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        r = self.resblock(x3)
        x4 = self.conv4(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.upsample1 = Upsample()
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.upsample2 = Upsample()
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        up = self.upsample1(x7, downsample4.size())
        x8 = self.conv8(downsample4)
        x8 = torch.cat([x8, up], dim=1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        up = self.upsample2(x14, downsample3.size())
        x15 = self.conv15(downsample3)
        x15 = torch.cat([x15, up], dim=1)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):

    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=
            False, bias=True)
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn
            =False, bias=True)
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear',
            bn=False, bias=True)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        x3 = self.conv3(input1)
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x8)
        x11 = torch.cat([x11, input3], dim=1)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        return [x2, x10, x18]


class Yolov4(nn.Module):

    def __init__(self, yolov4conv137weight=None, n_classes=80):
        super().__init__()
        output_ch = (4 + 1 + n_classes) * 3
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        self.neck = Neck()
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self
                .down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)
            model_dict = _model.state_dict()
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.
                items(), model_dict)}
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        self.head = Yolov4Head(output_ch)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        x20, x13, x6 = self.neck(d5, d4, d3)
        output = self.head(x20, x13, x6)
        return output


class Mish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.tanh(torch.nn.functional.softplus(x))
        return x


class MaxPoolStride1(nn.Module):

    def __init__(self, size=2):
        super(MaxPoolStride1, self).__init__()
        self.size = size
        if (self.size - 1) % 2 == 0:
            self.padding1 = (self.size - 1) // 2
            self.padding2 = self.padding1
        else:
            self.padding1 = (self.size - 1) // 2
            self.padding2 = self.padding1 + 1

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (self.padding1, self.padding2, self.
            padding1, self.padding2), mode='replicate'), self.size, stride=1)
        return x


class Upsample(nn.Module):

    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride
            ).contiguous().view(B, C, H * stride, W * stride)
        return x


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % stride == 0
        assert W % stride == 0
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class EmptyModule(nn.Module):

    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w])
        .reshape(conv_model.weight.data.shape))
    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w])
        .reshape(conv_model.weight.data.shape))
    start = start + num_w
    return start


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()
    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    print('layer     filters    size              input                output')
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print(
                '%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d'
                 % (ind, 'conv', filters, kernel_size, kernel_size, stride,
                prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            print(
                '%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d'
                 % (ind, 'max', pool_size, pool_size, stride, prev_width,
                prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
                ind, 'avg', prev_width, prev_height, prev_filters,
                prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (
                ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' %
                (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            print(
                '%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d'
                 % (ind, 'reorg', stride, prev_width, prev_height,
                prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print(
                '%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d'
                 % (ind, 'upsample', stride, prev_width, prev_height,
                prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]]
                assert prev_height == out_heights[layers[1]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            elif len(layers) == 4:
                print('%5d %-6s %d %d %d %d' % (ind, 'route', layers[0],
                    layers[1], layers[2], layers[3]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]] == out_widths[
                    layers[2]] == out_widths[layers[3]]
                assert prev_height == out_heights[layers[1]] == out_heights[
                    layers[2]] == out_heights[layers[3]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]
                    ] + out_filters[layers[2]] + out_filters[layers[3]]
            else:
                print('route error !!! {} {} {}'.format(sys._getframe().
                    f_code.co_filename, sys._getframe().f_code.co_name, sys
                    ._getframe().f_lineno))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind,
                'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % block['type'])


class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)
        self.loss = self.models[len(self.models) - 1]
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        if self.blocks[len(self.blocks) - 1]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg',
                'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in
                    layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    None
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'yolo':
                if self.training:
                    pass
                else:
                    boxes = self.models[ind](x)
                    out_boxes.append(boxes)
            elif block['type'] == 'cost':
                continue
            else:
                None
        if self.training:
            return loss
        else:
            return out_boxes

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(
                        prev_filters, filters, kernel_size, stride, pad,
                        bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.
                        BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(
                        prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.
                        LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(
                        inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                else:
                    None
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                padding = 0
                if 'pad' in block.keys() and int(block['pad']) == 1:
                    padding = int((pool_size - 1) / 2)
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride, padding=padding)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in
                    layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert layers[0] == ind - 1
                    prev_filters = out_filters[layers[0]] + out_filters[
                        layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    assert layers[0] == ind - 1
                    prev_filters = out_filters[layers[0]] + out_filters[
                        layers[1]] + out_filters[layers[2]] + out_filters[
                        layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    None
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors) // loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors
                    ) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                None
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                None


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0
            )
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0
            )
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0
            )
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0
            )
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = (cw <= 0) + (ch <= 0) > 0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH,
    nW, noobject_scale, object_scale, sil_thresh, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors) / num_anchors
    conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)
    nAnchors = nA * nH * nW
    nPixels = nH * nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors,
                1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes,
                cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b][cur_ious > sil_thresh] = 0
    if seen < 12800:
        if anchor_step == 4:
            tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(
                1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1,
                nH, nW)
            ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step
                ).index_select(1, torch.LongTensor([2])).view(1, nA, 1, 1
                ).repeat(nB, 1, nH, nW)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = [0, 0, aw, ah]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step * n + 2]
                    ay = anchors[anchor_step * n + 3]
                    dist = pow(gi + ax - gx, 2) + pow(gj + ay - gy, 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist
            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW +
                gi]
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw / anchors[anchor_step * best_n]
                )
            th[b][best_n][gj][gi] = math.log(gh / anchors[anchor_step *
                best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t * 5]
            if iou > 0.5:
                nCorrect = nCorrect + 1
    return (nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th,
        tconf, tcls)


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


class RegionLoss(nn.Module):

    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) / num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        output = output.view(nB, nA, 5 + nC, nH, nW)
        x = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([0])
            )).view(nB, nA, nH, nW))
        y = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([1])
            )).view(nB, nA, nH, nW))
        w = output.index_select(2, Variable(torch.LongTensor([2]))).view(nB,
            nA, nH, nW)
        h = output.index_select(2, Variable(torch.LongTensor([3]))).view(nB,
            nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.LongTensor([
            4]))).view(nB, nA, nH, nW))
        cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC - 1,
            nC).long()))
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(
            nB * nA * nH * nW, nC)
        t1 = time.time()
        pred_boxes = torch.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA,
            1, 1).view(nB * nA * nH * nW)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB *
            nA, 1, 1).view(nB * nA * nH * nW)
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step
            ).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step
            ).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB *
            nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB *
            nA * nH * nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().
            view(-1, 4))
        t2 = time.time()
        (nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th,
            tconf, tcls) = (build_targets(pred_boxes, target.data, self.
            anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale,
            self.thresh, self.seen))
        cls_mask = cls_mask == 1
        nProposals = int((conf > 0.25).sum().data[0])
        tx = Variable(tx)
        ty = Variable(ty)
        tw = Variable(tw)
        th = Variable(th)
        tconf = Variable(tconf)
        tcls = Variable(tcls.view(-1)[cls_mask].long())
        coord_mask = Variable(coord_mask)
        conf_mask = Variable(conf_mask.sqrt())
        cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC))
        cls = cls[cls_mask].view(-1, nC)
        t3 = time.time()
        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x *
            coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y *
            coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w *
            coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h *
            coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf *
            conf_mask) / 2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(
            cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            None
            None
            None
            None
            None
            None
        None
        return loss


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes_in_model(output, conf_thresh, num_classes, anchors,
    num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert output.size(1) == (5 + num_classes) * num_anchors
    h = output.size(2)
    w = output.size(3)
    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w
        ).transpose(0, 1).contiguous().view(5 + num_classes, batch *
        num_anchors * h * w)
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch *
        num_anchors, 1, 1).view(batch * num_anchors * h * w).type_as(output)
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch *
        num_anchors, 1, 1).view(batch * num_anchors * h * w).type_as(output)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step
        ).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step
        ).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch *
        num_anchors * h * w).type_as(output)
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch *
        num_anchors * h * w).type_as(output)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
    det_confs = torch.sigmoid(output[4])
    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].
        transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf,
                            cls_max_conf, cls_max_id]
                        if not only_objectness and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind
                                    ] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


class YoloLayer(nn.Module):
    """ Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    """

    def __init__(self, anchor_mask=[], num_classes=0, anchors=[],
        num_anchors=1, stride=32, model_out=True):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            t0 = time.time()
            nB = output.data.size(0)
            nA = self.num_anchors
            nC = self.num_classes
            nH = output.data.size(2)
            nW = output.data.size(3)
            output = output.view(nB, nA, 5 + nC, nH, nW)
            x = F.sigmoid(output.index_select(2, Variable(torch.LongTensor(
                [0]))).view(nB, nA, nH, nW))
            y = F.sigmoid(output.index_select(2, Variable(torch.LongTensor(
                [1]))).view(nB, nA, nH, nW))
            w = output.index_select(2, Variable(torch.LongTensor([2]))).view(nB
                , nA, nH, nW)
            h = output.index_select(2, Variable(torch.LongTensor([3]))).view(nB
                , nA, nH, nW)
            conf = F.sigmoid(output.index_select(2, Variable(torch.
                LongTensor([4]))).view(nB, nA, nH, nW))
            cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC -
                1, nC).long()))
            cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous(
                ).view(nB * nA * nH * nW, nC)
            t1 = time.time()
            pred_boxes = torch.FloatTensor(4, nB * nA * nH * nW)
            grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB *
                nA, 1, 1).view(nB * nA * nH * nW)
            grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(
                nB * nA, 1, 1).view(nB * nA * nH * nW)
            anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step
                ).index_select(1, torch.LongTensor([0]))
            anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step
                ).index_select(1, torch.LongTensor([1]))
            anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB *
                nA * nH * nW)
            anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB *
                nA * nH * nW)
            pred_boxes[0] = x.data + grid_x
            pred_boxes[1] = y.data + grid_y
            pred_boxes[2] = torch.exp(w.data) * anchor_w
            pred_boxes[3] = torch.exp(h.data) * anchor_h
            pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous(
                ).view(-1, 4))
            t2 = time.time()
            (nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th,
                tconf, tcls) = (build_targets(pred_boxes, target.data, self
                .anchors, nA, nC, nH, nW, self.noobject_scale, self.
                object_scale, self.thresh, self.seen))
            cls_mask = cls_mask == 1
            nProposals = int((conf > 0.25).sum().data[0])
            tx = Variable(tx)
            ty = Variable(ty)
            tw = Variable(tw)
            th = Variable(th)
            tconf = Variable(tconf)
            tcls = Variable(tcls.view(-1)[cls_mask].long())
            coord_mask = Variable(coord_mask)
            conf_mask = Variable(conf_mask.sqrt())
            cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC))
            cls = cls[cls_mask].view(-1, nC)
            t3 = time.time()
            loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x *
                coord_mask, tx * coord_mask) / 2.0
            loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y *
                coord_mask, ty * coord_mask) / 2.0
            loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w *
                coord_mask, tw * coord_mask) / 2.0
            loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h *
                coord_mask, th * coord_mask) / 2.0
            loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, 
                tconf * conf_mask) / 2.0
            loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=
                False)(cls, tcls)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            t4 = time.time()
            if False:
                None
                None
                None
                None
                None
                None
            None
            return loss
        elif self.model_out:
            return output
        else:
            masked_anchors = []
            for m in self.anchor_mask:
                masked_anchors += self.anchors[m * self.anchor_step:(m + 1) *
                    self.anchor_step]
            masked_anchors = [(anchor / self.stride) for anchor in
                masked_anchors]
            boxes = get_region_boxes_in_model(output.data, self.thresh,
                self.num_classes, masked_anchors, len(self.anchor_mask))
            return boxes


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`.         An element at index :math:`(n, k)` contains IoUs between         :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding         box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, (None), :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, (None), 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(bboxes_a[:, (None), :2] - bboxes_a[:, (None), 2:] / 
            2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
        br = torch.min(bboxes_a[:, (None), :2] + bboxes_a[:, (None), 2:] / 
            2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, (None)] + area_b - area_i)


class Yolo_loss(nn.Module):

    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [
            72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5
        (self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y,
            self.anchor_w, self.anchor_h) = [], [], [], [], [], []
        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for
                w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.
                anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32
                )
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3,
                fsize, 1)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3,
                fsize, 1).permute(0, 1, 3, 2)
            anchor_w = torch.from_numpy(masked_anchors[:, (0)]).repeat(batch,
                fsize, fsize, 1).permute(0, 3, 1, 2)
            anchor_h = torch.from_numpy(masked_anchors[:, (1)]).repeat(batch,
                fsize, fsize, 1).permute(0, 3, 1, 2)
            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 +
            self.n_classes)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch)
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        truth_x_all = (labels[:, :, (2)] + labels[:, :, (0)]) / (self.
            strides[output_id] * 2)
        truth_y_all = (labels[:, :, (3)] + labels[:, :, (1)]) / (self.
            strides[output_id] * 2)
        truth_w_all = (labels[:, :, (2)] - labels[:, :, (0)]) / self.strides[
            output_id]
        truth_h_all = (labels[:, :, (3)] - labels[:, :, (1)]) / self.strides[
            output_id]
        truth_i_all = truth_x_all.cpu().numpy()
        truth_j_all = truth_y_all.cpu().numpy()
        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4)
            truth_box[:n, (2)] = truth_w_all[(b), :n]
            truth_box[:n, (3)] = truth_h_all[(b), :n]
            truth_i = truth_i_all[(b), :n]
            truth_j = truth_j_all[(b), :n]
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[
                output_id])
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = (best_n_all == self.anch_masks[output_id][0]) | (
                best_n_all == self.anch_masks[output_id][1]) | (best_n_all ==
                self.anch_masks[output_id][2])
            if sum(best_n_mask) == 0:
                continue
            truth_box[:n, (0)] = truth_x_all[(b), :n]
            truth_box[:n, (1)] = truth_y_all[(b), :n]
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            obj_mask[b] = ~pred_best_iou
            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[(b), (a), (j), (i), :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[
                        b, ti].to(torch.int16)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[
                        b, ti].to(torch.int16)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] /
                        torch.Tensor(self.masked_anchors[output_id])[best_n
                        [ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] /
                        torch.Tensor(self.masked_anchors[output_id])[best_n
                        [ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].cpu().numpy()] = 1
                    tgt_scale[(b), (a), (j), (i), :] = torch.sqrt(2 - 
                        truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                        )
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.
                r_[:2, 4:n_ch]])
            pred = output[(...), :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred,
                labels, batchsize, fsize, n_ch, output_id)
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[(...), 2:4] *= tgt_scale
            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[(...), 2:4] *= tgt_scale
            loss_xy += F.binary_cross_entropy(input=output[(...), :2],
                target=target[(...), :2], weight=tgt_scale * tgt_scale,
                size_average=False)
            loss_wh += F.mse_loss(input=output[(...), 2:4], target=target[(
                ...), 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target
                =target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[(...), 5:],
                target=target[(...), 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average
                =False)
        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Tianxiaomo_pytorch_YOLOv4(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv_Bn_Activation(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'activation': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DownSample1(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_002(self):
        self._check(DownSample2(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_003(self):
        self._check(DownSample3(*[], **{}), [torch.rand([4, 128, 64, 64])], {})

    def test_004(self):
        self._check(DownSample4(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    def test_005(self):
        self._check(DownSample5(*[], **{}), [torch.rand([4, 512, 64, 64])], {})

    def test_006(self):
        self._check(EmptyModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(MaxPoolStride1(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(Mish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(ResBlock(*[], **{'ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(Upsample(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(YoloLayer(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

