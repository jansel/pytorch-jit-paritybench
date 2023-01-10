import sys
_module = sys.modules[__name__]
del sys
config = _module
yolov4_config = _module
cocoapi_evaluator = _module
evaluator = _module
voc_eval = _module
eval_coco = _module
eval_voc = _module
YOLOv4 = _module
model = _module
CSPDarknet53 = _module
backbones = _module
mobilenetv2 = _module
mobilenetv2_CoordAttention = _module
mobilenetv3 = _module
build_model = _module
head = _module
yolo_head = _module
layers = _module
activate = _module
attention_layers = _module
blocks_module = _module
conv_module = _module
global_context_block = _module
learnable_semantic_fusion = _module
loss = _module
yolo_loss = _module
onnx_transform = _module
train = _module
utils = _module
coco = _module
coco_to_voc = _module
cocodataset = _module
cosine_lr_scheduler = _module
data_augment = _module
datasets = _module
datasets_coco = _module
flops_counter = _module
get_gt_txt = _module
get_map = _module
gpu = _module
heatmap = _module
imshowAtt = _module
kmeans = _module
log = _module
modelsize = _module
tools = _module
torch_utils = _module
utils = _module
visualize = _module
voc = _module
xml_to_txt = _module
video_test = _module

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


from torch.autograd import Variable


import time


from collections import defaultdict


import logging


import torch.nn as nn


import torch.nn.functional as F


import math


import numpy as np


from torch.nn import init


from itertools import repeat


from torch._jit_internal import Optional


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


from torch import nn


import torch.optim as optim


from torch.utils.data import DataLoader


import random


from torch.utils.data import Dataset


from torchvision import transforms


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)


class SpatialPyramidPooling(nn.Module):

    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        self.head_conv = nn.Sequential(Conv(feature_channels[-1], feature_channels[-1] // 2, 1), Conv(feature_channels[-1] // 2, feature_channels[-1], 3), Conv(feature_channels[-1], feature_channels[-1] // 2, 1))
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x] + features, dim=1)
        return features

    def __initialize_weights(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                None


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(Conv(in_channels, out_channels, 1), nn.Upsample(scale_factor=scale))

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()
        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)


class PANet(nn.Module):

    def __init__(self, feature_channels):
        super(PANet, self).__init__()
        self.feature_transform3 = Conv(feature_channels[0], feature_channels[0] // 2, 1)
        self.feature_transform4 = Conv(feature_channels[1], feature_channels[1] // 2, 1)
        self.resample5_4 = Upsample(feature_channels[2] // 2, feature_channels[1] // 2)
        self.resample4_3 = Upsample(feature_channels[1] // 2, feature_channels[0] // 2)
        self.resample3_4 = Downsample(feature_channels[0] // 2, feature_channels[1] // 2)
        self.resample4_5 = Downsample(feature_channels[1] // 2, feature_channels[2] // 2)
        self.downstream_conv5 = nn.Sequential(Conv(feature_channels[2] * 2, feature_channels[2] // 2, 1), Conv(feature_channels[2] // 2, feature_channels[2], 3), Conv(feature_channels[2], feature_channels[2] // 2, 1))
        self.downstream_conv4 = nn.Sequential(Conv(feature_channels[1], feature_channels[1] // 2, 1), Conv(feature_channels[1] // 2, feature_channels[1], 3), Conv(feature_channels[1], feature_channels[1] // 2, 1), Conv(feature_channels[1] // 2, feature_channels[1], 3), Conv(feature_channels[1], feature_channels[1] // 2, 1))
        self.downstream_conv3 = nn.Sequential(Conv(feature_channels[0], feature_channels[0] // 2, 1), Conv(feature_channels[0] // 2, feature_channels[0], 3), Conv(feature_channels[0], feature_channels[0] // 2, 1), Conv(feature_channels[0] // 2, feature_channels[0], 3), Conv(feature_channels[0], feature_channels[0] // 2, 1))
        self.upstream_conv4 = nn.Sequential(Conv(feature_channels[1], feature_channels[1] // 2, 1), Conv(feature_channels[1] // 2, feature_channels[1], 3), Conv(feature_channels[1], feature_channels[1] // 2, 1), Conv(feature_channels[1] // 2, feature_channels[1], 3), Conv(feature_channels[1], feature_channels[1] // 2, 1))
        self.upstream_conv5 = nn.Sequential(Conv(feature_channels[2], feature_channels[2] // 2, 1), Conv(feature_channels[2] // 2, feature_channels[2], 3), Conv(feature_channels[2], feature_channels[2] // 2, 1), Conv(feature_channels[2] // 2, feature_channels[2], 3), Conv(feature_channels[2], feature_channels[2] // 2, 1))
        self.__initialize_weights()

    def forward(self, features):
        features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]
        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5)], dim=1))
        downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4)], dim=1))
        upstream_feature4 = self.upstream_conv4(torch.cat([self.resample3_4(downstream_feature3), downstream_feature4], dim=1))
        upstream_feature5 = self.upstream_conv5(torch.cat([self.resample4_5(upstream_feature4), downstream_feature5], dim=1))
        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def __initialize_weights(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                None


class PredictNet(nn.Module):

    def __init__(self, feature_channels, target_channels):
        super(PredictNet, self).__init__()
        self.predict_conv = nn.ModuleList([nn.Sequential(Conv(feature_channels[i] // 2, feature_channels[i], 3), nn.Conv2d(feature_channels[i], target_channels, 1)) for i in range(len(feature_channels))])
        self.__initialize_weights()

    def forward(self, features):
        predicts = [predict_conv(feature) for predict_conv, feature in zip(self.predict_conv, features)]
        return predicts

    def __initialize_weights(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                None


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes):
        super(ContextBlock2d, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        beta1 = context_mask
        beta2 = torch.transpose(beta1, 1, 2)
        atten = torch.matmul(beta2, beta1)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return context, atten

    def forward(self, x):
        context, atten = self.spatial_pool(x)
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term
        return out, atten


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size * 3))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_planes = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


class DOConv2d(Module):
    """
    DOConv2d can be used as an alternative for torch.nn.Conv2d.
    The interface is similar to that of Conv2d, with one exception:
         1. D_mul: the depth multiplier for the over-parameterization.
    Note that the groups parameter switchs between DO-Conv (groups=1),
    DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', 'in_channels', 'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)
            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:
                self.D_diag = Parameter(D_diag, requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode), weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = self.out_channels, self.in_channels // self.groups, M, N
        if M * N > 1:
            D = self.D + self.D_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
        else:
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)


class Mish(nn.Module):

    def __init__(self):
        super(Mish).__init__()

    def forward(self, x):
        x = x * torch.tanh(F.softplus(x))
        return x


activate_name = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'mish': Mish}


norm_name = {'bn': nn.BatchNorm2d}


class Convolutional(nn.Module):

    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, norm=None, activate=None):
        super(Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        if cfg.CONV_TYPE['TYPE'] == 'DO_CONV':
            self.__conv = DOConv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=not norm)
        else:
            self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=not norm)
        if norm:
            assert norm in norm_name.keys()
            if norm == 'bn':
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == 'leaky':
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == 'relu':
                self.__activate = activate_name[activate](inplace=True)

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original = x
        x = self.avg_pool(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return original * x


class CSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None, residual_activation='linear'):
        super(CSPBlock, self).__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.block = nn.Sequential(Convolutional(in_channels, hidden_channels, 1), Convolutional(hidden_channels, out_channels, 3))
        self.activation = activate_name[residual_activation]
        self.attention = cfg.ATTENTION['TYPE']
        if self.attention == 'SEnet':
            self.attention_module = SEModule(out_channels)
        elif self.attention == 'CBAM':
            self.attention_module = CBAM(out_channels)
        else:
            self.attention = None

    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.attention is not None:
            out = self.attention_module(out)
        out += residual
        return out


class CSPFirstStage(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()
        self.downsample_conv = Convolutional(in_channels, out_channels, 3, stride=2)
        self.split_conv0 = Convolutional(out_channels, out_channels, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels, 1)
        self.blocks_conv = nn.Sequential(CSPBlock(out_channels, out_channels, in_channels), Convolutional(out_channels, out_channels, 1))
        self.concat_conv = Convolutional(out_channels * 2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class CSPStage(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()
        self.downsample_conv = Convolutional(in_channels, out_channels, 3, stride=2)
        self.split_conv0 = Convolutional(out_channels, out_channels // 2, 1)
        self.split_conv1 = Convolutional(out_channels, out_channels // 2, 1)
        self.blocks_conv = nn.Sequential(*[CSPBlock(out_channels // 2, out_channels // 2) for _ in range(num_blocks)], Convolutional(out_channels // 2, out_channels // 2, 1))
        self.concat_conv = Convolutional(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)
        return x


class CSPDarknet53(nn.Module):

    def __init__(self, stem_channels=32, feature_channels=[64, 128, 256, 512, 1024], num_features=3, weight_path=None, resume=False):
        super(CSPDarknet53, self).__init__()
        self.stem_conv = Convolutional(3, stem_channels, 3)
        self.stages = nn.ModuleList([CSPFirstStage(stem_channels, feature_channels[0]), CSPStage(feature_channels[0], feature_channels[1], 2), CSPStage(feature_channels[1], feature_channels[2], 8), CSPStage(feature_channels[2], feature_channels[3], 8), CSPStage(feature_channels[3], feature_channels[4], 4)])
        self.feature_channels = feature_channels
        self.num_features = num_features
        if weight_path and not resume:
            self.load_CSPdarknet_weights(weight_path)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[-self.num_features:]

    def _initialize_weights(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                None

    def load_CSPdarknet_weights(self, weight_file, cutoff=52):
        """https://github.com/ultralytics/yolov3/blob/master/models.py"""
        None
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                conv_layer = m._Convolutional__conv
                if m.norm == 'bn':
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    None
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                None


def _BuildCSPDarknet53(weight_path, resume):
    model = CSPDarknet53(weight_path=weight_path, resume=resume)
    return model, model.feature_channels[-3:]


class FeatureExtractor(nn.Module):

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is 'features':
                for f_name, f_module in module._modules.items():
                    x = f_module(x)
                    if f_name in self.extracted_layers:
                        outputs.append(x)
            if name is 'conv':
                x = module(x)
                if name in self.extracted_layers:
                    outputs.append(x)
        return outputs


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, _make_divisible(channel // reduction, 8)), nn.ReLU(inplace=True), nn.Linear(_make_divisible(channel // reduction, 8), channel), h_sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class InvertedResidual(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        if inp == hidden_dim:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), h_swish() if use_hs else nn.ReLU(inplace=True), SELayer(hidden_dim) if use_se else nn.Identity(), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), h_swish() if use_hs else nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), SELayer(hidden_dim) if use_se else nn.Identity(), h_swish() if use_hs else nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish())


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), h_swish())


class MBV2_CA(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MBV2_CA, self).__init__()
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(output_channel, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2_CoordAttention(nn.Module):

    def __init__(self, weight_path=None, resume=False, extract_list=['6', '13', 'conv'], feature_channels=[32, 96, 1280], width_mult=1.0):
        super(MobileNetV2_CoordAttention, self).__init__()
        self.feature_channels = feature_channels
        self.__submodule = MBV2_CA(width_mult=width_mult)
        if weight_path and not resume:
            None
            load_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            model_dict = self.__submodule.state_dict()
            pretrained_dict = {k: v for k, v in load_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.__submodule.load_state_dict(model_dict)
            del pretrained_dict
            None
        self.__extractor = FeatureExtractor(self.__submodule, extract_list)

    def forward(self, x):
        return self.__extractor(x)


def _BuildMobileNetV2_CoordAttention(weight_path, resume):
    model = MobileNetV2_CoordAttention(weight_path=weight_path, resume=resume)
    return model, model.feature_channels[-3:]


class _MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(_MobileNetV2, self).__init__()
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                None
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                None


class MobilenetV2(nn.Module):

    def __init__(self, weight_path=None, resume=False, extract_list=['6', '13', 'conv'], feature_channels=[32, 96, 1280], width_mult=1.0):
        super(MobilenetV2, self).__init__()
        self.feature_channels = feature_channels
        self.__submodule = _MobileNetV2(width_mult=width_mult)
        if weight_path and not resume:
            None
            pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            model_dict = self.__submodule.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.__submodule.load_state_dict(model_dict)
            del pretrained_dict
            None
        self.__extractor = FeatureExtractor(self.__submodule, extract_list)

    def forward(self, x):
        return self.__extractor(x)


def _BuildMobilenetV2(weight_path, resume):
    model = MobilenetV2(weight_path=weight_path, resume=resume)
    return model, model.feature_channels[-3:]


class _MobileNetV3(nn.Module):

    def __init__(self, width_mult=1.0):
        super(_MobileNetV3, self).__init__()
        self.cfgs = [[3, 1, 16, 1, 0, 2], [3, 4.5, 24, 0, 0, 2], [3, 3.67, 24, 0, 0, 1], [5, 4, 40, 1, 1, 2], [5, 6, 40, 1, 1, 1], [5, 6, 40, 1, 1, 1], [5, 3, 48, 1, 1, 1], [5, 3, 48, 1, 1, 1], [5, 6, 96, 1, 1, 2], [5, 6, 96, 1, 1, 1], [5, 6, 96, 1, 1, 1]]
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1024 * width_mult, 8) if width_mult > 1.0 else 1024
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                None
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                None
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                None


class MobilenetV3(nn.Module):

    def __init__(self, extract_list=['3', '8', 'conv'], weight_path=None, resume=False, width_mult=1.0, feature_channels=[24, 48, 1024]):
        super(MobilenetV3, self).__init__()
        self.feature_channels = feature_channels
        self.__submodule = _MobileNetV3(width_mult=width_mult)
        if weight_path and not resume:
            None
            pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
            model_dict = self.__submodule.state_dict()
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                if 'features' in k:
                    new_state_dict[k] = v
            model_dict.update(new_state_dict)
            self.__submodule.load_state_dict(model_dict)
            del pretrained_dict
            None
        self.__extractor = FeatureExtractor(self.__submodule, extract_list)

    def forward(self, x):
        return self.__extractor(x)


def _BuildMobilenetV3(weight_path, resume):
    model = MobilenetV3(weight_path=weight_path, resume=resume)
    return model, model.feature_channels[-3:]


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


class swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)
        return x * y


class CoordAtt(nn.Module):

    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        y = identity * x_w * x_h
        return y


class Yolo_head(nn.Module):

    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()
        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)
        p_de = self.__decode(p.clone())
        return p, p_de

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        anchors = 1.0 * self.__anchors
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float()
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = torch.exp(conv_raw_dwdh) * anchors * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox


class Build_Model(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """

    def __init__(self, weight_path=None, resume=False, showatt=False):
        super(Build_Model, self).__init__()
        self.__showatt = showatt
        self.__anchors = torch.FloatTensor(cfg.MODEL['ANCHORS'])
        self.__strides = torch.FloatTensor(cfg.MODEL['STRIDES'])
        if cfg.TRAIN['DATA_TYPE'] == 'VOC':
            self.__nC = cfg.VOC_DATA['NUM']
        elif cfg.TRAIN['DATA_TYPE'] == 'COCO':
            self.__nC = cfg.COCO_DATA['NUM']
        else:
            self.__nC = cfg.Customer_DATA['NUM']
        self.__out_channel = cfg.MODEL['ANCHORS_PER_SCLAE'] * (self.__nC + 5)
        self.__yolov4 = YOLOv4(weight_path=weight_path, out_channels=self.__out_channel, resume=resume, showatt=showatt)
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

    def forward(self, x):
        out = []
        [x_s, x_m, x_l], atten = self.__yolov4(x)
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            if self.__showatt:
                return p, torch.cat(p_d, 0), atten
            return p, torch.cat(p_d, 0)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class Residual_block(nn.Module):

    def __init__(self, filters_in, filters_out, filters_medium):
        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.__conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3, stride=1, pad=1, norm='bn', activate='leaky')

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r
        return out


class FusionLayer(nn.Module):

    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss


class YoloV4Loss(nn.Module):

    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        super(YoloV4Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 150, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides
        loss_s, loss_s_ciou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox, sbboxes, strides[0])
        loss_m, loss_m_ciou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox, mbboxes, strides[1])
        loss_l, loss_l_ciou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox, lbboxes, strides[2])
        loss = loss_l + loss_m + loss_s
        loss_ciou = loss_s_ciou + loss_m_ciou + loss_l_ciou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls
        return loss, loss_ciou, loss_conf, loss_cls

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction='none')
        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]
        p_d_xywh = p_d[..., :4]
        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 6:]
        label_mix = label[..., 5:6]
        ciou = tools.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / img_size ** 2
        loss_ciou = label_obj_mask * bbox_loss_scale * (1.0 - ciou) * label_mix
        iou = tools.CIOU_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()
        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) + label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix
        loss_ciou = torch.sum(loss_ciou) / batch_size
        loss_conf = torch.sum(loss_conf) / batch_size
        loss_cls = torch.sum(loss_cls) / batch_size
        loss = loss_ciou + loss_conf + loss_cls
        return loss, loss_ciou, loss_conf, loss_cls


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CoordAtt,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Downsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureExtractor,
     lambda: ([], {'submodule': _mock_layer(), 'extracted_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FusionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([6, 4, 4, 4])], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'hidden_dim': 4, 'oup': 4, 'kernel_size': 4, 'stride': 1, 'use_se': 4, 'use_hs': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
    (MobilenetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialPyramidPooling,
     lambda: ([], {'feature_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_MobileNetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (h_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (h_swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_argusswift_YOLOv4_pytorch(_paritybench_base):
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

