import sys
_module = sys.modules[__name__]
del sys
base_model = _module
resnet = _module
xception = _module
config = _module
dataloader = _module
BaseDataset = _module
datasets = _module
ade = _module
ade = _module
cityscapes = _module
voc = _module
dfn = _module
engine = _module
engine = _module
evaluator = _module
logger = _module
lr_policy = _module
version = _module
eval = _module
model = _module
deeperlab = _module
depend = _module
seg_opr = _module
loss_opr = _module
metric = _module
parallel_apply = _module
seg_oprs = _module
sgd = _module
sync_bn = _module
comm = _module
functions = _module
parallel = _module
parallel_apply = _module
src = _module
cpu = _module
setup = _module
gpu = _module
setup = _module
syncbn = _module
train = _module
utils = _module
board = _module
csv_logger = _module
img_utils = _module
init_func = _module
pyt_utils = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import functools


import torch.nn as nn


import torch.nn.functional as F


import time


import math


import numpy as np


import torch.utils.model_zoo as model_zoo


import torch


from torch.utils import data


import torch.utils.data as data


import scipy.io as sio


from torch.utils.checkpoint import checkpoint


import torch.distributed as dist


import torch.multiprocessing as mp


import scipy.ndimage as nd


from torch.cuda._utils import _get_device_index


from collections import OrderedDict


from torch.optim.sgd import SGD


from torch.autograd import Variable


from torch.autograd import Function


from torch.nn.parallel.data_parallel import DataParallel


import torch.cuda.comm as comm


from torch.nn.parallel._functions import Broadcast


import queue


from torch.utils.cpp_extension import load


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import collections


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.functional import batch_norm


from torch.nn.parallel._functions import ReduceAddCoalesced


import torch.backends.cudnn as cudnn


from collections import defaultdict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bn_eps=1e-05, bn_momentum=0.1, downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

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
        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=inplace), nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0], inplace, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3], inplace, stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True, stride=1, bn_eps=1e-05, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion, eps=bn_eps, momentum=bn_momentum))
        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, inplace=inplace))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        return blocks


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, norm_layer=nn.BatchNorm2d, eps=1e-05, momentum=0.1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = norm_layer(out_filters)
        else:
            self.skip = None
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(norm_layer(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(norm_layer(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, inplane=3, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, bn_momentum=0.1, inplace=True):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(inplane, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32, eps=bn_eps, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = norm_layer(64, eps=bn_eps, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.block1 = Block(64, 128, 2, 2, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, norm_layer=norm_layer, eps=bn_eps, momentum=bn_momentum, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = norm_layer(1536, eps=bn_eps, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=inplace)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = norm_layer(2048, eps=bn_eps, momentum=bn_momentum)
        self.relu4 = nn.ReLU(inplace)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block1(x)
        low_feature = x
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
        x = self.relu3(x)
        return low_feature, x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        return x


class SELayer(nn.Module):

    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_planes, out_planes // reduction), nn.ReLU(inplace=True), nn.Linear(out_planes // reduction, out_planes), nn.Sigmoid())
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2
        return fm


class ConvBnRelu(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-05, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class RefineResidual(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, has_bias=False, has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-05):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1, ksize // 2, has_bias=has_bias, norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=ksize // 2, dilation=1, bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class DFNHead(nn.Module):

    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(DFNHead, self).__init__()
        self.rrb = RefineResidual(in_planes, out_planes * 9, 3, has_bias=False, has_relu=False, norm_layer=norm_layer)
        self.conv = nn.Conv2d(out_planes * 9, out_planes, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        x = self.rrb(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return x


def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()
    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in missing_keys)))
    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(', '.join('{}'.format(k) for k in unexpected_keys)))
    del state_dict
    t_end = time.time()
    logger.info('Load model, Time usage:\n\tIO: {}, initialize parameters: {}'.format(t_ioend - t_start, t_end - t_ioend))
    return model


def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


class DFN(nn.Module):

    def __init__(self, out_planes, criterion, aux_criterion, alpha, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(DFN, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum, deep_stem=False, stem_width=64)
        self.business_layer = []
        smooth_inner_channel = 512
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(2048, smooth_inner_channel, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer))
        self.business_layer.append(self.global_context)
        stage = [2048, 1024, 512, 256]
        self.smooth_pre_rrbs = []
        self.cabs = []
        self.smooth_aft_rrbs = []
        self.smooth_heads = []
        for i, channel in enumerate(stage):
            self.smooth_pre_rrbs.append(RefineResidual(channel, smooth_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.cabs.append(ChannelAttention(smooth_inner_channel * 2, smooth_inner_channel, 1))
            self.smooth_aft_rrbs.append(RefineResidual(smooth_inner_channel, smooth_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.smooth_heads.append(DFNHead(smooth_inner_channel, out_planes, scale=2 ** (5 - i), norm_layer=norm_layer))
        stage.reverse()
        border_inner_channel = 21
        self.border_pre_rrbs = []
        self.border_aft_rrbs = []
        self.border_heads = []
        for i, channel in enumerate(stage):
            self.border_pre_rrbs.append(RefineResidual(channel, border_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.border_aft_rrbs.append(RefineResidual(border_inner_channel, border_inner_channel, 3, has_bias=False, has_relu=True, norm_layer=norm_layer))
            self.border_heads.append(DFNHead(border_inner_channel, 1, 4, norm_layer=norm_layer))
        self.smooth_pre_rrbs = nn.ModuleList(self.smooth_pre_rrbs)
        self.cabs = nn.ModuleList(self.cabs)
        self.smooth_aft_rrbs = nn.ModuleList(self.smooth_aft_rrbs)
        self.smooth_heads = nn.ModuleList(self.smooth_heads)
        self.border_pre_rrbs = nn.ModuleList(self.border_pre_rrbs)
        self.border_aft_rrbs = nn.ModuleList(self.border_aft_rrbs)
        self.border_heads = nn.ModuleList(self.border_heads)
        self.business_layer.append(self.smooth_pre_rrbs)
        self.business_layer.append(self.cabs)
        self.business_layer.append(self.smooth_aft_rrbs)
        self.business_layer.append(self.smooth_heads)
        self.business_layer.append(self.border_pre_rrbs)
        self.business_layer.append(self.border_aft_rrbs)
        self.business_layer.append(self.border_heads)
        self.criterion = criterion
        self.aux_criterion = aux_criterion
        self.alpha = alpha

    def forward(self, data, label=None, aux_label=None):
        blocks = self.backbone(data)
        """
        >> Block we have 4 conv1 conv2 conv3 conv4
        conv1 -> 256 1/4
        conv2 -> 512 1/8
        conv3 ->1024 1/16
        conv4 ->2048 1/32
        """
        blocks.reverse()
        global_context = self.global_context(blocks[0])
        global_context = F.interpolate(global_context, size=blocks[0].size()[2:], mode='bilinear', align_corners=True)
        last_fm = global_context
        pred_out = []
        for i, (fm, pre_rrb, cab, aft_rrb, head) in enumerate(zip(blocks, self.smooth_pre_rrbs, self.cabs, self.smooth_aft_rrbs, self.smooth_heads)):
            fm = pre_rrb(fm)
            fm = cab(fm, last_fm)
            fm = aft_rrb(fm)
            pred_out.append(head(fm))
            if i != 3:
                last_fm = F.interpolate(fm, scale_factor=2, mode='bilinear', align_corners=True)
        blocks.reverse()
        last_fm = None
        boder_out = []
        """
        conv1 ---RRB------------>head
                       |
        conv2 ---RRB---+--RRB--->head
                            |
        conv3 ---RRB---+--RRB--->head
                            |
        conv4 ---RRB---+--RRB--->head
        """
        for i, (fm, pre_rrb, aft_rrb, head) in enumerate(zip(blocks, self.border_pre_rrbs, self.border_aft_rrbs, self.border_heads)):
            fm = pre_rrb(fm)
            if last_fm is not None:
                fm = F.interpolate(fm, scale_factor=2 ** i, mode='bilinear', align_corners=True)
                last_fm = last_fm + fm
                last_fm = aft_rrb(last_fm)
            else:
                last_fm = fm
            boder_out.append(head(last_fm))
        if label is not None and aux_label is not None:
            loss0 = self.criterion(pred_out[0], label)
            loss1 = self.criterion(pred_out[1], label)
            loss2 = self.criterion(pred_out[2], label)
            loss3 = self.criterion(pred_out[3], label)
            aux_loss0 = self.aux_criterion(boder_out[0], aux_label)
            aux_loss1 = self.aux_criterion(boder_out[1], aux_label)
            aux_loss2 = self.aux_criterion(boder_out[2], aux_label)
            aux_loss3 = self.aux_criterion(boder_out[3], aux_label)
            loss = loss0 + loss1 + loss2 + loss3
            aux_loss = aux_loss0 + aux_loss1 + aux_loss2 + aux_loss3
            return loss + self.alpha * aux_loss
        return F.log_softmax(pred_out[-1], dim=1)


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):

    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'deeperlab':
            inplanes = 728
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 3, 6, 12]
        elif output_stride == 8:
            dilations = [1, 6, 12, 18]
        else:
            raise NotImplementedError
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(inplanes, 256, 1, stride=1, bias=False), BatchNorm(256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class space_to_dense(nn.Module):

    def __init__(self, stride):
        super(space_to_dense, self).__init__()
        self.stride = stride

    def forward(self, input):
        assert len(input.shape) == 4, 'input tensor must be 4 dimenson'
        stride = self.stride
        B, C, W, H = input.shape
        assert W % stride == 0 and H % stride == 0, 'the W = {} or H = {} must be divided by {}'.format(W, H, stride)
        ws = W // stride
        hs = H // stride
        x = input.view(B, C, hs, stride, ws, stride).transpose(3, 4).contiguous()
        x = x.view(B, C, hs * ws, stride * stride).transpose(2, 3).contiguous()
        x = x.view(B, C, stride * stride, hs, ws).transpose(1, 2).contiguous()
        x = x.view(B, stride * stride * C, hs, ws)
        return x


class dense_to_space(nn.Module):

    def __init__(self, stride):
        super(dense_to_space, self).__init__()
        self.stride = stride
        self.ps = torch.nn.PixelShuffle(stride)

    def forward(self, input):
        return self.ps(input)


class deeperlab_seg_head(nn.Module):

    def __init__(self, inplane, outplane, scale=4, norm_layer=nn.BatchNorm2d):
        super(deeperlab_seg_head, self).__init__()
        self.conv = ConvBnRelu(inplane, 256, 7, 1, 3, norm_layer=norm_layer, bn_eps=config.bn_eps)
        self.conv_seg = nn.Conv2d(256, outplane, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_seg(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return x


class deeperlab(nn.Module):

    def __init__(self, inplane, outplane, criterion=None, aux_criterion=None, area_alpa=None, pretrained_model=None, norm_layer=nn.BatchNorm2d, detection=False):
        super(deeperlab, self).__init__()
        self.backbone = xception.xception71(pretrained_model, inplane=inplane, norm_layer=norm_layer, bn_eps=config.bn_eps, bn_momentum=config.bn_momentum, inplace=True)
        self.business_layer = []
        self.s2d = space_to_dense(4)
        self.d2s = torch.nn.PixelShuffle(upscale_factor=4)
        self.aspp = ASPP('deeperlab', 8, norm_layer)
        self.conv1 = ConvBnRelu(128, 32, 1, 1, 0, norm_layer=norm_layer, bn_eps=config.bn_eps)
        self.conv2 = ConvBnRelu(768, 4096, 3, 1, 1, norm_layer=norm_layer, bn_eps=config.bn_eps)
        self.conv3 = ConvBnRelu(4096, 4096, 3, 1, 1, norm_layer=norm_layer, bn_eps=config.bn_eps)
        self.seg_conv = deeperlab_seg_head(256, outplane, 4, norm_layer=norm_layer)
        self.business_layer.append(self.s2d)
        self.business_layer.append(self.d2s)
        self.business_layer.append(self.aspp)
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.conv2)
        self.business_layer.append(self.conv3)
        self.business_layer.append(self.seg_conv)
        self.criterion = criterion

    def forward(self, input, label=None, aux_label=None):
        low_level, high_level = self.backbone(input)
        high_level = self.aspp(high_level)
        low_level = self.conv1(low_level)
        low_level = self.s2d(low_level)
        decode = torch.cat((high_level, low_level), dim=1)
        decode = self.conv2(decode)
        decode = self.conv3(decode)
        decode = self.d2s(decode)
        pre = self.seg_conv(decode)
        if label is not None:
            loss = self.criterion(pre, label)
            return loss
        return F.log_softmax(pre, dim=1)


class SigmoidFocalLoss(nn.Module):

    def __init__(self, ignore_label, gamma=2.0, alpha=0.25, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = target.ne(self.ignore_label).float()
        target = mask * target
        onehot = target.view(b, -1, 1)
        pos_part = (1 - pred_sigmoid) ** self.gamma * torch.log(pred_sigmoid + 0.0001)
        neg_part = pred_sigmoid ** self.gamma * torch.log(1 - pred_sigmoid + 0.0001)
        loss = -(self.alpha * pos_part * (onehot == 1).float() + (1 - self.alpha) * neg_part * (onehot == 0).float()).sum(dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class ProbOhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256, down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)
        return self.criterion(pred, target)


class BootstrappedCrossEntropy(nn.Module):

    def __init__(self, K=0.15, criterion=None):
        super(BootstrappedCrossEntropy, self).__init__()
        assert criterion != None, 'you must give a criterion function'
        self.criterion = criterion
        self.K = K

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        num = int(self.K * B * H * W)
        loss = self.criterion(pred, target)
        loss = loss.view(-1)
        tk = torch.argsort(loss, descending=True)
        TK = loss[tk[num - 1]]
        loss = loss[loss >= TK]
        return loss.mean()


class SeparableConvBnRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, has_relu=True, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)
        return inputs


class BNRefine(nn.Module):

    def __init__(self, in_planes, out_planes, ksize, has_bias=False, has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-05):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1, ksize // 2, has_bias=has_bias, norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize, stride=1, padding=ksize // 2, dilation=1, bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


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

        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


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


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


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

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
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
        assert results[0][0] == 0, 'The first result should belongs to the master.'
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


_ChildMessage = collections.namedtuple('Message', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _batchnormtrain(Function):

    @staticmethod
    def forward(ctx, input, mean, std, gamma, beta):
        ctx.save_for_backward(input, mean, std, gamma, beta)
        if input.is_cuda:
            output = gpu.batchnorm_forward(input, mean, std, gamma, beta)
        else:
            output = cpu.batchnorm_forward(input, mean, std, gamma, beta)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, mean, std, gamma, beta = ctx.saved_variables
        if gradOutput.is_cuda:
            gradInput, gradMean, gradStd, gradGamma, gradBeta = gpu.batchnorm_backward(gradOutput, input, mean, std, gamma, beta, True)
        else:
            raise NotImplemented
        return gradInput, gradMean, gradStd, gradGamma, gradBeta


def batchnormtrain(input, mean, std, gamma, beta):
    """Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.

    .. _encoding.batchnormtrain:

    .. math::

        y = \\frac{x - \\mu[x]}{ \\sqrt{var[x] + \\epsilon}} * \\gamma + \\beta

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    """
    return _batchnormtrain.apply(input, mean, std, gamma, beta)


class _sum_square(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            xsum, xsqusum = gpu.sumsquare_forward(input)
        else:
            xsum, xsqusum = cpu.sumsquare_forward(input)
        return xsum, xsqusum

    @staticmethod
    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_variables
        if input.is_cuda:
            gradInput = gpu.sumsquare_backward(input, gradSum, gradSquare)
        else:
            raise NotImplemented
        return gradInput


def sum_square(input):
    """Calculate sum of elements and sum of squares for Batch Normalization"""
    return _sum_square.apply(input)


class _SyncBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not self.training:
            return batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input_shape[0], self.num_features, -1)
        N = input.size(0) * input.size(2)
        xsum, xsqsum = sum_square(input)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(xsum, xsqsum, N))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(xsum, xsqsum, N))
        return batchnormtrain(input, mean, 1.0 / inv_std, self.weight, self.bias).view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        return mean, (bias_var + self.eps) ** -0.5


class BatchNorm2d(_SyncBatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. note::
        We adapt the awesome python API from another `PyTorch SyncBN Implementation
        <https://github.com/vacancy/Synchronized-BatchNorm-PyTorch>`_ and provide
        efficient CUDA backend.

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
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*

    Examples:
        >>> m = BatchNorm2d(100)
        >>> net = torch.nn.DataParallel(m)
        >>> encoding.parallel.patch_replication_callback(net)
        >>> output = net(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm1d(_SyncBatchNorm):
    """Please see the docs in :class:`encoding.nn.BatchNorm2d`"""

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm3d(_SyncBatchNorm):
    """Please see the docs in :class:`encoding.nn.BatchNorm2d`"""

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(BatchNorm3d, self)._check_input_dim(input)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnRelu,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'ksize': 4, 'stride': 1, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DFNHead,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DataParallelModel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConvBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SigmoidFocalLoss,
     lambda: ([], {'ignore_label': 4}),
     lambda: ([torch.rand([4, 16]), torch.rand([4, 4, 4])], {}),
     True),
    (Xception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_ASPPModule,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1, 'BatchNorm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (dense_to_space,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (space_to_dense,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lingtengqiu_Deeperlab_pytorch(_paritybench_base):
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

