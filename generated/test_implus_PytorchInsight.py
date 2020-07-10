import sys
_module = sys.modules[__name__]
del sys
flops_counter = _module
imagenet = _module
imagenet_fast = _module
imagenet_mobile = _module
common_head = _module
resnet_bam = _module
resnet_cbam = _module
resnet_old = _module
resnet_se = _module
resnet_sge = _module
resnet_sk = _module
resnet_ws = _module
shufflenetv2 = _module
shufflenetv2_bng2 = _module
shufflenetv2_gl4gbn = _module
shufflenetv2_se = _module
layers = _module
utils = _module
eval = _module
logger = _module
misc = _module
progress = _module
bar = _module
counter = _module
helpers = _module
spinner = _module
setup = _module
test_progress = _module
visualize = _module
ptr = _module
cascade_rcnn_r101_fpn_20e = _module
cascade_rcnn_r101_fpn_20e_pretrain_sge_resnet101 = _module
cascade_rcnn_r50_fpn_20e = _module
cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50 = _module
faster_rcnn_r101_fpn_2x = _module
faster_rcnn_r101_fpn_2x_pretrain_sge_resnet101 = _module
faster_rcnn_r50_fpn_2x = _module
faster_rcnn_r50_fpn_2x_pretrain_sge_resnet50 = _module
mask_rcnn_r101_fpn_2x = _module
mask_rcnn_r101_fpn_2x_pretrain_sge_resnet101 = _module
mask_rcnn_r50_fpn_2x = _module
mask_rcnn_r50_fpn_2x_pretrain_sge_resnet50 = _module
retinanet_r101_fpn_2x = _module
retinanet_r101_fpn_2x_pretrain_cbam_resnet101 = _module
retinanet_r101_fpn_2x_pretrain_gc_resnet101 = _module
retinanet_r101_fpn_2x_pretrain_se_resnet101 = _module
retinanet_r101_fpn_2x_pretrain_sge_resnet101 = _module
retinanet_r50_fpn_2x = _module
retinanet_r50_fpn_2x_pretrain_bam_resnet50 = _module
retinanet_r50_fpn_2x_pretrain_cbam_resnet50 = _module
retinanet_r50_fpn_2x_pretrain_gc_resnet50 = _module
retinanet_r50_fpn_2x_pretrain_se_resnet50 = _module
retinanet_r50_fpn_2x_pretrain_sge_resnet50 = _module
retinanet_r50_fpn_2x_pretrain_sk_resnet50 = _module
mmdet = _module
apis = _module
env = _module
inference = _module
train = _module
core = _module
anchor = _module
anchor_generator = _module
anchor_target = _module
bbox = _module
assign_sampling = _module
assigners = _module
assign_result = _module
base_assigner = _module
max_iou_assigner = _module
bbox_target = _module
geometry = _module
samplers = _module
base_sampler = _module
combined_sampler = _module
instance_balanced_pos_sampler = _module
iou_balanced_neg_sampler = _module
ohem_sampler = _module
pseudo_sampler = _module
random_sampler = _module
sampling_result = _module
transforms = _module
evaluation = _module
bbox_overlaps = _module
class_names = _module
coco_utils = _module
eval_hooks = _module
mean_ap = _module
recall = _module
loss = _module
losses = _module
mask = _module
mask_target = _module
post_processing = _module
bbox_nms = _module
merge_augs = _module
dist_utils = _module
datasets = _module
coco = _module
concat_dataset = _module
custom = _module
extra_aug = _module
loader = _module
build_loader = _module
sampler = _module
repeat_dataset = _module
transforms = _module
utils = _module
voc = _module
xml_style = _module
models = _module
anchor_heads = _module
anchor_head = _module
retina_head = _module
rpn_head = _module
ssd_head = _module
backbones = _module
bam = _module
cbam = _module
global_context = _module
resnet = _module
resnet_bam = _module
resnet_cbam = _module
resnet_gc = _module
resnet_se = _module
resnet_sge = _module
resnet_sk = _module
resnext = _module
ssd_vgg = _module
bbox_heads = _module
bbox_head = _module
convfc_bbox_head = _module
builder = _module
detectors = _module
base = _module
cascade_rcnn = _module
fast_rcnn = _module
faster_rcnn = _module
mask_rcnn = _module
retinanet = _module
rpn = _module
single_stage = _module
test_mixins = _module
two_stage = _module
mask_heads = _module
fcn_mask_head = _module
necks = _module
fpn = _module
fpn_sge = _module
registry = _module
roi_extractors = _module
single_level = _module
conv_module = _module
norm = _module
weight_init = _module
ops = _module
dcn = _module
functions = _module
deform_conv = _module
deform_pool = _module
modules = _module
deform_conv = _module
deform_pool = _module
setup = _module
nms = _module
nms_wrapper = _module
roi_align = _module
roi_align = _module
gradcheck = _module
roi_align = _module
setup = _module
roi_pool = _module
roi_pool = _module
gradcheck = _module
roi_pool = _module
setup = _module
version = _module
tools = _module
coco_eval = _module
pascal_voc = _module
test = _module
train = _module
voc_eval = _module

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
xrange = range
wraps = functools.wraps


import torch.nn as nn


import torch


import numpy as np


import time


import random


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import torch.nn.functional as F


import torch.distributed as dist


import torch.utils.data.distributed


import warnings


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import math


import torch.utils.model_zoo as model_zoo


from torch.nn.parameter import Parameter


from torch.nn import init


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import functional as F


import torch.nn.init as init


import torchvision


import logging


import torch.multiprocessing as mp


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import Dataset


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch._utils import _take_tensors


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from functools import partial


from torch.utils.data import DataLoader


from torch.distributed import get_world_size


from torch.distributed import get_rank


from torch.utils.data.sampler import Sampler


import copy


from collections import Sequence


from torch import nn


import torch.utils.checkpoint as cp


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


from torch.nn.modules.module import Module


def flush_print(func):

    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print


print = flush_print(print)


class SoftCrossEntropyLoss(nn.NLLLoss):

    def __init__(self, label_smoothing=0, num_classes=1000, **kwargs):
        assert label_smoothing >= 0 and label_smoothing <= 1
        super(SoftCrossEntropyLoss, self).__init__(**kwargs)
        self.confidence = 1 - label_smoothing
        self.other = label_smoothing * 1.0 / (num_classes - 1)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        None

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.other)
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        input = F.log_softmax(input, 1)
        return self.criterion(input, one_hot)


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
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
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
        self.out_channels = out_planes
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


class BAM(nn.Module):

    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        ca = self.channel_att(in_tensor)
        sa = self.spatial_att(in_tensor)
        att = 1 + F.sigmoid(ca * sa)
        return att * in_tensor


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'SyncBN': ('bn', None), 'GN': ('gn', nn.GroupNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            frozen (bool): [optional] whether stop gradient updates
                of norm layer, it is helpful to set frozen mode
                in backbone's norms.
        num_features (int): number of channels from input
        postfix (int, str): appended into norm abbreation to
            create named layer.

    Returns:
        name (str): abbreation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    frozen = cfg_.pop('frozen', False)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    if frozen:
        for param in layer.parameters():
            param.requires_grad = False
    return name, layer


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, normalize=dict(type='BN'), dcn=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        assert not bias
        super(DeformConv, self).__init__()
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv(input, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
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
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError('module must be a child of nn.Module, but got {}'.format(module_class))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbone')


def kaiming_init(module, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1, groups=1, base_width=4, style='pytorch', with_cp=False, normalize=dict(type='BN'), dcn=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(normalize, planes * block.expansion)[1])
    layers = []
    layers.append(block(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, groups=groups, base_width=base_width, style=style, with_cp=with_cp, normalize=normalize, dcn=dcn))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, stride=1, dilation=dilation, groups=groups, base_width=base_width, style=style, with_cp=with_cp, normalize=normalize, dcn=dcn))
    return nn.Sequential(*layers)


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


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())
        None

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        None

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-05
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]
        oup_inc = oup // 2
        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True), nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False), nn.BatchNorm2d(oup_inc), nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
        else:
            self.banch1 = nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
            self.banch2 = nn.Sequential(nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True), nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False), nn.BatchNorm2d(oup_inc), nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False), nn.BatchNorm2d(oup_inc), nn.ReLU(inplace=True))
        self.se = SELayer(oup)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :x.shape[1] // 2, :, :]
            x2 = x[:, x.shape[1] // 2:, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))
        out = self.se(out)
        return channel_shuffle(out, 2)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class ShuffleNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(ShuffleNetV2, self).__init__()
        assert input_size % 32 == 0
        self.stage_repeats = [4, 8, 4]
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError("""{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x


class GBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, g=2):
        super(GBatchNorm2d, self).__init__(num_features, affine=False)
        self.weight = Parameter(torch.ones(num_features // g, 1))
        self.bias = Parameter(torch.zeros(num_features // g, 1))
        self.d = num_features // g
        assert num_features % g == 0, '%d / %d = %d error' % (num_features, g, self.d)
        self.g = g

    def expand(self, p):
        p = p.expand(self.d, self.g).reshape(-1)
        return p

    def forward(self, input):
        exponential_average_factor = 0.0
        weight = self.expand(self.weight)
        bias = self.expand(self.bias)
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        return F.batch_norm(input, self.running_mean, self.running_var, weight, bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)


class GL(nn.Module):

    def __init__(self, c, g=4):
        super(GL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, c // g))
        self.bias = Parameter(torch.ones(1, c // g))
        self.sig = nn.Sigmoid()
        self.c = c
        self.g = g
        self.d = c // g
        self.bn = nn.BatchNorm2d(1, affine=False)

    def forward(self, x):
        gx = self.avg_pool(x)
        bx = self.bn(gx.view(-1, 1, self.c, 1)).view(-1, self.c, 1, 1)
        weight = self.weight.expand(self.g, self.d).reshape(1, self.c, 1, 1)
        bias = self.bias.expand(self.g, self.d).reshape(1, self.c, 1, 1)
        bx = bx * weight + bias
        x = x * self.sig(bx)
        return x


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-05
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class A1Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(A1Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        sz = self.weight.size()
        d = 1.0
        for v in sz:
            d *= v
        None
        self.d = math.sqrt(d)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) * self.d + 1e-05
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, (None)] * self.scales[(None), :]).view(-1)
            hs = (h * h_ratios[:, (None)] * self.scales[(None), :]).view(-1)
        else:
            ws = (w * self.scales[:, (None)] * w_ratios[(None), :]).view(-1)
            hs = (h * self.scales[:, (None)] * h_ratios[(None), :]).view(-1)
        base_anchors = torch.stack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)], dim=-1).round()
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        all_anchors = base_anchors[(None), :, :] + shifts[:, (None), :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, (None)].expand(valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid


HEADS = Registry('head')


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[(self.pos_assigned_gt_inds), :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])


class BaseSampler(metaclass=ABCMeta):

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass

    def sample(self, assign_result, bboxes, gt_bboxes, gt_labels=None, **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & (flat_anchors[:, (0)] >= -allowed_border) & (flat_anchors[:, (1)] >= -allowed_border) & (flat_anchors[:, (2)] < img_w + allowed_border) & (flat_anchors[:, (3)] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def build_assigner(cfg, **kwargs):
    if isinstance(cfg, assigners.BaseAssigner):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, assigners, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


def build_sampler(cfg, **kwargs):
    if isinstance(cfg, samplers.BaseSampler):
        return cfg
    elif isinstance(cfg, dict):
        return mmcv.runner.obj_from_dict(cfg, samplers, default_args=kwargs)
    else:
        raise TypeError('Invalid type {} for building a sampler'.format(type(cfg)))


def assign_and_sample(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    bbox_assigner = build_assigner(cfg.assigner)
    bbox_sampler = build_sampler(cfg.sampler)
    assign_result = bbox_assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)
    sampling_result = bbox_sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
    return assign_result, sampling_result


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0
    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[(inds), :] = data
    return ret


def anchor_target_single(flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, target_means, target_stds, cfg, label_channels=1, sampling=True, unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], cfg.allowed_border)
    if not inside_flags.any():
        return (None,) * 6
    anchors = flat_anchors[(inside_flags), :]
    if sampling:
        assign_result, sampling_result = assign_and_sample(anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, target_means, target_stds)
        bbox_targets[(pos_inds), :] = pos_bbox_targets
        bbox_weights[(pos_inds), :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        if label_channels > 1:
            labels, label_weights = expand_binary_labels(labels, label_weights, label_channels)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def anchor_target(anchor_list, valid_flag_list, gt_bboxes_list, img_metas, target_means, target_stds, cfg, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, sampling=True, unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(anchor_target_single, anchor_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, target_means=target_means, target_stds=target_stds, cfg=cfg, label_channels=label_channels, sampling=sampling, unmap_outputs=unmap_outputs)
    if any([(labels is None) for labels in all_labels]):
        return None
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg


def delta2bbox(rois, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1], max_shape=None, wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, (0)] + rois[:, (2)]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, (1)] + rois[:, (3)]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, (2)] - rois[:, (0)] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, (3)] - rois[:, (1)] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)
    gy = torch.addcmul(py, 1, ph, dy)
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, (i)] > score_thr
        if not cls_inds.any():
            continue
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[(cls_inds), :]
        else:
            _bboxes = multi_bboxes[(cls_inds), i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, (None)]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0],), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, (-1)].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
    return bboxes, labels


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.0)
    return F.binary_cross_entropy_with_logits(pred, label.float(), weight.float(), reduction='sum')[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.0)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def sigmoid_focal_loss(pred, target, weight, gamma=2.0, alpha=0.25, reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred, target, weight, gamma=2.0, alpha=0.25, avg_factor=None, num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-06
    return sigmoid_focal_loss(pred, target, weight, gamma=gamma, alpha=alpha, reduction='sum')[None] / avg_factor


def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-06
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        use_sigmoid_cls (bool): Whether to use sigmoid loss for classification.
            (softmax by default)
        use_focal_loss (bool): Whether to use focal loss for classification.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1.0, 2.0], anchor_strides=[4, 8, 16, 32, 64], anchor_base_sizes=None, target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0), use_sigmoid_cls=False, use_focal_loss=False):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls
        self.use_focal_loss = use_focal_loss
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples, cfg):
        if self.use_sigmoid_cls:
            labels = labels.reshape(-1, self.cls_out_channels)
            label_weights = label_weights.reshape(-1, self.cls_out_channels)
        else:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            if self.use_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
            else:
                cls_criterion = weighted_binary_cross_entropy
        elif self.use_focal_loss:
            raise NotImplementedError
        else:
            cls_criterion = weighted_cross_entropy
        if self.use_focal_loss:
            loss_cls = cls_criterion(cls_score, labels, label_weights, gamma=cfg.gamma, alpha=cfg.alpha, avg_factor=num_total_samples)
        else:
            loss_cls = cls_criterion(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(bbox_pred, bbox_targets, bbox_weights, beta=cfg.smoothl1_beta, avg_factor=num_total_samples)
        return loss_cls, loss_reg

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        sampling = False if self.use_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(anchor_list, valid_flag_list, gt_bboxes, img_metas, self.target_means, self.target_stds, cfg, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels, sampling=sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos if self.use_focal_loss else num_total_pos + num_total_neg
        losses_cls, losses_reg = multi_apply(self.loss_single, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples, cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_anchors = [self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:], self.anchor_strides[i]) for i in range(num_levels)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[(topk_inds), :]
                bbox_pred = bbox_pred[(topk_inds), :]
                scores = scores[(topk_inds), :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class RetinaHead(AnchorHead):

    def __init__(self, num_classes, in_channels, stacked_convs=4, octave_base_scale=4, scales_per_octave=3, **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        octave_scales = np.array([(2 ** (i / scales_per_octave)) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHead, self).__init__(num_classes, in_channels, anchor_scales=anchor_scales, use_sigmoid_cls=True, use_focal_loss=True, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.reg_convs.append(nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m, std=0.01)
        for m in self.reg_convs:
            normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = self.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = self.relu(reg_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations."""
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        if dets.is_cuda:
            device_id = dets.get_device()
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError('dets must be either a Tensor or numpy array, but got {}'.format(type(dets)))
    if dets_np.shape[0] == 0:
        inds = []
    else:
        inds = gpu_nms(dets_np, iou_thr, device_id=device_id) if device_id is not None else cpu_nms(dets_np, iou_thr)
    if is_tensor:
        inds = dets.new_tensor(inds, dtype=torch.long)
    else:
        inds = np.array(inds, dtype=np.int64)
    return dets[(inds), :], inds


class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, cfg, gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(cls_scores, bbox_preds, gt_bboxes, None, img_metas, cfg, gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'])

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, (1)]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[(topk_inds), :]
                anchors = anchors[(topk_inds), :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means, self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, (2)] - proposals[:, (0)] + 1
                h = proposals[:, (3)] - proposals[:, (1)] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[(valid_inds), :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, (4)]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[(topk_inds), :]
        return proposals


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class SSDHead(AnchorHead):

    def __init__(self, input_size=300, num_classes=81, in_channels=(512, 1024, 512, 256, 256, 256), anchor_strides=(8, 16, 32, 64, 100, 300), basesize_ratio_range=(0.1, 0.9), anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]), target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [(len(ratios) * 2 + 2) for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(nn.Conv2d(in_channels[i], num_anchors[i] * 4, kernel_size=3, padding=1))
            cls_convs.append(nn.Conv2d(in_channels[i], num_anchors[i] * num_classes, kernel_size=3, padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = (stride - 1) / 2.0, (stride - 1) / 2.0
            scales = [1.0, np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.0]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]
            anchor_generator = AnchorGenerator(base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.use_focal_loss = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_single(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        loss_reg = weighted_smoothl1(bbox_pred, bbox_targets, bbox_weights, beta=cfg.smoothl1_beta, avg_factor=num_total_samples)
        return loss_cls[None], loss_reg

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(anchor_list, valid_flag_list, gt_bboxes, img_metas, self.target_means, self.target_stds, cfg, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=1, sampling=False, unmap_outputs=False)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_images = len(img_metas)
        all_cls_scores = torch.cat([s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        losses_cls, losses_reg = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, num_total_samples=num_total_pos, cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, 0)
        m[-1].inited = True
    else:
        nn.init.constant_(m.weight, 0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([(f in ['channel_add', 'channel_mul']) for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in', nonlinearity='relu')
            self.conv_mask.inited = True
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(3)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return self.weight[(None), :, (None), (None)].expand_as(x) * x / norm


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = topk,
        return_single = True
    else:
        return_single = False
    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros((bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros((bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[(i), start:end] = bbox_targets[(i), :]
        bbox_weights_expand[(i), start:end] = bbox_weights[(i), :]
    return bbox_targets_expand, bbox_weights_expand


def bbox_target_single(pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg, reg_classes=1, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    if reg_classes > 1:
        bbox_targets, bbox_weights = expand_target(bbox_targets, bbox_weights, labels, reg_classes)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target(pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg, reg_classes=1, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0], concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(bbox_target_single, pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=cfg, reg_classes=reg_classes, target_means=target_means, target_stds=target_stds)
    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self, with_avg_pool=False, with_cls=True, with_reg=True, roi_feat_size=7, in_channels=256, num_classes=81, target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2], reg_class_agnostic=False):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= self.roi_feat_size * self.roi_feat_size
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels, rcnn_train_cfg, reg_classes, target_means=self.target_means, target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, reduce=True):
        losses = dict()
        if cls_score is not None:
            losses['loss_cls'] = weighted_cross_entropy(cls_score, labels, label_weights, reduce=reduce)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            losses['loss_reg'] = weighted_smoothl1(bbox_pred, bbox_targets, bbox_weights, avg_factor=bbox_targets.size(0))
        return losses

    def get_det_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]
        if rescale:
            bboxes /= scale_factor
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, (0)].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, (0)] == i).squeeze()
            num_rois = inds.numel()
            bboxes_ = rois[(inds), 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds])
        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4
        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means, self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means, self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, ([0])], bboxes), dim=1)
        return new_rois


class ConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, normalize=None, activation='relu', inplace=True, activate_last=True):
        super(ConvModule, self).__init__()
        self.with_norm = normalize is not None
        self.with_activatation = activation is not None
        self.with_bias = bias
        self.activation = activation
        self.activate_last = activate_last
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm_name, norm = build_norm_layer(normalize, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_activatation:
            assert activation in ['relu'], 'Only ReLU supported.'
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        else:
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        return x


class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \\-> reg convs -> reg fcs -> reg
    """

    def __init__(self, num_shared_convs=0, num_shared_fcs=0, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0, num_reg_fcs=0, conv_out_channels=256, fc_out_channels=1024, normalize=None, *args, **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs > 0
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalize = normalize
        self.with_bias = normalize is None
        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_size * self.roi_feat_size
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_size * self.roi_feat_size
        self.relu = nn.ReLU(inplace=True)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self, num_branch_convs, num_branch_fcs, in_channels, is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
                branch_convs.append(ConvModule(conv_in_channels, self.conv_out_channels, 3, padding=1, normalize=self.normalize, bias=self.with_bias))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_size * self.roi_feat_size
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


class SharedFCBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead, self).__init__(*args, num_shared_convs=0, num_shared_fcs=num_fcs, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0, num_reg_fcs=0, fc_out_channels=fc_out_channels, **kwargs)


dataset_aliases = {'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'], 'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'], 'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'], 'coco': ['coco', 'mscoco', 'ms_coco']}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name
    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError('Unrecognized dataset: {}'.format(dataset))
    else:
        raise TypeError('dataset must a str, but got {}'.format(type(dataset)))
    return labels


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


class BaseDetector(nn.Module):
    """Base class for detectors"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(imgs), len(img_metas)))
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1
        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self, data, result, img_norm_cfg, dataset='coco', score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)
        if isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)) or dataset is None:
            class_names = dataset
        else:
            raise TypeError('dataset must be a valid dataset name or a sequence of class names, not {}'.format(type(dataset)))
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]
            bboxes = np.vstack(bbox_result)
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, (-1)] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names, score_thr=score_thr)


DETECTORS = Registry('detector')


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[(...), 0::4] = w - bboxes[(...), 2::4] - 1
    flipped[(...), 2::4] = w - bboxes[(...), 0::4] - 1
    return flipped


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes


def merge_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.
        img_metas (list[dict]): image info including "shape_scale" and "flip".
        rpn_test_cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape, scale_factor, flip)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, (4)]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[(order), :]
    return merged_proposals


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        merged_proposals = [merge_aug_proposals(proposals, img_meta, rpn_test_cfg) for proposals, img_meta in zip(aug_proposals, img_metas)]
        return merged_proposals


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[(labels == i), :] for i in range(num_classes - 1)]


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = [(mask if not img_info[0]['flip'] else mask[(...), ::-1]) for mask, img_info in zip(aug_masks, img_metas)]
    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks


class CascadeRCNN(BaseDetector, RPNTestMixin):

    def __init__(self, num_stages, backbone, neck=None, rpn_head=None, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(CascadeRCNN, self).__init__()
        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [bbox_roi_extractor for _ in range(num_stages)]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))
        if mask_head is not None:
            self.mask_roi_extractor = nn.ModuleList()
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [mask_roi_extractor for _ in range(num_stages)]
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_roi_extractor) == len(mask_head) == self.num_stages
            for roi_extractor, head in zip(mask_roi_extractor, mask_head):
                self.mask_roi_extractor.append(builder.build_roi_extractor(roi_extractor))
                self.mask_head.append(builder.build_head(head))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        for i in range(self.num_stages):
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]
            assign_results, sampling_results = multi_apply(assign_and_sample, proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg=rcnn_train_cfg)
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = value * lw if 'loss' in name else value
            if self.with_mask:
                mask_roi_extractor = self.mask_roi_extractor[i]
                mask_head = self.mask_head[i]
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
                mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], pos_rois)
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks, rcnn_train_cfg)
                pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = value * lw if 'loss' in name else value
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn
        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]
            bbox_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)
            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels, bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result
                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        segm_result = [[] for _ in range(mask_head.num_classes - 1)]
                    else:
                        _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result
            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred, img_meta[0])
        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result
        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
            else:
                _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks, [img_meta] * self.num_stages, self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks, _bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result
        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = ms_bbox_result['ensemble'], ms_segm_result['ensemble']
            else:
                results = ms_bbox_result['ensemble']
        elif self.with_mask:
            results = {stage: (ms_bbox_result[stage], ms_segm_result[stage]) for stage in ms_bbox_result}
        else:
            results = ms_bbox_result
        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError

    def show_result(self, data, result, img_norm_cfg, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = ms_bbox_result['ensemble'], ms_segm_result['ensemble']
        elif isinstance(result, dict):
            result = result['ensemble']
        super(CascadeRCNN, self).show_result(data, result, img_norm_cfg, **kwargs)


class SingleStageDetector(BaseDetector):

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in bbox_list]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores


class BBoxTestMixin(object):

    def simple_test_bboxes(self, x, img_meta, proposals, rcnn_test_cfg, rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip)
            rois = bbox2roi([proposals])
            roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores, rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self, x, img_meta, det_bboxes, det_labels, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape, scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
                mask_pred = self.mask_head(mask_feats)
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg.rcnn)
            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(merged_masks, det_bboxes, det_labels, self.test_cfg.rcnn, ori_shape, scale_factor=1.0, rescale=False)
        return segm_result


class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin, MaskTestMixin):

    def __init__(self, backbone, neck=None, rpn_head=None, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i], feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets, pos_labels)
            losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        det_bboxes, det_labels = self.simple_test_bboxes(x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        proposal_list = self.aug_test_rpn(self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(self.extract_feats(imgs), img_metas, proposal_list, self.test_cfg.rcnn)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels, self.bbox_head.num_classes)
        if self.with_mask:
            segm_results = self.aug_test_mask(self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, reduction='mean')[None]


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[(i), :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w], (mask_size, mask_size))
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float()
    else:
        mask_targets = pos_proposals.new_zeros((0, mask_size, mask_size))
    return mask_targets


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


class FCNMaskHead(nn.Module):

    def __init__(self, num_convs=4, roi_feat_size=14, in_channels=256, conv_kernel_size=3, conv_out_channels=256, upsample_method='deconv', upsample_ratio=2, num_classes=81, class_agnostic=False, normalize=None):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError('Invalid upsample method {}, accepted methods are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.normalize = normalize
        self.with_bias = normalize is None
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(ConvModule(in_channels, self.conv_out_channels, self.conv_kernel_size, padding=padding, normalize=normalize, bias=self.with_bias))
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(self.conv_out_channels, self.conv_out_channels, self.upsample_ratio, stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(scale_factor=self.upsample_ratio, mode=self.upsample_method)
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = nn.Conv2d(self.conv_out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, torch.zeros_like(labels))
        else:
            loss_mask = mask_cross_entropy(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0
        for i in range(bboxes.shape[0]):
            bbox = (bboxes[(i), :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)
            if not self.class_agnostic:
                mask_pred_ = mask_pred[(i), (label), :, :]
            else:
                mask_pred_ = mask_pred[(i), (0), :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(np.array(im_mask[:, :, (np.newaxis)], order='F'))[0]
            cls_segms[label - 1].append(rle)
        return cls_segms


NECKS = Registry('neck')


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, normalize=None, activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = self.in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class FPNSGE(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, normalize=None, activation=None):
        super(FPNSGE, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.sges = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
            sge = SpatialGroupEnhance(64)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.sges.append(sge)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = self.in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, normalize=normalize, bias=self.with_bias, activation=self.activation, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        outs = [self.sges[i](self.fpn_convs[i](laterals[i])) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                assert False, 'we are not here'
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


ROI_EXTRACTORS = Registry('roi_extractor')


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, (3)] - rois[:, (1)] + 1) * (rois[:, (4)] - rois[:, (2)] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-06))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = torch.FloatTensor(rois.size()[0], self.out_channels, out_size, out_size).fill_(0)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[(inds), :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
        return roi_feats


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups, bias)
        self.conv_offset_mask = nn.Conv2d(self.in_channels // self.groups, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(data, rois, offset, output, output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        deform_pool_cuda.deform_psroi_pooling_cuda_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_channels=1024):
        super(DeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            self.offset_fc = nn.Sequential(nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            self.offset_fc = nn.Sequential(nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            self.mask_fc = nn.Sequential(nn.Linear(self.out_size * self.out_size * self.out_channels, self.deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self.out_size * self.out_size * 1), nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(features, rois, out_h, out_w, spatial_scale, sample_num, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height, data_width)
            roi_align_cuda.backward(grad_output.contiguous(), rois, out_h, out_w, spatial_scale, sample_num, grad_input)
        return grad_input, grad_rois, None, None, None


class RoIAlign(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RoIAlign, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignFunction.apply(features, rois, self.out_size, self.spatial_scale, self.sample_num)


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')
        ctx.save_for_backward(rois)
        num_channels = features.size(1)
        num_rois = rois.size(0)
        out_size = num_rois, num_channels, out_h, out_w
        output = features.new_zeros(out_size)
        argmax = features.new_zeros(out_size, dtype=torch.int)
        roi_pool_cuda.forward(features, rois, out_h, out_w, spatial_scale, output, argmax)
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = features.size()
        ctx.argmax = argmax
        return output

    @staticmethod
    def backward(ctx, grad_output):
        spatial_scale = ctx.spatial_scale
        feature_size = ctx.feature_size
        argmax = ctx.argmax
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            roi_pool_cuda.backward(grad_output.contiguous(), rois, argmax, spatial_scale, grad_input)
        return grad_input, grad_rois, None, None


roi_pool = RoIPoolFunction.apply


class RoIPool(Module):

    def __init__(self, out_size, spatial_scale):
        super(RoIPool, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return roi_pool(features, rois, self.out_size, self.spatial_scale)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (A1Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AnchorHead,
     lambda: ([], {'num_classes': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 256, 64, 64])], {}),
     False),
    (BBoxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([12544, 12544])], {}),
     False),
    (BaseDetector,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBAM,
     lambda: ([], {'gate_channels': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channels': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvModule,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCNMaskHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GL,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RPNHead,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 64, 64])], {}),
     False),
    (RetinaHead,
     lambda: ([], {'num_classes': 4, 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 64, 64])], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (SharedFCBBoxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([12544, 12544])], {}),
     False),
    (ShuffleNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGroupEnhance,
     lambda: ([], {'groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_implus_PytorchInsight(_paritybench_base):
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

    def test_022(self):
        self._check(*TESTCASES[22])

