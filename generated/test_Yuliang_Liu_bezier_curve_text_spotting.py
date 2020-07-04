import sys
_module = sys.modules[__name__]
del sys
predictor = _module
vis_bezier = _module
maskrcnn_benchmark = _module
config = _module
defaults = _module
paths_catalog = _module
data = _module
build = _module
collate_batch = _module
datasets = _module
bezier = _module
coco = _module
concat_dataset = _module
evaluation = _module
word = _module
word_eval = _module
list_dataset = _module
rec = _module
word_dataset = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
engine = _module
bbox_aug = _module
inference = _module
searcher = _module
trainer = _module
layers = _module
_utils = _module
balanced_l1_loss = _module
batch_norm = _module
bezier_align = _module
context_block = _module
dcn = _module
deform_conv_func = _module
deform_conv_module = _module
deform_pool_func = _module
deform_pool_module = _module
iou_loss = _module
misc = _module
nms = _module
non_local = _module
roi_align = _module
roi_pool = _module
scale = _module
seg_loss = _module
sigmoid_focal_loss = _module
smooth_l1_loss = _module
modeling = _module
backbone = _module
fbnet = _module
fbnet_builder = _module
fbnet_modeldef = _module
fpn = _module
hnasnet = _module
mobilenet = _module
msr = _module
necks = _module
pan = _module
resnet = _module
resnet_bn = _module
resnet_layers = _module
balanced_positive_negative_sampler = _module
box_coder = _module
detector = _module
detectors = _module
generalized_rcnn = _module
one_stage = _module
make_layers = _module
matcher = _module
one_stage_head = _module
align = _module
poolers = _module
registry = _module
roi_heads = _module
box_head = _module
box_head = _module
inference = _module
loss = _module
roi_box_feature_extractors = _module
roi_box_predictors = _module
mask_head = _module
inference = _module
loss = _module
mask_head = _module
roi_mask_feature_extractors = _module
roi_mask_predictors = _module
roi_heads = _module
rpn = _module
anchor_generator = _module
fcos = _module
fcos = _module
inference = _module
loss = _module
predictors = _module
inference = _module
loss = _module
retinanet = _module
loss = _module
retinanet = _module
rpn = _module
utils = _module
solver = _module
lr_scheduler = _module
structures = _module
bounding_box = _module
boxlist_ops = _module
image_list = _module
segmentation_mask = _module
c2_model_loading = _module
checkpoint = _module
collect_env = _module
comm = _module
cv2_util = _module
env = _module
imports = _module
logger = _module
measure = _module
metric_logger = _module
miscellaneous = _module
model_serialization = _module
model_zoo = _module
timer = _module
setup = _module
checkpoint = _module
test_backbones = _module
test_box_coder = _module
test_configs = _module
test_data_samplers = _module
test_detectors = _module
test_fbnet = _module
test_feature_extractors = _module
test_metric_logger = _module
test_nms = _module
test_predictors = _module
test_rpn_heads = _module
test_segmentation_mask = _module
test_net = _module
single_demo_bezier = _module
train_net = _module

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


import numpy as np


import torch


import torchvision


from torch.nn import functional as F


import math


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


import logging


import time


from torch import nn


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import torch.nn as nn


from torch.nn.modules.utils import _ntuple


import copy


from collections import OrderedDict


import torch.nn.functional as F


from torch.nn import BatchNorm2d


from collections import namedtuple


from torch.autograd import Variable


import random


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5,
    reduction='none'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(
        b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - 
        alpha * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()
    return loss


def weighted_balanced_l1_loss(pred, target, weight, beta=1.0, alpha=0.5,
    gamma=1.5, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() + 1e-06
    loss = balanced_l1_loss(pred, target, beta, alpha, gamma, reduction='none')
    return torch.sum(loss.sum(dim=1) * weight)[None] / avg_factor


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss
    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss_bbox = self.loss_weight * weighted_balanced_l1_loss(pred,
            target, weight, *args, alpha=self.alpha, gamma=self.gamma, beta
            =self.beta, **kwargs)
        return loss_bbox


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x):
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class _BezierAlign(Function):

    @staticmethod
    def forward(ctx, input, bezier, output_size, spatial_scale, sampling_ratio
        ):
        ctx.save_for_backward(bezier)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.bezier_align_forward(input, bezier, spatial_scale,
            output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        beziers, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.bezier_align_backward(grad_output, beziers,
            spatial_scale, output_size[0], output_size[1], bs, ch, h, w,
            sampling_ratio)
        return grad_input, None, None, None, None


bezier_align = _BezierAlign.apply


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0,
    distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode,
            nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity
            =nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=(
        'channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True
        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1,
        groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(
                'Expected 4D tensor as input, got {}D tensor instead.'.
                format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input,
            weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0
                ] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            _C.deform_conv_forward(input, weight, offset, output, ctx.bufs_
                [0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.
                stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0],
                ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.
                deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0
                ] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                _C.deform_conv_backward_input(input, offset, grad_output,
                    grad_input, grad_offset, weight, ctx.bufs_[0], weight.
                    size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
                    ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.
                    dilation[0], ctx.groups, ctx.deformable_groups,
                    cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                _C.deform_conv_backward_parameters(input, offset,
                    grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1],
                    weight.size(3), weight.size(2), ctx.stride[1], ctx.
                    stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation
                    [1], ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                    1, cur_im2col_step)
        return (grad_input, grad_offset, grad_weight, None, None, None,
            None, None)

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
            raise ValueError(
                'convolution input is too small (output would be {})'.
                format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        self.with_bias = bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(
            in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(
            out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            self.groups, *self.kernel_size))
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, input, offset):
        y = deform_conv(input, offset, self.weight, self.stride, self.
            padding, self.dilation, self.groups, self.deformable_groups)
        if self.with_bias:
            assert len(y.size()) == 4
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'in_channels={}, '.format(self.in_channels),
            'out_channels={}, '.format(self.out_channels),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'groups={}, '.format(self.
            groups), 'deformable_groups={}, '.format(self.deformable_groups
            ), 'bias={})'.format(self.with_bias)])


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1):
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
        if (weight.requires_grad or mask.requires_grad or offset.
            requires_grad or input.requires_grad):
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(
            ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0],
            offset, mask, output, ctx._bufs[1], weight.shape[2], weight.
            shape[3], ctx.stride, ctx.stride, ctx.padding[1], ctx.padding[0
            ], ctx.dilation, ctx.dilation, ctx.groups, ctx.
            deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0],
            offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias,
            grad_offset, grad_mask, grad_output, weight.shape[2], weight.
            shape[3], ctx.stride, ctx.stride, ctx.padding[1], ctx.padding[0
            ], ctx.dilation, ctx.dilation, ctx.groups, ctx.
            deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
            None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding[0] - (ctx.dilation * (
            kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding[1] - (ctx.dilation * (kernel_w -
            1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = _pair(padding)
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            groups, *self.kernel_size))
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
        return modulated_deform_conv(input, offset, mask, self.weight, self
            .bias, self.stride, self.padding, self.dilation, self.groups,
            self.deformable_groups)

    def __repr__(self):
        return ''.join(['{}('.format(self.__class__.__name__),
            'in_channels={}, '.format(self.in_channels),
            'out_channels={}, '.format(self.out_channels),
            'kernel_size={}, '.format(self.kernel_size), 'stride={}, '.
            format(self.stride), 'dilation={}, '.format(self.dilation),
            'padding={}, '.format(self.padding), 'groups={}, '.format(self.
            groups), 'deformable_groups={}, '.format(self.deformable_groups
            ), 'bias={})'.format(self.with_bias)])


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size,
        out_channels, no_trans, group_size=1, part_size=None,
        sample_per_part=4, trans_std=0.0):
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
        _C.deform_psroi_pooling_forward(data, rois, offset, output,
            output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels,
            ctx.group_size, ctx.out_size, ctx.part_size, ctx.
            sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        _C.deform_psroi_pooling_backward(grad_output, data, rois, offset,
            output_count, grad_input, grad_offset, ctx.no_trans, ctx.
            spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size,
            ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return (grad_input, grad_rois, grad_offset, None, None, None, None,
            None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
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
        return deform_roi_pooling(data, rois, offset, self.spatial_scale,
            self.out_size, self.out_channels, self.no_trans, self.
            group_size, self.part_size, self.sample_per_part, self.trans_std)


class IOULoss(nn.Module):

    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, (0)]
        pred_top = pred[:, (1)]
        pred_right = pred[:, (2)]
        pred_bottom = pred[:, (3)]
        target_left = target[:, (0)]
        target_top = target[:, (1)]
        target_right = target[:, (2)]
        target_bottom = target[:, (3)]
        target_aera = (target_left + target_right) * (target_top +
            target_bottom)
        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right,
            target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
            pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(torch.nn.Conv2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation,
            self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        output_shape = [((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op) for
            i, p, di, k, d, op in zip(x.shape[-2:], self.padding, self.
            dilation, self.kernel_size, self.stride, self.output_padding)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm2d(torch.nn.BatchNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class DFConv2d(nn.Module):
    """Deformable convolutional layer"""

    def __init__(self, in_channels, out_channels, with_modulated_dcn=True,
        kernel_size=3, stride=1, groups=1, dilation=1, deformable_groups=1,
        bias=False, padding=None):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (
                kernel_size[1] - 1) // 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2
            conv_block = DeformConv
        self.offset = Conv2d(in_channels, deformable_groups *
            offset_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=1, dilation=dilation)
        for l in [self.offset]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.0)
        self.conv = conv_block(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, deformable_groups=deformable_groups, bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x, return_offset=False):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset_mask = self.offset(x)
                x = self.conv(x, offset_mask)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if return_offset:
                return x, offset_mask
            return x
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation,
            self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, 'GroupNorm: can only specify G or C/G.'
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, 'dim: {}, dim_per_gp: {}'.format(dim,
            dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, 'dim: {}, num_groups: {}'.format(dim,
            num_groups)
        group_gn = num_groups
    return group_gn


_global_config['MODEL'] = 4


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp,
        num_groups), out_channels, eps, affine)


def conv_with_kaiming_uniform(use_gn=False, use_relu=False, use_deformable=
    False, use_bn=False):

    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1
        ):
        if use_deformable:
            conv_func = DFConv2d
        else:
            conv_func = Conv2d
        conv = conv_func(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation, bias=not (use_gn or use_bn))
        if not use_deformable:
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if not (use_gn or use_bn):
                nn.init.constant_(conv.bias, 0)
        module = [conv]
        if use_gn:
            module.append(group_norm(out_channels))
        elif use_bn:
            module.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv
    return make_conv


class NonLocal2D(nn.Module):
    """Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self, in_channels, reduction=2, use_scale=True, use_gn=
        False, use_deformable=False, mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
        ConvModule = conv_with_kaiming_uniform()
        last_conv = conv_with_kaiming_uniform(use_gn=use_gn, use_deformable
            =use_deformable)
        self.g = ConvModule(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.theta = ConvModule(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.phi = ConvModule(self.in_channels, self.inter_channels,
            kernel_size=1)
        self.conv_out = last_conv(self.inter_channels, self.in_channels,
            kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        output = x + self.conv_out(y)
        return output


class _ROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(input, roi, spatial_scale,
            output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(grad_output, rois, spatial_scale,
            output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class _ROIPool(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output, argmax = _C.roi_pool_forward(input, roi, spatial_scale,
            output_size[0], output_size[1])
        ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_pool_backward(grad_output, input, rois, argmax,
            spatial_scale, output_size[0], output_size[1], bs, ch, h, w)
        return grad_input, None, None, None


roi_pool = _ROIPool.apply


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class SegLoss(nn.Module):

    def __init__(self, other=-1, scale_factor=1):
        super(SegLoss, self).__init__()
        self.other = other
        self.scale_factor = scale_factor

    def prepare_target(self, targets, mask):
        labels = []
        for t in targets:
            t = t.get_field('seg_masks').get_mask_tensor().unsqueeze(0)
            if self.other > 0:
                t = torch.clamp(t, max=self.other)
            if self.scale_factor != 1:
                t = F.interpolate(t.unsqueeze(0), scale_factor=self.
                    scale_factor, mode='nearest').long().squeeze()
            labels.append(t)
        batched_labels = mask.new_full((mask.size(0), mask.size(2), mask.
            size(3)), mask.size(1) - 1, dtype=torch.long)
        for label, pad_label in zip(labels, batched_labels):
            pad_label[:label.shape[0], :label.shape[1]].copy_(label)
        return batched_labels

    def forward(self, mask, target):
        """
            mask : Tensor
            target : list[Boxlist]
        """
        target = self.prepare_target(target, mask)
        loss = F.cross_entropy(mask, target)
        return loss


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device
        ).unsqueeze(0)
    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range
        ) * (t >= 0)).float() * term2 * (1 - alpha)


class _SigmoidFocalLoss(Function):

    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        losses = _C.sigmoid_focalloss_forward(logits, targets, num_classes,
            gamma, alpha)
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(logits, targets, d_loss,
            num_classes, gamma, alpha)
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu
        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'gamma=' + str(self.gamma)
        tmpstr += ', alpha=' + str(self.alpha)
        tmpstr += ')'
        return tmpstr


def _get_trunk_cfg(arch_def):
    """ Get all stages except the last one """
    num_stages = mbuilder.get_num_stages(arch_def)
    trunk_stages = arch_def.get('backbone', range(num_stages - 1))
    ret = mbuilder.get_blocks(arch_def, stage_indices=trunk_stages)
    return ret


class FBNetTrunk(nn.Module):

    def __init__(self, builder, arch_def, dim_in):
        super(FBNetTrunk, self).__init__()
        self.first = builder.add_first(arch_def['first'], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg['stages'])

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        ret = [y]
        return ret


logger = logging.getLogger(__name__)


def _get_rpn_stage(arch_def, num_blocks):
    rpn_stage = arch_def.get('rpn')
    ret = mbuilder.get_blocks(arch_def, stage_indices=rpn_stage)
    if num_blocks > 0:
        logger.warn('Use last {} blocks in {} as rpn'.format(num_blocks, ret))
        block_count = len(ret['stages'])
        assert num_blocks <= block_count, 'use block {}, block count {}'.format(
            num_blocks, block_count)
        blocks = range(block_count - num_blocks, block_count)
        ret = mbuilder.get_blocks(ret, block_indices=blocks)
    return ret['stages']


class FBNetRPNHead(nn.Module):

    def __init__(self, cfg, in_channels, builder, arch_def):
        super(FBNetRPNHead, self).__init__()
        assert in_channels == builder.last_depth
        rpn_bn_type = cfg.MODEL.FBNET.RPN_BN_TYPE
        if len(rpn_bn_type) > 0:
            builder.bn_type = rpn_bn_type
        use_blocks = cfg.MODEL.FBNET.RPN_HEAD_BLOCKS
        stages = _get_rpn_stage(arch_def, use_blocks)
        self.head = builder.add_blocks(stages)
        self.out_channels = builder.last_depth

    def forward(self, x):
        x = [self.head(y) for y in x]
        return x


ARCH_CFG_NAME_MAPPING = {'bbox': 'ROI_BOX_HEAD', 'kpts':
    'ROI_KEYPOINT_HEAD', 'mask': 'ROI_MASK_HEAD'}


def _get_head_stage(arch, head_name, blocks):
    if head_name not in arch:
        head_name = 'head'
    head_stage = arch.get(head_name)
    ret = mbuilder.get_blocks(arch, stage_indices=head_stage, block_indices
        =blocks)
    return ret['stages']


class FBNetROIHead(nn.Module):

    def __init__(self, cfg, in_channels, builder, arch_def, head_name,
        use_blocks, stride_init, last_layer_scale):
        super(FBNetROIHead, self).__init__()
        assert in_channels == builder.last_depth
        assert isinstance(use_blocks, list)
        head_cfg_name = ARCH_CFG_NAME_MAPPING[head_name]
        self.pooler = poolers.make_pooler(cfg, head_cfg_name)
        stage = _get_head_stage(arch_def, head_name, use_blocks)
        assert stride_init in [0, 1, 2]
        if stride_init != 0:
            stage[0]['block'][3] = stride_init
        blocks = builder.add_blocks(stage)
        last_info = copy.deepcopy(arch_def['last'])
        last_info[1] = last_layer_scale
        last = builder.add_last(last_info)
        self.head = nn.Sequential(OrderedDict([('blocks', blocks), ('last',
            last)]))
        self.out_channels = builder.last_depth

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class Identity(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.conv = ConvBNRelu(C_in, C_out, kernel=1, stride=stride, pad=0,
            no_bias=1, use_relu='relu', bn_type='bn'
            ) if C_in != C_out or stride != 1 else None

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):

    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [Conv2d(C_in, C_in, 3, stride, 1, bias=False), BatchNorm2d(
            C_in), nn.ReLU(inplace=True), Conv2d(C_in, C_out, 3, 1, 1, bias
            =False), BatchNorm2d(C_out)]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = stride == 1 and C_in == C_out

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):

    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.
            float32)
        ch_idx = 0
        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1
        hks = kernel_size // 2
        ksq = kernel_size ** 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx:ch_idx + num_ch, (0), (i), (j)] = 1
                ch_idx += num_ch
        self.register_parameter('bias', None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(x, self.kernel, self.bias, (self.
                stride, self.stride), (self.padding, self.padding), self.
                dilation, self.C)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i,
            p, di, k, d in zip(x.shape[-2:], (self.padding, self.dilation),
            (self.dilation, self.dilation), (self.kernel_size, self.
            kernel_size), (self.stride, self.stride))]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


class ShiftBlock5x5(nn.Sequential):

    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = stride == 1 and C_in == C_out
        C_mid = _get_divisible_by(C_in * expansion, 8, 8)
        ops = [Conv2d(C_in, C_mid, 1, 1, 0, bias=False), BatchNorm2d(C_mid),
            nn.ReLU(inplace=True), Shift(C_mid, 5, stride, 2), Conv2d(C_mid,
            C_out, 1, 1, 0, bias=False), BatchNorm2d(C_out)]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, 'Incompatible group size {} for input channel {}'.format(
            g, C)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4
            ).contiguous().view(N, C, H, W)


class ConvBNRelu(nn.Sequential):

    def __init__(self, input_depth, output_depth, kernel, stride, pad,
        no_bias, use_relu, bn_type, group=1, *args, **kwargs):
        super(ConvBNRelu, self).__init__()
        assert use_relu in ['relu', None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == 'gn'
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ['bn', 'af', 'gn', None]
        assert stride in [1, 2, 4]
        op = Conv2d(input_depth, output_depth, *args, kernel_size=kernel,
            stride=stride, padding=pad, bias=not no_bias, groups=group, **
            kwargs)
        nn.init.kaiming_normal_(op.weight, mode='fan_out', nonlinearity='relu')
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module('conv', op)
        if bn_type == 'bn':
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == 'gn':
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth
                )
        elif bn_type == 'af':
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module('bn', bn_op)
        if use_relu == 'relu':
            self.add_module('relu', nn.ReLU(inplace=True))


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)
        self.op = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(
            inplace=True), conv2, nn.Sigmoid())

    def forward(self, x):
        return x * self.op(x)


def interpolate(input, size=None, scale_factor=None, mode='nearest',
    align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor,
            mode, align_corners)

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError(
                'only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple
            ) and len(scale_factor) != dim:
            raise ValueError(
                'scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'
                .format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in
            range(dim)]
    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners)


def _get_upsample_op(stride):
    assert stride in [1, 2, 4] or stride in [-1, -2, -4] or isinstance(stride,
        tuple) and all(x in [-1, -2, -4] for x in stride)
    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [(-x) for x in stride] if isinstance(stride, tuple
            ) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode='nearest', align_corners=None)
    return ret, stride


class ShuffleV2Block(nn.Module):

    def __init__(self, input_depth, output_depth, expansion, stride,
        bn_type='bn', kernel=3, width_divisor=1, shuffle_type=None,
        pw_group=1, se=False, cdw=False, dw_skip_bn=False, dw_skip_relu=False):
        super(ShuffleV2Block, self).__init__()
        assert kernel in [1, 3, 5, 7], kernel
        assert input_depth == output_depth
        self.input_depth = input_depth // 2
        self.output_depth = self.input_depth
        mid_depth = int(output_depth * expansion) // 2
        self.pw = ConvBNRelu(self.input_depth, mid_depth, kernel=1, stride=
            1, pad=0, no_bias=1, use_relu='relu', bn_type=bn_type, group=
            pw_group)
        self.upscale, stride = _get_upsample_op(stride)
        if kernel == 1:
            self.dw = nn.Sequential()
        else:
            self.dw = ConvBNRelu(mid_depth, mid_depth, kernel=kernel,
                stride=stride, pad=kernel // 2, group=mid_depth, no_bias=1,
                use_relu='relu' if not dw_skip_relu else None, bn_type=
                bn_type if not dw_skip_bn else None)
        self.pwl = ConvBNRelu(mid_depth, self.output_depth, kernel=1,
            stride=1, pad=0, no_bias=1, use_relu=None, bn_type=bn_type,
            group=pw_group)
        self.shuffle = ChannelShuffle(pw_group * 2)

    def forward(self, x):
        x1 = x[:, :self.input_depth, :, :]
        x2 = x[:, self.input_depth:, :, :]
        y = self.pw(x1)
        y = self.dw(y)
        y = self.pwl(y)
        out = torch.cat((y, x2), dim=1)
        out = self.shuffle(out)
        return out


class IRFBlock(nn.Module):

    def __init__(self, input_depth, output_depth, expansion, stride,
        bn_type='bn', kernel=3, width_divisor=1, shuffle_type=None,
        pw_group=1, se=False, cdw=False, dw_skip_bn=False, dw_skip_relu=
        False, use_res_connect=True):
        super(IRFBlock, self).__init__()
        assert kernel in [1, 3, 5, 7], kernel
        self.use_res_connect = (stride == 1 and input_depth == output_depth and
            use_res_connect)
        self.output_depth = output_depth
        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)
        if input_depth == mid_depth:
            self.pw = nn.Sequential()
        else:
            self.pw = ConvBNRelu(input_depth, mid_depth, kernel=1, stride=1,
                pad=0, no_bias=1, use_relu='relu', bn_type=bn_type, group=
                pw_group)
        self.upscale, stride = _get_upsample_op(stride)
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=
                stride, pad=kernel // 2, group=mid_depth, no_bias=1,
                use_relu='relu', bn_type=bn_type)
            dw2 = ConvBNRelu(mid_depth, mid_depth, kernel=kernel, stride=1,
                pad=kernel // 2, group=mid_depth, no_bias=1, use_relu=
                'relu' if not dw_skip_relu else None, bn_type=bn_type if 
                not dw_skip_bn else None)
            self.dw = nn.Sequential(OrderedDict([('dw1', dw1), ('dw2', dw2)]))
        else:
            self.dw = ConvBNRelu(mid_depth, mid_depth, kernel=kernel,
                stride=stride, pad=kernel // 2, group=mid_depth, no_bias=1,
                use_relu='relu' if not dw_skip_relu else None, bn_type=
                bn_type if not dw_skip_bn else None)
        self.pwl = ConvBNRelu(mid_depth, output_depth, kernel=1, stride=1,
            pad=0, no_bias=1, use_relu=None, bn_type=bn_type, group=pw_group)
        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)
        self.se4 = SEModule(output_depth) if se else nn.Sequential()
        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == 'mid':
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, conv_block,
        top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = 'fpn_inner{}'.format(idx)
            layer_block = 'fpn_layer{}'.format(idx)
            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(x[:-1][::-1], self.
            inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.upsample(last_inner, size=inner_lateral.
                shape[-2:], mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        else:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return tuple(results)


class Scaler(nn.Module):

    def __init__(self, in_channels_list, out_channels, conv_block,
        top_blocks=None):
        super(Scaler, self).__init__()
        self.layers = nn.ModuleList()
        for in_c in in_channels_list:
            if in_c == 0:
                continue
            self.layers.append(conv_block(in_c, out_channels, 3, 1))
        self.top_blocks = top_blocks

    def forward(self, x):
        results = []
        start_ind = len(x) - len(self.layers)
        for layer, feature in zip(self.layers, x[start_ind:]):
            results.append(layer(feature))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return tuple(results)


class LastLevelMaxPool(nn.Module):

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels, last_stride=2):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, last_stride, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)

    def forward(self, p5):
        p6 = self.p6(p5)
        return [p6]


class Scaler(nn.Module):
    """Reshape features"""

    def __init__(self, scale, inp, C, relu=True):
        """
        Arguments:
            scale (int) [-2, 2]: scale < 0 for downsample
            inp (int): input channel
            C (int): output channel
            relu (bool): set to False if the modules are pre-relu
        """
        super(Scaler, self).__init__()
        if scale == 0:
            self.scaler = conv1x1_bn(inp, C, 1, relu=relu)
        if scale == 1:
            self.scaler = nn.Sequential(nn.Upsample(scale_factor=2, mode=
                'bilinear', align_corners=False), conv1x1_bn(inp, C, 1,
                relu=relu))
        if scale == -1:
            self.scaler = conv3x3_bn(inp, C, 2, relu=relu)

    def forward(self, hidden_state):
        return self.scaler(hidden_state)


class DeepLabScaler(nn.Module):
    """Official implementation
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py#L90
    """

    def __init__(self, scale, inp, C):
        super(DeepLabScaler, self).__init__()
        self.scale = 2 ** scale
        self.conv = conv1x1_bn(inp, C, 1, relu=False)

    def forward(self, hidden_state):
        if self.scale != 1:
            hidden_state = F.interpolate(hidden_state, scale_factor=self.
                scale, mode='bilinear', align_corners=False)
        return self.conv(F.relu(hidden_state))


class HNASNet(nn.Module):

    def __init__(self, cfg):
        super(HNASNet, self).__init__()
        geno_file = cfg.MODEL.HNASNET.GENOTYPE
        None
        geno_cell, geno_path = torch.load(geno_file)
        self.geno_path = geno_path
        self.f = cfg.MODEL.HNASNET.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.HNASNET.NUM_LAYERS
        self.num_blocks = cfg.MODEL.HNASNET.NUM_BLOCKS
        BxF = self.f * self.num_blocks
        stride_mults = cfg.MODEL.HNASNET.STRIDE_MULTIPLIER
        self.num_strides = len(stride_mults)
        self.stem1 = nn.Sequential(conv3x3_bn(3, 64, 2), conv3x3_bn(64, 64, 1))
        self.stem2 = conv3x3_bn(64, BxF, 2)
        self.bases = nn.ModuleList()
        in_channels = 64
        for s in range(self.num_strides):
            out_channels = BxF * stride_mults[s]
            self.bases.append(conv3x3_bn(in_channels, out_channels, 2))
            in_channels = out_channels
        self.cells = nn.ModuleList()
        self.scalers = nn.ModuleList()
        if cfg.MODEL.HNASNET.TIE_CELL:
            geno_cell = [geno_cell] * self.num_layers
        h_0 = 0
        for layer, (geno, h) in enumerate(zip(geno_cell, geno_path), 1):
            stride = stride_mults[h]
            self.cells.append(FixCell(geno, self.f * stride))
            inp0 = BxF * stride_mults[h_0]
            scaler0 = Scaler(h_0 - h, inp0, stride * self.f, relu=False)
            scaler1 = Scaler(0, BxF * stride, stride * self.f, relu=False)
            h_0 = h
            self.scalers.append(scaler0)
            self.scalers.append(scaler1)

    def forward(self, x, drop_prob=-1):
        h1 = self.stem1(x)
        h0 = self.stem2(h1)
        fps = []
        for base in self.bases:
            h1 = base(h1)
            fps.append(h1)
        s_1 = 0
        for i, (cell, s) in enumerate(zip(self.cells, self.geno_path)):
            input_0 = self.scalers[i * 2](h0)
            input_1 = self.scalers[i * 2 + 1](fps[s])
            fps[s_1] = h0
            if s == s_1:
                h0 = cell(input_0, input_1, drop_prob) + h0
            else:
                h0 = cell(input_0, input_1, drop_prob) + fps[s]
            s_1 = s
        fps[s_1] = h0
        return fps


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(
                hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup,
                1, 1, 0, bias=False), BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(Conv2d(inp, hidden_dim, 1, 1, 0, bias
                =False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=
                hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(
                inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):
    """
    Should freeze bn
    """

    def __init__(self, cfg, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.return_features_indices = [3, 6, 13, 17]
        self.return_features_num_channels = []
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2)])
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                if len(self.features) - 1 in self.return_features_indices:
                    self.return_features_num_channels.append(output_channel)
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                res.append(x)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ConcatUpConv(nn.Module):

    def __init__(self, inplanes, outplanes, upsample=True):
        super(ConcatUpConv, self).__init__()
        out_channels = outplanes
        self.upsample = upsample
        self.con_1x1 = nn.Conv2d(inplanes, outplanes, 1, bias=False)
        nn.init.kaiming_uniform_(self.con_1x1.weight, a=1)
        self.nor_1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu_1 = nn.ReLU()
        if self.upsample:
            self.con_3x3 = nn.Conv2d(outplanes, out_channels // 2,
                kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_uniform_(self.con_3x3.weight, a=1)
            self.nor_3 = nn.BatchNorm2d(out_channels // 2)
            self.leakyrelu_3 = nn.ReLU()

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out_1 = self.leakyrelu_1(self.nor_1(self.con_1x1(fusion)))
        out = None
        if self.upsample:
            out = self.leakyrelu_3(self.nor_3(self.con_3x3(out_1)))
            out = F.interpolate(out, scale_factor=2, mode='bilinear',
                align_corners=False)
        return out, out_1


class MSR(nn.Module):

    def __init__(self, body, channels, fpn=None, pan=None):
        super(MSR, self).__init__()
        self.body = body
        cucs = nn.ModuleList()
        channel = channels[0]
        cucs.append(ConcatUpConv(channel * 2, channel, upsample=False))
        for i, channel in enumerate(channels[1:]):
            cucs.append(ConcatUpConv(channel * 2, channel))
        self.cucs = cucs
        if fpn is not None:
            self.fpn = fpn
        if pan is not None:
            self.pan = pan

    def forward(self, x):
        outputs = self.body(x)
        re_x = F.interpolate(x, scale_factor=0.5, mode='bilinear',
            align_corners=False)
        output_re = self.body(re_x)[-1]
        low = F.interpolate(output_re, size=outputs[-1].shape[2:], mode=
            'bilinear', align_corners=False)
        new_outputs = []
        for cuc, high in zip(self.cucs[::-1], outputs[::-1]):
            low, out = cuc(high, low)
            new_outputs.append(out)
        outs = new_outputs[::-1]
        if hasattr(self, 'pan'):
            outs = self.pan(outs)
        if hasattr(self, 'fpn'):
            outs = self.fpn(outs)
        return outs


class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)
    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    """

    def __init__(self, in_channels, num_levels, refine_level=1, refine_type
        =None, use_gn=False, use_deformable=False):
        """
        Arguments:
            in_channels (int): Number of input channels (feature maps of all levels
                should have the same channels).
            num_levels (int): Number of input feature levels.
            refine_level (int): Index of integration and refine level of BSF in
                multi-level features from bottom to top.
            refine_type (str): Type of the refine op, currently support
                [None, 'conv', 'non_local'].
        """
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local', 'gc_block']
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        if self.refine_type == 'conv':
            conv_block = conv_with_kaiming_uniform(use_gn=use_gn,
                use_deformable=use_deformable)
            self.refine = conv_block(self.in_channels, self.in_channels, 3,
                padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        elif self.refine_type == 'gc_block':
            self.refine = ContextBlock(self.in_channels, ratio=1.0 / 16.0)

    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=
                    gather_size)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode=
                    'nearest')
            feats.append(gathered)
        bsf = sum(feats) / len(feats)
        if self.refine_type is not None:
            bsf = self.refine(bsf)
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
        return tuple(outs)


class FPA(nn.Module):

    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels / 4)
        self.channels_cond = channels
        self.conv_master = nn.Conv2d(self.channels_cond, channels,
            kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size
            =1, bias=False)
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid,
            kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=
            (5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=
            (3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)
        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=
            (7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=
            (5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=
            (3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)
        self.conv1x1_up1 = nn.Conv2d(channels_mid, channels, kernel_size=(1,
            1), stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.
            channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_upsample = F.interpolate(x3_2, size=x2_2.shape[-2:], mode=
            'bilinear', align_corners=False)
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = F.interpolate(x2_merge, size=x1_2.shape[-2:], mode=
            'bilinear', align_corners=False)
        x1_merge = self.relu(x1_2 + x2_upsample)
        x1_merge_upsample = F.interpolate(x1_merge, size=x_master.shape[-2:
            ], mode='bilinear', align_corners=False)
        x1_merge_upsample_ch = self.relu(self.bn_upsample_1(self.
            conv1x1_up1(x1_merge_upsample)))
        x_master = x_master * x1_merge_upsample_ch
        out = self.relu(x_master + x_gpb)
        return out


class GAU(nn.Module):

    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3,
            padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)
        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1,
            padding=0, bias=False)
        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high,
                channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low,
                kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(
            fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)
        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(self.bn_upsample(self.conv_upsample(fms_high)) +
                fms_att)
        else:
            out = self.relu(self.bn_reduction(self.conv_reduction(fms_high)
                ) + fms_att)
        return out


class PAN(nn.Module):

    def __init__(self):
        """
        :param blocks: Blocks of the network with reverse sequential.
        """
        super(PAN, self).__init__()
        channels_blocks = [2048, 1024, 512, 256]
        self.fpa = FPA(channels=channels_blocks[0])
        self.gau_block1 = GAU(channels_blocks[0], channels_blocks[1])
        self.gau_block2 = GAU(channels_blocks[1], channels_blocks[2])
        self.gau_block3 = GAU(channels_blocks[2], channels_blocks[3])
        self.gau = [self.gau_block1, self.gau_block2, self.gau_block3]

    def forward(self, fms):
        """
        :param fms: Feature maps of forward propagation in the network with reverse sequential. shape:[b, c, h, w]
        :return: fm_high. [b, 256, h, w]
        """
        feats = []
        for i, fm_low in enumerate(fms[::-1]):
            if i == 0:
                fm_high = self.fpa(fm_low)
            else:
                fm_high = self.gau[int(i - 1)](fm_high, fm_low)
            feats.append(fm_high)
        feats.reverse()
        return tuple(feats)


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        if module is not None:
            _register_generic(self, module_name, module)
            return

        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn
        return register_fn


StageSpec = namedtuple('StageSpec', ['index', 'block_count', 'return_features']
    )


ResNet101FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c,
    return_features=r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 23, 
    True), (4, 3, True)))


ResNet101StagesTo4 = tuple(StageSpec(index=i, block_count=c,
    return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 23,
    True)))


ResNet101StagesTo5 = tuple(StageSpec(index=i, block_count=c,
    return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 23,
    False), (4, 3, True)))


ResNet14FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c,
    return_features=r) for i, c, r in ((1, 1, True), (2, 1, True), (3, 1, 
    True), (4, 1, True)))


ResNet152FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c,
    return_features=r) for i, c, r in ((1, 3, True), (2, 8, True), (3, 36, 
    True), (4, 3, True)))


ResNet50FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c,
    return_features=r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 6, 
    True), (4, 3, True)))


ResNet50StagesTo4 = tuple(StageSpec(index=i, block_count=c, return_features
    =r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 6, True)))


ResNet50StagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features
    =r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3,
    True)))


_STAGE_SPECS = Registry({'R-14': ResNet14FPNStagesTo5, 'R-50':
    ResNet50FPNStagesTo5, 'R-50-C4': ResNet50StagesTo4, 'R-50-C5':
    ResNet50StagesTo5, 'R-101-C4': ResNet101StagesTo4, 'R-101-C5':
    ResNet101StagesTo5, 'R-50-FPN': ResNet50FPNStagesTo5,
    'R-50-FPN-RETINANET': ResNet50FPNStagesTo5, 'R-101-FPN':
    ResNet101FPNStagesTo5, 'R-101-PAN': ResNet101FPNStagesTo5,
    'R-101-FPN-RETINANET': ResNet101FPNStagesTo5, 'R-152-FPN':
    ResNet152FPNStagesTo5, 'R-152-PAN': ResNet152FPNStagesTo5})


def _make_stage(transformation_module, in_channels, bottleneck_channels,
    out_channels, block_count, num_groups, stride_in_1x1, first_stride,
    dilation=1, dcn_config={}):
    blocks = []
    stride = first_stride
    max_dcn_layer = dcn_config.get('max_dcn_layer', 0)
    for i in range(block_count):
        if i < block_count - max_dcn_layer:
            block_dcn_config = {}
        else:
            block_dcn_config = dcn_config
        blocks.append(transformation_module(in_channels,
            bottleneck_channels, out_channels, num_groups, stride_in_1x1,
            stride, dilation=dilation, dcn_config=block_dcn_config))
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class ResNet(nn.Module):

    def __init__(self, cfg):
        super(ResNet, self).__init__()
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.
            TRANS_FUNC]
        self.stem = stem_module(cfg)
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = 'layer' + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = (stage2_bottleneck_channels *
                stage2_relative_factor)
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.
                index - 1]
            stage_with_context = cfg.MODEL.RESNETS.STAGE_WITH_CONTEXT[
                stage_spec.index - 1]
            module = _make_stage(transformation_module, in_channels,
                bottleneck_channels, out_channels, stage_spec.block_count,
                num_groups, cfg.MODEL.RESNETS.STRIDE_IN_1X1, first_stride=
                int(stage_spec.index > 1) + 1, dcn_config={'stage_with_dcn':
                stage_with_dcn, 'stage_with_context': stage_with_context,
                'max_dcn_layer': cfg.MODEL.RESNETS.MAX_DCN_LAYER,
                'with_modulated_dcn': cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                'deformable_groups': cfg.MODEL.RESNETS.DEFORMABLE_GROUPS})
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem
            else:
                m = getattr(self, 'layer' + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):

    def __init__(self, block_module, stages, num_groups=1, width_per_group=
        64, stride_in_1x1=True, stride_init=None, res2_out_channels=256,
        dilation=1, dcn_config={}):
        super(ResNetHead, self).__init__()
        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = (stage2_bottleneck_channels *
            stage2_relative_factor)
        block_module = _TRANSFORMATION_MODULES[block_module]
        self.stages = []
        stride = stride_init
        for stage in stages:
            name = 'layer' + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(block_module, in_channels,
                bottleneck_channels, out_channels, stage.block_count,
                num_groups, stride_in_1x1, first_stride=stride, dilation=
                dilation, dcn_config=dcn_config)
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, bottleneck_channels, out_channels,
        num_groups, stride_in_1x1, stride, dilation, norm_func, dcn_config):
        super(Bottleneck, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(Conv2d(in_channels,
                out_channels, kernel_size=1, stride=down_stride, bias=False
                ), norm_func(out_channels))
            for modules in [self.downsample]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
        if dilation > 1:
            stride = 1
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1,
            stride=stride_1x1, bias=False)
        self.bn1 = norm_func(bottleneck_channels)
        with_dcn = dcn_config.get('stage_with_dcn', False)
        with_context = dcn_config.get('stage_with_context', False)
        if with_dcn:
            deformable_groups = dcn_config.get('deformable_groups', 1)
            with_modulated_dcn = dcn_config.get('with_modulated_dcn', False)
            self.conv2 = DFConv2d(bottleneck_channels, bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn, kernel_size=3,
                stride=stride_3x3, groups=num_groups, dilation=dilation,
                deformable_groups=deformable_groups, bias=False)
        else:
            self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels,
                kernel_size=3, stride=stride_3x3, padding=dilation, bias=
                False, groups=num_groups, dilation=dilation)
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)
        self.bn2 = norm_func(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=
            1, bias=False)
        self.bn3 = norm_func(out_channels)
        if with_context:
            self.context = ContextBlock(out_channels, 1.0 / 16.0)
        else:
            self.context = None
        for l in [self.conv1, self.conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.context is not None:
            out = self.context(out)
        out += identity
        out = F.relu_(out)
        return out


class BaseStem(nn.Module):

    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_func(out_channels)
        for l in [self.conv1]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


class ResNet(nn.Module):
    """Residual network definition.
    More information about the model: https://arxiv.org/abs/1512.03385
    Args:
        block (nn.Module): type of building block (Basic or Bottleneck).
        layers (list of ints): number of blocks in each layer.
        return_idx (list or int): indices of the layers to be returned
                                  during the forward pass.
    Attributes:
      in_planes (int): number of channels in the stem block.
    """

    def __init__(self, block, layers, return_idx=[0, 1, 2, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self._out_c = []
        self.return_idx = make_list(return_idx)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self._out_c = [out_c for idx, out_c in enumerate(self._out_c) if 
            idx in return_idx]

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create residual layer.
        Args:
            block (nn.Module): type of building block (Basic or Bottleneck).
            planes (int): number of input channels.
            blocks (int): number of blocks.
            stride (int): stride inside the first block.
        Returns:
            `nn.Sequential' instance of all created layers.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        self._out_c.append(self.inplanes)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        outs.append(self.layer1(x))
        outs.append(self.layer2(outs[-1]))
        outs.append(self.layer3(outs[-1]))
        outs.append(self.layer4(outs[-1]))
        return [outs[idx] for idx in self.return_idx]


def batchnorm(in_planes, affine=True, eps=1e-05, momentum=0.1):
    """2D Batch Normalisation.
    Args:
      in_planes (int): number of input channels.
      affine (bool): whether to add learnable affine parameters.
      eps (float): stability constant in the denominator.
      momentum (float): running average decay coefficient.
    Returns:
      `nn.BatchNorm2d' instance.
    """
    return nn.BatchNorm2d(in_planes, affine=affine, eps=eps, momentum=momentum)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1, bias=False):
    """2D 3x3 convolution.
    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the operation.
      dilation (int): dilation rate of the operation.
      groups (int): number of groups in the operation.
      bias (bool): whether to add learnable bias parameter.
    Returns:
      `nn.Conv2d' instance.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, dilation=dilation, groups=groups, bias=bias)


class BasicBlock(nn.Module):
    """Basic residual block.
    Conv-BN-ReLU => Conv-BN => Residual => ReLU.
    Args:
      inplanes (int): number of input channels.
      planes (int): number of intermediate and output channels.
      stride (int): stride of the first convolution.
      downsample (nn.Module or None): downsampling operation.
    Attributes:
      expansion (int): equals to the ratio between the numbers
                       of output and intermediate channels.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batchnorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batchnorm(planes)
        self.downsample = downsample
        self.stride = stride

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


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """2D 1x1 convolution.
    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the operation.
      groups (int): number of groups in the operation.
      bias (bool): whether to add learnable bias parameter.
    Returns:
      `nn.Conv2d' instance.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        padding=0, groups=groups, bias=bias)


class Bottleneck(nn.Module):
    """Bottleneck residual block.
    Conv-BN-ReLU => Conv-BN-ReLU => Conv-BN => Residual => ReLU.
    Args:
      inplanes (int): number of input channels.
      planes (int): number of intermediate and output channels.
      stride (int): stride of the first convolution.
      downsample (nn.Module or None): downsampling operation.
    Attributes:
      expansion (int): equals to the ratio between the numbers
                       of output and intermediate channels.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, bias=False)
        self.bn1 = batchnorm(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = batchnorm(planes)
        self.conv3 = conv1x1(planes, planes * 4, bias=False)
        self.bn3 = batchnorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, 'cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry'.format(
        cfg.MODEL.BACKBONE.CONV_BODY)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)


def build_neck(cfg):
    assert cfg.MODEL.NECK.CONV_BODY in registry.NECKS, 'cfg.MODEL.NECK.CONV_BODY: {} is not registered in registry'.format(
        cfg.MODEL.NECK.CONV_BODY)
    return registry.NECKS[cfg.MODEL.NECK.CONV_BODY](cfg)


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)


def build_roi_heads(cfg, in_channels):
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(('mask', build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(('keypoint', build_roi_keypoint_head(cfg,
            in_channels)))
    if cfg.MODEL.INST_ON:
        roi_heads.append(('inst', build_roi_inst_head(cfg, in_channels)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)


def build_retinanet(cfg, in_channels):
    return RetinaNetModule(cfg, in_channels)


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.FCOS_ON:
        return build_fcos(cfg, in_channels)
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)
    return RPNModule(cfg, in_channels)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]
    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
        if size_divisible > 0:
            import math
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        image_sizes = [im.shape[-2:] for im in tensors]
        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(
            type(tensors)))


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.has_aux_heads = False

    def forward(self, images, targets=None, vis=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            vis (bool): not used

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        if self.training and self.has_aux_heads:
            targets, targets_aux = targets
        images = to_image_list(images)
        features = self.neck(self.backbone(images.tensors))
        proposals, proposal_losses = self.rpn(images, features, targets,
            vis=vis)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals,
                targets)
        else:
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return result


def build_one_stage_head(cfg, in_channels):
    assert cfg.MODEL.ONE_STAGE_HEAD in registry.ONE_STAGE_HEADS, 'cfg.MODEL.ONE_STAGE_HEAD: {} are not registered in registry'.format(
        cfg.MODEL.ONE_STAGE_HEAD)
    return registry.ONE_STAGE_HEADS[cfg.MODEL.ONE_STAGE_HEAD](cfg, in_channels)


class OneStage(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(OneStage, self).__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.decoder = build_one_stage_head(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, vis=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        images = to_image_list(images)
        features = self.neck(self.backbone(images.tensors))
        result, decoder_losses = self.decoder(images, features, targets,
            vis=vis)
        if self.training:
            losses = {}
            losses.update(decoder_losses)
            return losses
        return result


class CTCPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(CTCPredictor, self).__init__()
        self.voc_size = cfg.DATASETS.TEXT.VOC_SIZE
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = nn.LSTM(in_channels, in_channels, num_layers=1,
            bidirectional=True)
        self.clf = nn.Linear(in_channels * 2, self.voc_size)

    def forward(self, x, targets=None):
        x = self.convs(x)
        x = x.mean(dim=2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.clf(x)
        if self.training:
            x = F.log_softmax(x, dim=-1)
            input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch
                .long)
            target_lengths, targets = self.prepare_targets(targets)
            loss = F.ctc_loss(x, targets, input_lengths, target_lengths,
                blank=self.voc_size - 1) / 10
            return loss
        return x

    def prepare_targets(self, targets):
        target_lengths = (targets != self.voc_size - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        sum_targets = torch.cat(sum_targets)
        return target_lengths, sum_targets


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, cfg, in_channels):
        super(CRNN, self).__init__()
        self.voc_size = cfg.DATASETS.TEXT.VOC_SIZE
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels, in_channels, in_channels)

    def forward(self, x):
        x = self.convs(x)
        x = x.mean(dim=2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        return x


class Attention(nn.Module):

    def __init__(self, cfg, in_channels):
        super(Attention, self).__init__()
        self.hidden_size = in_channels
        self.output_size = cfg.DATASETS.TEXT.VOC_SIZE
        self.dropout_p = 0.1
        self.max_len = cfg.DATASETS.TEXT.NUM_CHARS
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.vat = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        """
        hidden: 1 x n x self.hidden_size
        encoder_outputs: time_step x n x self.hidden_size (T,N,C)
        """
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.vat(torch.tanh(alpha))
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_applied = torch.matmul(attn_weights, encoder_outputs.permute((
            1, 0, 2)))
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

    def prepare_targets(self, targets):
        target_lengths = (targets != self.output_size - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        return target_lengths, sum_targets


class ATTPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(ATTPredictor, self).__init__()
        self.CRNN = CRNN(cfg, in_channels)
        self.criterion = torch.nn.NLLLoss()
        self.attention = Attention(cfg, in_channels)
        self.teach_prob = 1.0

    def forward(self, rois, targets=None):
        rois = self.CRNN(rois)
        if self.training:
            text = targets
            target_variable = text
            _init = torch.zeros((rois.size()[1], 1)).long()
            _init = torch.LongTensor(_init)
            target_variable = torch.cat((_init, target_variable), 1)
            target_variable = target_variable
            decoder_input = target_variable[:, (0)]
            decoder_hidden = self.attention.initHidden(rois.size()[1])
            loss = 0.0
            try:
                for di in range(1, target_variable.shape[1]):
                    decoder_output, decoder_hidden, decoder_attention = (self
                        .attention(decoder_input, decoder_hidden, rois))
                    loss += self.criterion(decoder_output, target_variable[
                        :, (di)])
                    teach_forcing = True if random.random(
                        ) > self.teach_prob else False
                    if teach_forcing:
                        decoder_input = target_variable[:, (di)]
                    else:
                        topv, topi = decoder_output.data.topk(1)
                        ni = topi.squeeze()
                        decoder_input = ni
            except Exception as e:
                None
                loss = 0.0
            return loss
        else:
            n = rois.size()[1]
            decodes = torch.zeros((n, self.attention.max_len))
            prob = 1.0
            decoder_input = torch.zeros(n).long()
            decoder_hidden = self.attention.initHidden(n)
            try:
                for di in range(self.attention.max_len):
                    decoder_output, decoder_hidden, decoder_attention = (self
                        .attention(decoder_input, decoder_hidden, rois))
                    probs = torch.exp(decoder_output)
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni
                    prob *= probs[:, (ni)]
                    decodes[:, (di)] = decoder_input.clone()
                decodes = torch.as_tensor(decodes)
            except:
                decodes += 96
            return decodes


class AlignHead(nn.Module):

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AlignHead, self).__init__()
        resolution = cfg.MODEL.ALIGN.POOLER_RESOLUTION
        canonical_scale = cfg.MODEL.ALIGN.POOLER_CANONICAL_SCALE
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.pooler = Pooler(output_size=resolution, scales=self.scales,
            sampling_ratio=1, canonical_scale=canonical_scale, mode='bezier')
        for head in ['rec']:
            tower = []
            conv_block = conv_with_kaiming_uniform(True, True, False, False)
            for i in range(cfg.MODEL.ALIGN.NUM_CONVS):
                tower.append(conv_block(in_channels, in_channels, 3, 1))
            self.add_module('{}_tower'.format(head), nn.Sequential(*tower))
        self.predict_type = cfg.MODEL.ALIGN.PREDICTOR
        if self.predict_type == 'ctc':
            self.predictor = CTCPredictor(cfg, in_channels)
        elif self.predict_type == 'attention':
            self.predictor = ATTPredictor(cfg, in_channels)
        else:
            raise 'Unknown recognition predictor.'

    def forward(self, x, proposals):
        """
        offset related operations are messy
        """
        beziers = [p.get_field('beziers') for p in proposals]
        rois = self.pooler(x, proposals, beziers)
        rois = self.rec_tower(rois)
        if self.training:
            targets = []
            for proposals_per_im in proposals:
                targets.append(proposals_per_im.get_field('rec').rec)
            targets = torch.cat(targets, dim=0)
            loss = self.predictor(rois, targets)
            return None, loss
        else:
            if self.predict_type == 'ctc':
                logits = self.predictor(rois)
                _, preds = logits.permute(1, 0, 2).max(dim=-1)
            elif self.predict_type == 'attention':
                preds = self.predictor(rois)
            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.add_field('recs', preds[start_ind:end_ind])
                start_ind = end_ind
            return proposals, {}


class AlignModule(torch.nn.Module):
    """
    Module for BezierAlign computation. Takes feature maps from the backbone and
    BezierAlign outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(AlignModule, self).__init__()
        self.cfg = cfg.clone()
        self.head = AlignHead(cfg, in_channels)
        self.detector = build_fcos(cfg, in_channels)
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES

    def forward(self, images, features, targets=None, vis=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
            vis (bool): visualise offsets

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        boxes, losses = self.detector(images, features[1:], targets)
        rec_features = features[:len(self.scales)]
        if self.training:
            _, mask_loss = self.head(rec_features, targets)
            losses.update({'rec_loss': mask_loss})
            return None, losses
        preds, _ = self.head(rec_features, boxes)
        return preds, {}


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4,
        eps=1e-06):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self
            .eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min

    def get_random(self, level):
        """ Generate a random roi for target level
        """
        xmin, ymin, xmax, ymax = torch.tensor


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0,
        deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale,
            out_size, out_channels, no_trans, group_size, part_size,
            sample_per_part, trans_std)
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            self.offset_fc = nn.Sequential(nn.Linear(self.out_size * self.
                out_size * self.out_channels, self.deform_fc_channels), nn.
                ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self
                .deform_fc_channels), nn.ReLU(inplace=True), nn.Linear(self
                .deform_fc_channels, self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            self.mask_fc = nn.Sequential(nn.Linear(self.out_size * self.
                out_size * self.out_channels, self.deform_fc_channels), nn.
                ReLU(inplace=True), nn.Linear(self.deform_fc_channels, self
                .out_size * self.out_size * 1), nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.
                spatial_scale, self.out_size, self.out_channels, self.
                no_trans, self.group_size, self.part_size, self.
                sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                self.out_size, self.out_channels, True, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(data, rois, offset, self.
                spatial_scale, self.out_size, self.out_channels, self.
                no_trans, self.group_size, self.part_size, self.
                sample_per_part, self.trans_std) * mask


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio, output_channel=
        256, canonical_scale=160, mode='align'):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            if mode == 'align':
                pooler = ROIAlign(output_size, spatial_scale=scale,
                    sampling_ratio=sampling_ratio)
            elif mode == 'deformable':
                pooler = ModulatedDeformRoIPoolingPack(spatial_scale=scale,
                    out_size=output_size[0], out_channels=output_channel,
                    no_trans=False, group_size=1, trans_std=0.1)
            elif mode == 'bezier':
                pooler = BezierAlign(output_size, spatial_scale=scale,
                    sampling_ratio=1)
            else:
                raise NotImplementedError()
            poolers.append(pooler)
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)
            ).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)
            ).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max, canonical_scale=
            canonical_scale)

    def convert_to_roi_format(self, boxes):
        if isinstance(boxes[0], torch.Tensor):
            concat_boxes = cat([b for b in boxes], dim=0)
        else:
            concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for
            i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes, beziers=None):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        if beziers is not None:
            rois = self.convert_to_roi_format(beziers)
        else:
            rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)
        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = x[0].shape[1]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, *self.output_size),
            dtype=dtype, device=device)
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.
            poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
        return result


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.
        FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:
                num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:
                num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000.0 / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, (2)] - proposals[:, (0)] + TO_REMOVE
        ex_heights = proposals[:, (3)] - proposals[:, (1)] + TO_REMOVE
        ex_ctr_x = proposals[:, (0)] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, (1)] + 0.5 * ex_heights
        gt_widths = reference_boxes[:, (2)] - reference_boxes[:, (0)
            ] + TO_REMOVE
        gt_heights = reference_boxes[:, (3)] - reference_boxes[:, (1)
            ] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, (0)] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, (1)] + 0.5 * gt_heights
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw,
            targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)
        TO_REMOVE = 1
        widths = boxes[:, (2)] - boxes[:, (0)] + TO_REMOVE
        heights = boxes[:, (3)] - boxes[:, (1)] + TO_REMOVE
        ctr_x = boxes[:, (0)] + 0.5 * widths
        ctr_y = boxes[:, (1)] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, (None)] + ctr_x[:, (None)]
        pred_ctr_y = dy * heights[:, (None)] + ctr_y[:, (None)]
        pred_w = torch.exp(dw) * widths[:, (None)]
        pred_h = torch.exp(dh) * heights[:, (None)]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold,
        allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    'No ground-truth boxes available for one of the images during training'
                    )
            else:
                raise ValueError(
                    'No proposal boxes available for one of the images during training'
                    )
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches,
                match_quality_matrix)
        return matches

    def set_low_quality_matches_(self, matches, all_matches,
        match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, (None)])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, (1)]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError('boxlists should have same image size, got {}, {}'
            .format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert('xyxy')
    boxlist2 = boxlist2.convert('xyxy')
    N = len(boxlist1)
    M = len(boxlist2)
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, (None), :2], box2[:, :2])
    rb = torch.min(box1[:, (None), 2:], box2[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, :, (0)] * wh[:, :, (1)]
    iou = inter / (area1[:, (None)] + area2 - inter)
    return iou


def smooth_l1_loss(input, target, beta=1.0 / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
        cls_agnostic_bbox_reg=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields('labels')
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals):
            proposals_per_image.add_field('labels', labels_per_image)
            proposals_per_image.add_field('regression_targets',
                regression_targets_per_image)
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(
            sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img
                ).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device
        if not hasattr(self, '_proposals'):
            raise RuntimeError('subsample needs to be called before')
        proposals = self._proposals
        labels = cat([proposal.get_field('labels') for proposal in
            proposals], dim=0)
        regression_targets = cat([proposal.get_field('regression_targets') for
            proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, (None)] + torch.tensor([0, 1, 2, 3
                ], device=device)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset[:,
            (None)], map_inds], regression_targets[sampled_pos_inds_subset],
            size_average=False, beta=1)
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler,
        box_coder, cls_agnostic_bbox_reg)
    return loss_evaluator


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    postprocessor = PostProcessor(score_thresh, nms_thresh,
        detections_per_img, box_coder, cls_agnostic_bbox_reg, bbox_aug_enabled)
    return postprocessor


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg,
            in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor
            .out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)
        if not self.training:
            result = self.post_processor((class_logits, box_regression),
                proposals)
            return x, result, {}
        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits],
            [box_regression])
        return x, proposals, dict(loss_classifier=loss_classifier,
            loss_box_reg=loss_box_reg)


FLIP_LEFT_RIGHT = 0


FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor
            ) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format
                (bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError(
                'last dimension of bbox should have a size of 4, got {}'.
                format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def pop_field(self, field):
        return self.extra_fields.pop(field)

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax -
                ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + (w - TO_REMOVE).clamp(min=0), ymin + (h -
                TO_REMOVE).clamp(min=0)
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size,
            self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor) and k != 'rles':
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox
        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax,
            scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and k != 'rles':
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def pad(self, new_size):
        bbox = BoxList(self.bbox, new_size, mode=self.mode)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.pad(new_size)
            bbox.add_field(k, v)
        return bbox

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                'Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin,
            transposed_xmax, transposed_ymax), dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box, remove_empty=False):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin ==
                cropped_ymax)
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax,
            cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        if remove_empty:
            box = bbox.bbox
            keep = (box[:, (3)] > box[:, (1)]) & (box[:, (2)] > box[:, (0)])
            bbox = bbox[keep]
        return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, (0)].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, (1)].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, (2)].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, (3)].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, (3)] > box[:, (1)]) & (box[:, (2)] > box[:, (0)])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            area = (box[:, (2)] - box[:, (0)] + TO_REMOVE) * (box[:, (3)] -
                box[:, (1)] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, (2)] * box[:, (3)]
        else:
            raise RuntimeError('Should not be here')
        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self)
                    )
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field='scores'):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)
    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)
    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode
        )
    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100,
        box_coder=None, cls_agnostic_bbox_reg=False, bbox_aug_enabled=False):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(box_regression.view(sum(
            boxes_per_image), -1), concat_boxes)
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])
        num_classes = class_prob.shape[1]
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals,
            image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode='xyxy')
        boxlist.add_field('scores', scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field('scores').reshape(-1, num_classes)
        device = scores.device
        result = []
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, (j)].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[(inds), j * 4:(j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
            boxlist_for_class.add_field('scores', scores_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field('labels', torch.full((num_labels,),
                j, dtype=torch.int64, device=device))
            result.append(boxlist_for_class)
        result = cat_boxlist(result)
        number_of_detections = len(result)
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field('scores')
            image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                number_of_detections - self.detections_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_fc(dim_in, hidden_dim, use_gn=False):
    """
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    """
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()
        num_masks = x.shape[0]
        labels = [bbox.get_field('labels') for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, (None)]
        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)
        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field('mask', prob)
            results.append(bbox)
        return results


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field('labels')
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field('labels')
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[cfg.MODEL.ROI_MASK_HEAD.
        FEATURE_EXTRACTOR]
    return func(cfg, in_channels)


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert('xyxy')
    assert segmentation_masks.size == proposals.size, '{}, {}'.format(
        segmentation_masks, proposals)
    proposals = proposals.bbox.to(torch.device('cpu'))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):

    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['labels', 'masks'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
            segmentation_masks = matched_targets.get_field('masks')
            segmentation_masks = segmentation_masks[positive_inds]
            positive_proposals = proposals_per_image[positive_inds]
            masks_per_image = project_masks_on_boxes(segmentation_masks,
                positive_proposals, self.discretization_size)
            labels.append(labels_per_image)
            masks.append(masks_per_image)
        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)
        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits[
            positive_inds, labels_pos], mask_targets)
        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.
        ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    loss_evaluator = MaskRCNNLossComputation(matcher, cfg.MODEL.
        ROI_MASK_HEAD.RESOLUTION)
    return loss_evaluator


def expand_boxes(boxes, scale):
    w_half = (boxes[:, (2)] - boxes[:, (0)]) * 0.5
    h_half = (boxes[:, (3)] - boxes[:, (1)]) * 0.5
    x_c = (boxes[:, (2)] + boxes[:, (0)]) * 0.5
    y_c = (boxes[:, (3)] + boxes[:, (1)]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, (0)] = x_c - w_half
    boxes_exp[:, (2)] = x_c + w_half
    boxes_exp[:, (1)] = y_c - h_half
    boxes_exp[:, (3)] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    mask = mask.float()
    box = box.float()
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    if thresh >= 0:
        mask = mask > thresh
    else:
        mask = (mask * 255).to(torch.uint8)
    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0
        ]:x_1 - box[0]]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert('xyxy')
        im_w, im_h = boxes.size
        res = [paste_mask_in_image(mask, box, im_h, im_w, self.threshold,
            self.padding) for mask, box in zip(masks, boxes.bbox)]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, (None)]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == len(masks
            ), 'Masks and boxes should have the same length.'
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box
                ), 'Number of objects should be the same.'
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor


def make_roi_mask_predictor(cfg, in_channels):
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class ROIMaskHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg,
            in_channels)
        self.predictor = make_roi_mask_predictor(cfg, self.
            feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if (self.training and self.cfg.MODEL.ROI_MASK_HEAD.
            SHARE_BOX_FEATURE_EXTRACTOR):
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)
        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}
        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
        return x, all_proposals, dict(loss_mask=loss_mask)


def make_conv3x3(in_channels, out_channels, dilation=1, stride=1, use_gn=
    False, use_relu=False, kaiming_init=True):
    conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
        padding=dilation, dilation=dilation, bias=False if use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity=
            'relu')
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if (cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.
            SHARE_BOX_FEATURE_EXTRACTOR):
            self.mask.feature_extractor = self.box.feature_extractor
            self.inst.feature_extractor = self.box.feature_extractor
        if (cfg.MODEL.KE_ON and cfg.MODEL.ROI_KE_HEAD.
            SHARE_BOX_FEATURE_EXTRACTOR):
            self.ke.feature_extractor = self.box.feature_extractor
        if (cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.
            SHARE_BOX_FEATURE_EXTRACTOR):
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, prefix=''):
        """
        prefix (str): Some model may use auxiliary heads which don't share rpn,
        use this to separate the loss names
        """
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if (self.training and self.cfg.MODEL.ROI_MASK_HEAD.
                SHARE_BOX_FEATURE_EXTRACTOR):
                mask_features = x
            x, detections, loss_mask = self.mask(mask_features, detections,
                targets)
            losses.update(loss_mask)
        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            if (self.training and self.cfg.MODEL.ROI_KEYPOINT_HEAD.
                SHARE_BOX_FEATURE_EXTRACTOR):
                keypoint_features = x
            x, detections, loss_keypoint = self.keypoint(keypoint_features,
                detections, targets)
            losses.update(loss_keypoint)
        if self.cfg.MODEL.INST_ON:
            inst_features = features
            if (self.training and self.cfg.MODEL.ROI_MASK_HEAD.
                SHARE_BOX_FEATURE_EXTRACTOR):
                inst_features = x
            x, detections, loss_inst = self.inst(inst_features, detections,
                targets)
            losses.update(loss_inst)
        losses = {(prefix + k): losses[k] for k in losses}
        return x, detections, losses


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, (np.newaxis)]
    hs = hs[:, (np.newaxis)]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), 
        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[(i), :], scales) for i in
        range(anchors.shape[0])])
    return torch.from_numpy(anchors)


def generate_anchors(stride=16, sizes=(32, 64, 128, 256, 512),
    aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(stride, np.array(sizes, dtype=np.float) /
        stride, np.array(aspect_ratios, dtype=np.float))


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32), straddle_thresh=0):
        super(AnchorGenerator, self).__init__()
        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [generate_anchors(anchor_stride, sizes,
                aspect_ratios).float()]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError('FPN should have #anchor_strides == #sizes')
            cell_anchors = [generate_anchors(anchor_stride, size if
                isinstance(size, (tuple, list)) else (size,), aspect_ratios
                ).float() for anchor_stride, size in zip(anchor_strides, sizes)
                ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides,
            self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width * stride, step=stride,
                dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride,
                dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1,
                4)).reshape(-1, 4))
        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (anchors[..., 0] >= -self.straddle_thresh) & (anchors
                [..., 1] >= -self.straddle_thresh) & (anchors[..., 2] < 
                image_width + self.straddle_thresh) & (anchors[..., 3] < 
                image_height + self.straddle_thresh)
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8,
                device=device)
        boxlist.add_field('visibility', inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes
            ):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(anchors_per_feature_map, (image_width,
                    image_height), mode='xyxy')
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def snv2_block(in_channels, out_channels, kernel_size, stride):
    return ShuffleV2Block(in_channels, out_channels, expansion=2, stride=
        stride, kernel=kernel_size)


class FCOSHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        if cfg.MODEL.FCOS.USE_LIGHTWEIGHT:
            conv_block = snv2_block
        else:
            conv_block = conv_with_kaiming_uniform(cfg.MODEL.FCOS.USE_GN,
                cfg.MODEL.FCOS.USE_RELU, cfg.MODEL.FCOS.USE_DEFORMABLE, cfg
                .MODEL.FCOS.USE_BN)
        for head in ['bbox']:
            tower = []
            for i in range(cfg.MODEL.FCOS.NUM_CONVS):
                tower.append(conv_block(in_channels, in_channels, 3, 1))
            self.add_module('{}_tower'.format(head), nn.Sequential(*tower))
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3,
            stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1,
            padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1,
            padding=1)
        self.bezier_pred = nn.Conv2d(in_channels, 16, kernel_size=3, stride
            =1, padding=1)
        for modules in [self.cls_logits, self.bbox_pred, self.centerness,
            self.bezier_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        bezier_reg = []
        centerness = []
        tt = 0.0
        for l, feature in enumerate(x):
            bbox_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(bbox_tower))
            centerness.append(self.centerness(bbox_tower))
            bbox_reg.append(F.relu(self.bbox_pred(bbox_tower)))
            bezier_reg.append(self.bezier_pred(bbox_tower))
        return logits, bbox_reg, centerness, bezier_reg


INF = 100000000


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA)
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.loc_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.object_sizes_of_interest = soi

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys,
        radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            center_gt[beg:end, :, (0)] = torch.where(xmin > gt[beg:end, :,
                (0)], xmin, gt[beg:end, :, (0)])
            center_gt[beg:end, :, (1)] = torch.where(ymin > gt[beg:end, :,
                (1)], ymin, gt[beg:end, :, (1)])
            center_gt[beg:end, :, (2)] = torch.where(xmax > gt[beg:end, :,
                (2)], gt[beg:end, :, (2)], xmax)
            center_gt[beg:end, :, (3)] = torch.where(ymax > gt[beg:end, :,
                (3)], gt[beg:end, :, (3)], ymax)
            beg = end
        left = gt_xs[:, (None)] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, (None)]
        top = gt_ys[:, (None)] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, (None)]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = self.object_sizes_of_interest
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = points_per_level.new_tensor(
                object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(
                points_per_level), -1))
        expanded_object_sizes_of_interest = torch.cat(
            expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in
            points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, bezier_targets = (self.
            compute_targets_for_locations(points_all_level, targets,
            expanded_object_sizes_of_interest))
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i],
                num_points_per_level, dim=0)
            bezier_targets[i] = torch.split(bezier_targets[i],
                num_points_per_level, dim=0)
        labels_level_first = []
        reg_targets_level_first = []
        bezier_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(torch.cat([labels_per_im[level] for
                labels_per_im in labels], dim=0))
            reg_targets_level_first.append(torch.cat([reg_targets_per_im[
                level] for reg_targets_per_im in reg_targets], dim=0) /
                self.strides[level])
            bezier_targets_level_first.append(torch.cat([
                bezier_targets_per_im[level] for bezier_targets_per_im in
                bezier_targets], dim=0) / self.strides[level])
        return (labels_level_first, reg_targets_level_first,
            bezier_targets_level_first)

    def compute_targets_for_locations(self, locations, targets,
        object_sizes_of_interest):
        labels = []
        reg_targets = []
        bezier_targets = []
        xs, ys = locations[:, (0)], locations[:, (1)]
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == 'xyxy'
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field('labels')
            area = targets_per_im.area()
            l = xs[:, (None)] - bboxes[:, (0)][None]
            t = ys[:, (None)] - bboxes[:, (1)][None]
            r = bboxes[:, (2)][None] - xs[:, (None)]
            b = bboxes[:, (3)][None] - ys[:, (None)]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            bezier_pts = targets_per_im.get_field('beziers').bbox.view(-1, 8, 2
                )
            y_targets = bezier_pts[:, :, (0)][None] - ys[:, (None), (None)]
            x_targets = bezier_pts[:, :, (1)][None] - xs[:, (None), (None)]
            bezier_targets_per_im = torch.stack((y_targets, x_targets), dim=3)
            bezier_targets_per_im = bezier_targets_per_im.view(xs.size(0),
                bboxes.size(0), 16)
            if self.center_sample:
                is_in_boxes = self.get_sample_region(bboxes, self.strides,
                    self.num_points_per_level, xs, ys, radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            is_cared_in_the_level = (max_reg_targets_per_im >=
                object_sizes_of_interest[:, ([0])]) & (max_reg_targets_per_im
                 <= object_sizes_of_interest[:, ([1])])
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            locations_to_min_aera, locations_to_gt_inds = (locations_to_gt_area
                .min(dim=1))
            reg_targets_per_im = reg_targets_per_im[range(len(locations)),
                locations_to_gt_inds]
            bezier_targets_per_im = bezier_targets_per_im[range(len(
                locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_aera == INF] = 0
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            bezier_targets.append(bezier_targets_per_im)
        return labels, reg_targets, bezier_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, ([0, 2])]
        top_bottom = reg_targets[:, ([1, 3])]
        centerness = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression,
        bezier_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        num_classes = box_cls[0].size(1)
        labels, reg_targets, bezier_targets = self.prepare_targets(locations,
            targets)
        box_cls_flatten = []
        box_regression_flatten = []
        bezier_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        bezier_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-
                1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3,
                1).reshape(-1, 4))
            bezier_regression_flatten.append(bezier_regression[l].permute(0,
                2, 3, 1).reshape(-1, 16))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            bezier_targets_flatten.append(bezier_targets[l].reshape(-1, 16))
            centerness_flatten.append(centerness[l].reshape(-1))
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        bezier_regression_flatten = torch.cat(bezier_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        bezier_targets_flatten = torch.cat(bezier_targets_flatten, dim=0)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        num_pos_per_gpu = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_per_gpu])
            ).item()
        box_regression_flatten = box_regression_flatten[pos_inds]
        bezier_regression_flatten = bezier_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        bezier_targets_flatten = bezier_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()
            ) / max(total_num_pos / num_gpus, 1.0)
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(
                reg_targets_flatten)
            sum_centerness_targets = centerness_targets.sum()
            sum_centerness_targets = reduce_sum(sum_centerness_targets).item()
            reg_loss = self.box_reg_loss_func(box_regression_flatten,
                reg_targets_flatten, centerness_targets) / (
                sum_centerness_targets / num_gpus)
            centerness_loss = self.centerness_loss_func(centerness_flatten,
                centerness_targets) / max(total_num_pos / num_gpus, 1.0)
        else:
            reg_loss = box_regression_flatten.sum()
            bezier_loss = bezier_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()
        bezier_loss = F.smooth_l1_loss(bezier_regression_flatten,
            bezier_targets_flatten, reduction='none')
        bezier_loss = (bezier_loss.mean(dim=-1) * centerness_targets).sum() / (
            sum_centerness_targets / num_gpus)
        return cls_loss, reg_loss, bezier_loss, centerness_loss

    def compute_offsets_targets(self, mask_targets, reg_targets):
        num_chars = mask_targets.sum(dim=1).long()
        N, K = mask_targets.size()
        offsets_x = torch.zeros(N, K, dtype=torch.float32, device=
            mask_targets.device)
        offsets_y = torch.zeros(N, K, dtype=torch.float32, device=
            mask_targets.device)
        for i, (nc, reg) in enumerate(zip(num_chars, reg_targets)):
            xs = (reg[2] + reg[0]) * (torch.tensor(list(range(nc)), dtype=
                torch.float32, device=mask_targets.device) * 2 + 1) / (nc * 2
                ) - reg[0]
            offsets_x[(i), :nc] = xs
            offsets_y[(i), :nc] = (reg[3] - reg[1]) / 2
        return torch.stack((offsets_y, offsets_x), dim=2).view(N, -1)


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator


def make_fcos_postprocessor(config, is_train=False):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    if is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
        pre_nms_thresh = 0.01
    box_selector = FCOSPostProcessor(pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n, min_size=0, num_classes=
        config.MODEL.FCOS.NUM_CLASSES, fpn_strides=config.MODEL.FCOS.
        FPN_STRIDES)
    return box_selector


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()
        self.cfg = cfg.clone()
        head = FCOSHead(cfg, in_channels)
        box_selector_train = make_fcos_postprocessor(cfg, is_train=True)
        box_selector_test = make_fcos_postprocessor(cfg)
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_iters = 0

    def forward(self, images, features, targets=None, vis=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, bezier_regression = self.head(
            features)
        locations = self.compute_locations(features)
        if self.training:
            return self._forward_train(locations, box_cls, box_regression,
                bezier_regression, centerness, targets, images.image_sizes)
        else:
            box_regression = [(r * s) for r, s in zip(box_regression, self.
                fpn_strides)]
            bezier_regression = [(r * s) for r, s in zip(bezier_regression,
                self.fpn_strides)]
            return self._forward_test(locations, box_cls, box_regression,
                bezier_regression, centerness, images.image_sizes)

    def _forward_train(self, locations, box_cls, box_regression,
        bezier_regression, centerness, targets, image_sizes):
        loss_box_cls, loss_box_reg, loss_bezier_reg, loss_centerness = (self
            .loss_evaluator(locations, box_cls, box_regression,
            bezier_regression, centerness, targets))
        """
        if self.cfg.MODEL.RPN_ONLY:
            boxes = None
        else:
            with torch.no_grad():
                boxes = self.box_selector_train(
                    locations, box_cls, box_regression,
                    centerness, image_sizes)
        """
        boxes = None
        losses = {'loss_cls': loss_box_cls, 'loss_reg': loss_box_reg,
            'loss_bezier': loss_bezier_reg, 'loss_centerness': loss_centerness}
        return boxes, losses

    def _forward_test(self, locations, box_cls, box_regression,
        bezier_regression, centerness, image_sizes):
        boxes = self.box_selector_test(locations, box_cls, box_regression,
            bezier_regression, centerness, image_sizes)
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w,
                self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.
            float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.
            float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    xywh_boxes = boxlist.convert('xywh').bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
        fpn_post_nms_top_n, min_size, num_classes, fpn_strides=None):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides

    def forward_for_single_feature_map(self, locations, box_cls,
        box_regression, bezier_regression, centerness, image_sizes, offsets
        =None):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        bezier_regression = bezier_regression.view(N, 16, H, W).permute(0, 
            2, 3, 1)
        bezier_regression = bezier_regression.reshape(N, -1, 16)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        if offsets is not None:
            offsets = torch.cat((offsets, mask), dim=1)
            offsets = offsets.permute(0, 2, 3, 1).reshape(N, H * W, -1)
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        box_cls = box_cls * centerness[:, :, (None)]
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, (0)]
            per_class = per_candidate_nonzeros[:, (1)] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_bezier_regression = bezier_regression[i]
            per_bezier_regression = per_bezier_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if offsets is not None:
                per_offsets = offsets[i]
                per_offsets = per_offsets[per_box_loc]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n
                    , sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_bezier_regression = per_bezier_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if offsets is not None:
                    per_offsets = per_offsets[top_k_indices]
            detections = torch.stack([per_locations[:, (0)] -
                per_box_regression[:, (0)], per_locations[:, (1)] -
                per_box_regression[:, (1)], per_locations[:, (0)] +
                per_box_regression[:, (2)], per_locations[:, (1)] +
                per_box_regression[:, (3)]], dim=1)
            bezier_detections = per_locations[:, ([1, 0])].unsqueeze(1
                ) + per_bezier_regression.view(-1, 8, 2)
            bezier_detections = bezier_detections.view(-1, 16)
            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            boxlist.add_field('beziers', bezier_detections)
            if offsets is not None:
                boxlist.add_field('offsets', per_offsets[:, :max_len * 2])
                boxlist.add_field('rec_masks', per_offsets[:, max_len * 2:]
                    .sigmoid())
                boxlist.add_field('locations', per_locations)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def forward(self, locations, box_cls, box_regression, bezier_regression,
        centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            bezier_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for i, (l, o, b, z, c) in enumerate(zip(locations, box_cls,
            box_regression, bezier_regression, centerness)):
            """
            if len(f) == 0:
                f = None
            else:
                f = f * self.fpn_strides[i]
            """
            sampled_boxes.append(self.forward_for_single_feature_map(l, o,
                b, z, c, image_sizes))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        has_offsets = boxlists[0].has_field('offsets')
        for i in range(num_images):
            scores = boxlists[i].get_field('scores')
            labels = boxlists[i].get_field('labels')
            if has_offsets:
                offsets = boxlists[i].get_field('offsets')
                locations = boxlists[i].get_field('locations')
                rec_masks = boxlists[i].get_field('rec_masks')
            beziers = boxlists[i].get_field('beziers')
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes[(inds), :].view(-1, 4)
                beziers_j = beziers[(inds), :].view(-1, 16)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
                boxlist_for_class.add_field('scores', scores_j)
                boxlist_for_class.add_field('beziers', beziers_j)
                if has_offsets:
                    boxlist_for_class.add_field('offsets', offsets[inds])
                    boxlist_for_class.add_field('locations', locations[inds])
                    boxlist_for_class.add_field('rec_masks', rec_masks[inds])
                boxlist_for_class = boxlist_nms(boxlist_for_class, self.
                    nms_thresh, score_field='scores')
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field('labels', torch.full((
                    num_labels,), j, dtype=torch.int64, device=scores.device))
                result.append(boxlist_for_class)
            result = cat_boxlist(result)
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                    number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


class PolarPredictor(nn.Module):
    """
    Use center point to predict all offsets
    """

    def __init__(self, in_channels, num_chars=32, voc_size=38, kernel_size=3):
        super(PolarPredictor, self).__init__()
        self.num_chars = num_chars
        self.locator = nn.Conv2d(in_channels, 3 * num_chars, kernel_size=3,
            stride=1, padding=1)
        self.clf = DeformConv(in_channels, voc_size, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.offset_repeat = kernel_size ** 2

    def forward(self, x, y, vis=False):
        """ Predict offsets with x and rec with y
        Offsets is relative starting from the center
        """
        N, _, H, W = x.size()
        features = self.locator(x)
        offsets, masks = features[:, :self.num_chars * 2], features[:, self
            .num_chars * 2:]
        location = offsets[:, :2]
        recs = [self.clf(y, location)]
        locations = [location]
        for i in range(1, self.num_chars):
            delta = offsets[:, i * 2:(i + 1) * 2]
            location = location + delta
            recs.append(self.clf(y, location))
            locations.append(location)
        return torch.stack(recs, dim=4), masks, torch.cat(locations, dim=1)


class SequentialPredictor(nn.Module):
    """
    Sequentially predict the offsets
    """

    def __init__(self, in_channels, num_chars=32, voc_size=38, kernel_size=3):
        super(SequentialPredictor, self).__init__()
        self.num_chars = num_chars
        self.voc_size = voc_size
        self.locator = nn.Conv2d(in_channels, num_chars + 2, kernel_size=3,
            stride=1, padding=1)
        self.clf = DeformConv(in_channels, voc_size + 2, kernel_size=1,
            stride=1, padding=0, bias=True)
        self.offset_repeat = kernel_size ** 2

    def forward(self, x, y, max_len):
        """ Predict offsets with x and rec with y
        Offsets is relative starting from the center
        """
        N, _, H, W = x.size()
        init_features = self.locator(x)
        location, masks = init_features[:, :2], init_features[:, 2:]
        recs = torch.zeros(N, self.voc_size, H, W, max_len)
        locations = torch.zeros(N, max_len * 2, H, W)
        delta = 0
        for i in range(max_len):
            location = location + delta
            locations[:, i * 2:i * 2 + 2] = location
            local_features = self.clf(y, location)
            rec, delta = local_features[:, :-2], local_features[:, -2:]
            recs[:, :, :, :, (i)] = rec
        return recs, masks, locations


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size,
        box_coder=None, fpn_post_nms_top_n=None, fpn_post_nms_per_batch=True):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder
        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        device = proposals[0].bbox.device
        gt_boxes = [target.copy_with_fields([]) for target in targets]
        for gt_box in gt_boxes:
            gt_box.add_field('objectness', torch.ones(len(gt_box), device=
                device))
        proposals = [cat_boxlist((proposal, gt_box)) for proposal, gt_box in
            zip(proposals, gt_boxes)]
        return proposals

    def forward_for_single_feature_map(self, anchors, objectness,
        box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        num_anchors = A * H * W
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted
            =True)
        batch_idx = torch.arange(N, device=device)[:, (None)]
        box_regression = box_regression[batch_idx, topk_idx]
        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        proposals = self.box_coder.decode(box_regression.view(-1, 4),
            concat_anchors.view(-1, 4))
        proposals = proposals.view(N, -1, 4)
        result = []
        for proposal, score, im_shape in zip(proposals, objectness,
            image_shapes):
            boxlist = BoxList(proposal, im_shape, mode='xyxy')
            boxlist.add_field('objectness', score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(boxlist, self.nms_thresh, max_proposals=
                self.post_nms_top_n, score_field='objectness')
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)
        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        if self.training and self.fpn_post_nms_per_batch:
            objectness = torch.cat([boxlist.get_field('objectness') for
                boxlist in boxlists], dim=0)
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0,
                sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field('objectness')
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim
                    =0, sorted=True)
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS
            ) * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            cls_tower.append(nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=1, padding=1))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes,
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
            kernel_size=3, stride=1, padding=1)
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits,
            self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE
    assert len(anchor_strides) == len(anchor_sizes), 'Only support FPN now'
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(
                scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
    anchor_generator = AnchorGenerator(tuple(new_anchor_sizes),
        aspect_ratios, anchor_strides, straddle_thresh)
    return anchor_generator


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls,
        box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C,
            H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level
            , N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
        generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(copied_fields)
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(anchors_per_image,
                targets_per_image, self.copied_fields)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            if 'not_visibility' in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field('visibility')
                    ] = -1
            if 'between_thresholds' in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox)
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in
            anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)
            ).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)
            ).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness, box_regression = concat_box_prediction_layers(objectness,
            box_regression)
        objectness = objectness.squeeze()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds], beta=1.0 / 9,
            size_average=False) / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[
            sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss


class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder, generate_labels_func,
        sigmoid_focal_loss, bbox_reg_beta=0.11, regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in
            anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        N = len(labels)
        box_cls, box_regression = concat_box_prediction_layers(box_cls,
            box_regression)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        retinanet_regression_loss = smooth_l1_loss(box_regression[pos_inds],
            regression_targets[pos_inds], beta=self.bbox_reg_beta,
            size_average=False) / max(1, pos_inds.numel() * self.regress_norm)
        labels = labels.int()
        retinanet_cls_loss = self.box_cls_loss_func(box_cls, labels) / (
            pos_inds.numel() + N)
        return retinanet_cls_loss, retinanet_regression_loss


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field('labels')
    return labels_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RETINANET.FG_IOU_THRESHOLD, cfg.MODEL.
        RETINANET.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    sigmoid_focal_loss = SigmoidFocalLoss(cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA)
    loss_evaluator = RetinaNetLossComputation(matcher, box_coder,
        generate_retinanet_labels, sigmoid_focal_loss, bbox_reg_beta=cfg.
        MODEL.RETINANET.BBOX_REG_BETA, regress_norm=cfg.MODEL.RETINANET.
        BBOX_REG_WEIGHT)
    return loss_evaluator


class RetinaNetPostProcessor(RPNPostProcessor):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
        fpn_post_nms_top_n, min_size, num_classes, box_coder=None):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetPostProcessor, self).__init__(pre_nms_thresh, 0,
            nms_thresh, min_size)
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)
        num_anchors = A * H * W
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors in zip(
            box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_box_cls = per_box_cls[per_candidate_inds]
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n,
                sorted=False)
            per_candidate_nonzeros = per_candidate_inds.nonzero()[(
                top_k_indices), :]
            per_box_loc = per_candidate_nonzeros[:, (0)]
            per_class = per_candidate_nonzeros[:, (1)]
            per_class += 1
            detections = self.box_coder.decode(per_box_regression[(
                per_box_loc), :].view(-1, 4), per_anchors.bbox[(per_box_loc
                ), :].view(-1, 4))
            boxlist = BoxList(detections, per_anchors.size, mode='xyxy')
            boxlist.add_field('labels', per_class)
            boxlist.add_field('scores', per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)
        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field('scores')
            labels = boxlists[i].get_field('labels')
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes[(inds), :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode='xyxy')
                boxlist_for_class.add_field('scores', scores_j)
                boxlist_for_class = boxlist_nms(boxlist_for_class, self.
                    nms_thresh, score_field='scores')
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field('labels', torch.full((
                    num_labels,), j, dtype=torch.int64, device=scores.device))
                result.append(boxlist_for_class)
            result = cat_boxlist(result)
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field('scores')
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                    number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_retinanet_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINANET.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINANET.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINANET.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0
    box_selector = RetinaNetPostProcessor(pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n, nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n, min_size=min_size,
        num_classes=config.MODEL.RETINANET.NUM_CLASSES, box_coder=rpn_box_coder
        )
    return box_selector


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RetinaNetModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = RetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        box_selector_test = make_retinanet_postprocessor(cfg, box_coder,
            is_train=False)
        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression,
                targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):
        loss_box_cls, loss_box_reg = self.loss_evaluator(anchors, box_cls,
            box_regression, targets)
        losses = {'loss_retina_cls': loss_box_cls, 'loss_retina_reg':
            loss_box_reg}
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1,
            stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
            kernel_size=1, stride=1)
        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]
        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=1, padding=1)
        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]
        return x


def make_anchor_generator(config):
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH
    if config.MODEL.RPN.USE_FPN:
        assert len(anchor_stride) == len(anchor_sizes
            ), 'FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)'
    else:
        assert len(anchor_stride
            ) == 1, 'Non-FPN should have a single ANCHOR_STRIDE'
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios,
        anchor_stride, straddle_thresh)
    return anchor_generator


def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field('matched_idxs')
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.
        BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.RPN.
        BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder,
        generate_rpn_labels)
    return loss_evaluator


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n, nms_thresh=nms_thresh, min_size=
        min_size, box_coder=rpn_box_coder, fpn_post_nms_top_n=
        fpn_post_nms_top_n, fpn_post_nms_per_batch=fpn_post_nms_per_batch)
    return box_selector


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs 
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()
        self.cfg = cfg.clone()
        anchor_generator = make_anchor_generator(cfg)
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head = rpn_head(cfg, in_channels, anchor_generator.
            num_anchors_per_location()[0])
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder,
            is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder,
            is_train=False)
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None, prefix=''):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
        if self.training:
            return self._forward_train(anchors, objectness,
                rpn_box_regression, targets, prefix)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression,
        targets, prefix):
        if self.cfg.MODEL.RPN_ONLY:
            boxes = anchors
        else:
            with torch.no_grad():
                boxes = self.box_selector_train(anchors, objectness,
                    rpn_box_regression, targets)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors,
            objectness, rpn_box_regression, targets)
        losses = {(prefix + 'loss_objectness'): loss_objectness, (prefix +
            'loss_rpn_box_reg'): loss_rpn_box_reg}
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            inds = [box.get_field('objectness').sort(descending=True)[1] for
                box in boxes]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


class Model(nn.Module):

    def __init__(self, input_size, output_size, scale):
        super(Model, self).__init__()
        self.bezier_align = BezierAlign(output_size, scale, 1)
        self.masks = nn.Parameter(torch.ones(input_size, dtype=torch.float32))

    def forward(self, input, rois):
        x = input * self.masks
        rois = self.convert_to_roi_format(rois)
        return self.bezier_align(x, rois)

    def convert_to_roi_format(self, beziers):
        concat_boxes = cat([b for b in beziers], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for
            i, b in enumerate(beziers)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Yuliang_Liu_bezier_curve_text_spotting(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BalancedL1Loss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BatchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BidirectionalLSTM(*[], **{'nIn': 4, 'nHidden': 4, 'nOut': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(CascadeConv3x3(*[], **{'C_in': 4, 'C_out': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(ChannelShuffle(*[], **{'groups': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(ConcatUpConv(*[], **{'inplanes': 4, 'outplanes': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(ContextBlock(*[], **{'inplanes': 4, 'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(ConvBNRelu(*[], **{'input_depth': 1, 'output_depth': 1, 'kernel': 4, 'stride': 1, 'pad': 4, 'no_bias': 4, 'use_relu': 'relu', 'bn_type': 'bn'}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_010(self):
        self._check(ConvTranspose2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(FPA(*[], **{}), [torch.rand([4, 2048, 64, 64])], {})

    def test_012(self):
        self._check(FrozenBatchNorm2d(*[], **{'n': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_013(self):
        self._check(GAU(*[], **{'channels_high': 4, 'channels_low': 4}), [torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16])], {})

    @_fails_compile()
    def test_014(self):
        self._check(IOULoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_015(self):
        self._check(IRFBlock(*[], **{'input_depth': 1, 'output_depth': 1, 'expansion': 4, 'stride': 1}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_016(self):
        self._check(Identity(*[], **{'C_in': 4, 'C_out': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_017(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(LastLevelMaxPool(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_019(self):
        self._check(LastLevelP6(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(LastLevelP6P7(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_021(self):
        self._check(NonLocal2D(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_022(self):
        self._check(SEModule(*[], **{'C': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_023(self):
        self._check(Scale(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_024(self):
        self._check(Shift(*[], **{'C': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_025(self):
        self._check(ShiftBlock5x5(*[], **{'C_in': 4, 'C_out': 4, 'expansion': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_026(self):
        self._check(ShuffleV2Block(*[], **{'input_depth': 64, 'output_depth': 64, 'expansion': 4, 'stride': 1}), [torch.rand([4, 32, 64, 64])], {})

