import sys
_module = sys.modules[__name__]
del sys
rrc_evaluation_funcs = _module
script = _module
dcn = _module
functions = _module
deform_conv = _module
deform_pool = _module
modules = _module
deform_conv = _module
deform_pool = _module
setup = _module
backbones = _module
mobilenetv3 = _module
resnet = _module
concern = _module
average_meter = _module
box2seg = _module
config = _module
convert = _module
icdar2015_eval = _module
detection = _module
deteval = _module
icdar2013 = _module
iou = _module
mtwi2018 = _module
log = _module
signal_monitor = _module
visualizer = _module
webcv2 = _module
manager = _module
server = _module
convert_to_onnx = _module
data = _module
augmenter = _module
data_loader = _module
dataset = _module
image_dataset = _module
make_border_map = _module
make_seg_detector_data = _module
meta_loader = _module
processes = _module
augment_data = _module
data_process = _module
filter_keys = _module
make_center_distance_map = _module
make_center_map = _module
make_center_points = _module
make_icdar_data = _module
make_seg_detection_data = _module
normalize_image = _module
random_crop_data = _module
resize_image = _module
serialize_box = _module
quad = _module
random_crop_aug = _module
simple_detection = _module
text_lines = _module
transform_data = _module
unpack_msgpack_data = _module
decoders = _module
balance_cross_entropy_loss = _module
dice_loss = _module
feature_attention = _module
l1_loss = _module
pss_loss = _module
seg_detector = _module
seg_detector_asf = _module
seg_detector_loss = _module
simple_detection = _module
demo = _module
eval = _module
experiment = _module
structure = _module
builder = _module
measurers = _module
icdar_detection_measurer = _module
quad_measurer = _module
model = _module
representers = _module
seg_detector_representer = _module
visualizers = _module
seg_detector_visualizer = _module
train = _module
trainer = _module
checkpoint = _module
learning_rate = _module
model_saver = _module
optimizer_scheduler = _module

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


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import math


import torch.nn as nn


from torch import nn


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


import numpy as np


import torch.distributed as dist


from torch.utils.data import Sampler


from torch.utils.data import ConcatDataset


from torch.utils.data import BatchSampler


from torch.utils.data import Dataset as TorchDataset


import functools


import logging


import torch.utils.data as data


from collections import OrderedDict


from scipy import ndimage


import time


import torch.optim.lr_scheduler as lr_scheduler


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
        super(DeformConv, self).__init__()
        assert not bias
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

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformConvPack(DeformConv):

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


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

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset_mask = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


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

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, deform_fc_channels=1024):
        super(DeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size * 2
                seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*seq)
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

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, num_mask_fcs=2, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.num_mask_fcs = num_mask_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            offset_fc_seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size * 2
                offset_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    offset_fc_seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*offset_fc_seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            mask_fc_seq = []
            ic = self.out_size * self.out_size * self.out_channels
            for i in range(self.num_mask_fcs):
                if i < self.num_mask_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size * self.out_size
                mask_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_mask_fcs - 1:
                    mask_fc_seq.append(nn.ReLU(inplace=True))
                else:
                    mask_fc_seq.append(nn.Sigmoid())
            self.mask_fc = nn.Sequential(*mask_fc_seq)
            self.mask_fc[-2].weight.data.zero_()
            self.mask_fc[-2].bias.data.zero_()

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


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), Hsigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):

    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):

    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity
        self.conv = nn.Sequential(conv_layer(inp, exp, 1, 1, 0, bias=False), norm_layer(exp), nlin_layer(inplace=True), conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False), norm_layer(exp), SELayer(exp), nlin_layer(inplace=True), conv_layer(exp, oup, 1, 1, 0, bias=False), norm_layer(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(conv_layer(inp, oup, 1, 1, 0, bias=False), norm_layer(oup), nlin_layer(inplace=True))


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(conv_layer(inp, oup, 3, stride, 1, bias=False), norm_layer(oup), nlin_layer(inplace=True))


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileNetV3(nn.Module):

    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            mobile_setting = [[3, 16, 16, False, 'RE', 1], [3, 64, 24, False, 'RE', 2], [3, 72, 24, False, 'RE', 1], [5, 72, 40, True, 'RE', 2], [5, 120, 40, True, 'RE', 1], [5, 120, 40, True, 'RE', 1], [3, 240, 80, False, 'HS', 2], [3, 200, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 480, 112, True, 'HS', 1], [3, 672, 112, True, 'HS', 1], [5, 672, 160, True, 'HS', 2], [5, 960, 160, True, 'HS', 1], [5, 960, 160, True, 'HS', 1]]
        elif mode == 'small':
            mobile_setting = [[3, 16, 16, True, 'RE', 2], [3, 72, 24, False, 'RE', 2], [3, 88, 24, False, 'RE', 1], [5, 96, 40, True, 'HS', 2], [5, 240, 40, True, 'HS', 1], [5, 240, 40, True, 'HS', 1], [5, 120, 48, True, 'HS', 1], [5, 144, 48, True, 'HS', 1], [5, 288, 96, True, 'HS', 2], [5, 576, 96, True, 'HS', 1], [5, 576, 96, True, 'HS', 1]]
        else:
            raise NotImplementedError
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2, nlin_layer=Hswish)])
        self.classifier = []
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        """x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x"""
        x2, x3, x4, x5 = None, None, None, None
        for stage in range(17):
            x = self.features[stage](x)
            if stage == 3:
                x2 = x
            elif stage == 6:
                x3 = x
            elif stage == 12:
                x4 = x
            elif stage == 16:
                x5 = x
        return x2, x3, x4, x5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, kernel_size=3, padding=1)
            self.conv2 = conv_op(planes, planes, kernel_size=3, padding=1, deformable_groups=deformable_groups, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, kernel_size=3, padding=1)
            self.conv2 = conv_op(planes, planes, kernel_size=3, padding=1, stride=stride, deformable_groups=deformable_groups, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dcn=None, stage_with_dcn=(False, False, False, False)):
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5


class BalanceCrossEntropyLoss(nn.Module):
    """
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, negative_ratio=3.0, eps=1e-06):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, return_origin=False):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        positive = (gt[:, 0, :, :] * mask).byte()
        negative = ((1 - gt[:, 0, :, :]) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')[:, 0, :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    """
    DiceLoss on binary.
    For SegDetector without adaptive module.
    """

    def __init__(self, eps=1e-06):
        super(DiceLoss, self).__init__()
        self.loss = Loss(eps)

    def forward(self, pred, batch):
        loss = self.loss(pred['binary'], batch['gt'], batch['mask'])
        return loss, dict(dice_loss=loss)


class LeakyDiceLoss(nn.Module):
    """
    Variation from DiceLoss.
    The coverage and union are computed separately.
    """

    def __init__(self, eps=1e-06, coverage_scale=5.0):
        super(LeakyDiceLoss, self).__init__()
        self.eps = eps
        self.coverage_scale = coverage_scale

    def forward(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        coverage = (pred * mask * gt).sum() / ((gt * mask).sum() + self.eps)
        assert coverage <= 1
        coverage = 1 - coverage
        excede = (pred * mask * gt).sum() / ((pred * mask).sum() + self.eps)
        assert excede <= 1
        excede = 1 - excede
        loss = coverage * self.coverage_scale + excede
        return loss, dict(coverage=coverage, excede=excede)


class InstanceDiceLoss(DiceLoss):
    """
    DiceLoss normalized on each instance.
    Input:
        pred: (N, 1, H, W)
        gt: (N, 1, H, W)
        mask: (N, H, W)
    Note: This class assume that input tensors are on gpu,
        while cput computation is required to find union areas.
    """
    REDUCTION = ['mean', 'sum', 'none']

    def __init__(self, threshold=0.3, iou_thresh=0.2, reduction=None, max_regions=100, eps=1e-06):
        nn.Module.__init__(self)
        self.threshold = threshold
        self.iou_thresh = iou_thresh
        self.reduction = reduction
        if self.reduction is None:
            self.reduction = 'mean'
        assert self.reduction in self.REDUCTION
        self.max_regions = max_regions
        self.eps = eps

    def label(self, tensor_on_gpu, blur=None):
        """
        Args:
            tensor_on_gpu: (N, 1, H, W)
            blur: Lambda. If exists, each instance will be blured using `blur`.
        """
        tensor = tensor_on_gpu.cpu().detach().numpy()
        instance_maps = []
        instance_counts = []
        for batch_index in range(tensor_on_gpu.shape[0]):
            instance = tensor[batch_index]
            if blur is not None:
                instance = blur(instance)
            lable_map, instance_count = ndimage.label(instance[0])
            instance_count = min(self.max_regions, instance_count)
            instance_map = []
            for index in range(1, instance_count):
                instance = torch.from_numpy(lable_map == index).type(torch.float32)
                instance_map.append(instance)
            instance_maps.append(instance_map)
        return instance_maps, instance_counts

    def iou(self, pred, gt):
        overlap = (pred * gt).sum()
        return max(overlap / pred.sum(), overlap / gt.sum())

    def replace_or_add(self, dest, value):
        if dest is None:
            return value
        if value is None:
            return dest
        return dest + value

    def forward(self, pred, gt, mask):
        torch.cuda.synchronize()
        pred_label_maps, _ = self.label(pred > self.threshold)
        gt_label_maps, _ = self.label(gt)
        losses = []
        for batch_index, gt_instance_maps in enumerate(gt_label_maps):
            pred_instance_maps = pred_label_maps[batch_index]
            if gt_instance_maps is None or pred_instance_maps is None:
                continue
            single_loss = None
            mask_not_matched = set(range(len(pred_instance_maps)))
            for gt_instance_map in gt_instance_maps:
                instance_loss = None
                for instance_index, pred_instance_map in enumerate(pred_instance_maps):
                    if self.iou(pred_instance_map, gt_instance_map) > self.iou_thresh:
                        match_loss = self._compute(pred[batch_index][0], gt[batch_index][0], mask[batch_index] * (pred_instance_map + gt_instance_map > 0).type(torch.float32))
                        instance_loss = self.replace_or_add(instance_loss, match_loss)
                        if instance_index in mask_not_matched:
                            mask_not_matched.remove(instance_index)
                if instance_loss is None:
                    instance_loss = self._compute(pred[batch_index][0], gt[batch_index][0], mask[batch_index] * gt_instance_map)
                single_loss = self.replace_or_add(single_loss, instance_loss)
            """Whether to compute single loss on instances which contrain no positive sample.
            if single_loss is None:
                single_loss = self._compute(
                        pred[batch_index][0], gt[batch_index][0],
                        mask[batch_index])
            """
            for instance_index in mask_not_matched:
                single_loss = self.replace_or_add(single_loss, self._compute(pred[batch_index][0], gt[batch_index][0], mask[batch_index] * pred_instance_maps[instance_index]))
            if single_loss is not None:
                losses.append(single_loss)
        if self.reduction == 'none':
            loss = losses
        else:
            assert self.reduction in ['sum', 'mean']
            count = len(losses)
            loss = sum(losses)
            if self.reduction == 'mean':
                loss = loss / count
        return loss


class ScaleChannelAttention(nn.Module):

    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        None
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, bias=False)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = F.softmax(global_x, 1)
        return global_x


class ScaleChannelSpatialAttention(nn.Module):

    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_planes, out_planes, 1, bias=False), nn.ReLU(), nn.Conv2d(out_planes, in_planes, 1, bias=False))
        self.spatial_wise = nn.Sequential(nn.Conv2d(1, 1, 3, bias=False, padding=1), nn.ReLU(), nn.Conv2d(1, 1, 1, bias=False), nn.Sigmoid())
        self.attention_wise = nn.Sequential(nn.Conv2d(in_planes, num_features, 1, bias=False), nn.Sigmoid())
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = self.channel_wise(x).sigmoid()
        global_x = global_x + x
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleSpatialAttention(nn.Module):

    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(nn.Conv2d(1, 1, 3, bias=False, padding=1), nn.ReLU(), nn.Conv2d(1, 1, 1, bias=False), nn.Sigmoid())
        self.attention_wise = nn.Sequential(nn.Conv2d(in_planes, num_features, 1, bias=False), nn.Sigmoid())
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleFeatureSelection(nn.Module):

    def __init__(self, in_channels, inter_channels, out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        if self.type == 'scale_spatial':
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels // 2, out_features_num)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0001)

    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i + 1] * features_list[i])
        return torch.cat(x, dim=1)


class MaskL1Loss(nn.Module):

    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss=mask_sum)
        else:
            loss = (torch.abs(pred[:, 0] - gt) * mask).sum() / mask_sum
            return loss, dict(l1_loss=loss)


class BalanceL1Loss(nn.Module):

    def __init__(self, negative_ratio=3.0):
        super(BalanceL1Loss, self).__init__()
        self.negative_ratio = negative_ratio

    def forward(self, pred: torch.Tensor, gt, mask):
        """
        Args:
            pred: (N, 1, H, W).
            gt: (N, H, W).
            mask: (N, H, W).
        """
        loss = torch.abs(pred[:, 0] - gt)
        positive = loss * mask
        negative = loss * (1 - mask)
        positive_count = int(mask.sum())
        negative_count = min(int((1 - mask).sum()), int(positive_count * self.negative_ratio))
        negative_loss, _ = torch.topk(negative.view(-1), negative_count)
        negative_loss = negative_loss.sum() / negative_count
        positive_loss = positive.sum() / positive_count
        return positive_loss + negative_loss, dict(l1_loss=positive_loss, nge_l1_loss=negative_loss)


class PSS_Loss(nn.Module):

    def __init__(self, cls_loss):
        super(PSS_Loss, self).__init__()
        self.eps = 1e-06
        self.criterion = eval('self.' + cls_loss + '_loss')

    def dice_loss(self, pred, gt, m):
        intersection = torch.sum(pred * gt * m)
        union = torch.sum(pred * m) + torch.sum(gt * m) + self.eps
        loss = 1 - 2.0 * intersection / union
        if loss > 1:
            None
        return loss

    def dice_ohnm_loss(self, pred, gt, m):
        pos_index = (gt == 1) * (m == 1)
        neg_index = (gt == 0) * (m == 1)
        pos_num = pos_index.float().sum().item()
        neg_num = neg_index.float().sum().item()
        if pos_num == 0 or neg_num < pos_num * 3.0:
            return self.dice_loss(pred, gt, m)
        else:
            neg_num = int(pos_num * 3)
            pos_pred = pred[pos_index]
            neg_pred = pred[neg_index]
            neg_sort, _ = torch.sort(neg_pred, descending=True)
            sampled_neg_pred = neg_sort[:neg_num]
            pos_gt = pos_pred.clone()
            pos_gt.data.fill_(1.0)
            pos_gt = pos_gt.detach()
            neg_gt = sampled_neg_pred.clone()
            neg_gt.data.fill_(0)
            neg_gt = neg_gt.detach()
            tpred = torch.cat((pos_pred, sampled_neg_pred))
            tgt = torch.cat((pos_gt, neg_gt))
            intersection = torch.sum(tpred * tgt)
            union = torch.sum(tpred) + torch.sum(gt) + self.eps
            loss = 1 - 2.0 * intersection / union
        return loss

    def focal_loss(self, pred, gt, m, alpha=0.25, gamma=0.6):
        pos_mask = (gt == 1).float()
        neg_mask = (gt == 0).float()
        mask = alpha * pos_mask * torch.pow(1 - pred.data, gamma) + (1 - alpha) * neg_mask * torch.pow(pred.data, gamma)
        l = F.binary_cross_entropy(pred, gt, weight=mask, reduction='none')
        loss = torch.sum(l * m) / (self.eps + m.sum())
        loss *= 10
        return loss

    def wbce_orig_loss(self, pred, gt, m):
        n, h, w = pred.size()
        assert torch.max(gt) == 1
        pos_neg_p = pred[m.byte()]
        pos_neg_t = gt[m.byte()]
        pos_mask = (pos_neg_t == 1).squeeze()
        w = pos_mask.float() * (1 - pos_mask).sum().item() + (1 - pos_mask).float() * pos_mask.sum().item()
        w = w / pos_mask.size(0)
        loss = F.binary_cross_entropy(pos_neg_p, pos_neg_t, w, reduction='sum')
        return loss

    def wbce_loss(self, pred, gt, m):
        pos_mask = (gt == 1).float() * m
        neg_mask = (gt == 0).float() * m
        mask = pos_mask * neg_mask.sum() / pos_mask.sum() + neg_mask
        l = F.binary_cross_entropy(pred, gt, weight=mask, reduction='none')
        loss = torch.sum(l) / (m.sum() + self.eps)
        return loss

    def bce_loss(self, pred, gt, m):
        l = F.binary_cross_entropy(pred, gt, weight=m, reduction='sum')
        loss = l / (m.sum() + self.eps)
        return loss

    def dice_bce_loss(self, pred, gt, m):
        return (self.dice_loss(pred, gt, m) + self.bce_loss(pred, gt, m)) / 2.0

    def dice_ohnm_bce_loss(self, pred, gt, m):
        return (self.dice_ohnm_loss(pred, gt, m) + self.bce_loss(pred, gt, m)) / 2.0

    def forward(self, pred, gt, mask, gt_type='shrink'):
        if gt_type == 'shrink':
            loss = self.get_loss(pred, gt, mask)
            return loss
        elif gt_type == 'pss':
            loss = self.get_loss(pred, gt[:, :4, :, :], mask)
            g_g = gt[:, 4, :, :]
            g_p, _ = torch.max(pred, 1)
            loss += self.criterion(g_p, g_g, mask)
            return loss
        elif gt_type == 'both':
            pss_loss = self.get_loss(pred[:, :4, :, :], gt[:, :4, :, :], mask)
            g_g = gt[:, 4, :, :]
            g_p, _ = torch.max(pred, 1)
            pss_loss += self.criterion(g_p, g_g, mask)
            shrink_loss = self.criterion(pred[:, 4, :, :], gt[:, 5, :, :], mask)
            return pss_loss, shrink_loss
        else:
            return NotImplementedError('gt_type [%s] is not implemented', gt_type)

    def get_loss(self, pred, gt, mask):
        loss = torch.tensor(0.0)
        for ind in range(pred.size(1)):
            loss += self.criterion(pred[:, ind, :, :], gt[:, ind, :, :], mask)
        return loss


class SegDetector(nn.Module):

    def __init__(self, in_channels=[64, 128, 256, 512], inner_channels=256, k=10, bias=False, adaptive=False, smooth=False, serial=False, *args, **kwargs):
        """
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        """
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)
        self.out5 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)
        self.binarize = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        self.binarize.apply(self.weights_init)
        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)
        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0001)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias), nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        out4 = self.up5(in5) + in4
        out3 = self.up4(out4) + in3
        out2 = self.up3(out3) + in2
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat((fuse, nn.functional.interpolate(binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class SegSpatialScaleDetector(nn.Module):

    def __init__(self, in_channels=[64, 128, 256, 512], inner_channels=256, k=10, bias=False, adaptive=False, smooth=False, serial=False, fpn=True, attention_type='scale_spatial', *args, **kwargs):
        """
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        """
        super(SegSpatialScaleDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.fpn = fpn
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)
        if self.fpn:
            self.out5 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.Upsample(scale_factor=8, mode='nearest'))
            self.out4 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.Upsample(scale_factor=4, mode='nearest'))
            self.out3 = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.Upsample(scale_factor=2, mode='nearest'))
            self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)
            self.out5.apply(self.weights_init)
            self.out4.apply(self.weights_init)
            self.out3.apply(self.weights_init)
            self.out2.apply(self.weights_init)
            self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels // 4, attention_type=attention_type)
            self.binarize = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, bias=bias, padding=1), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        else:
            self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels // 4)
            self.binarize = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, bias=bias, padding=1), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        self.binarize.apply(self.weights_init)
        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)
        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0001)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias), BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias), nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        out4 = self.up5(in5) + in4
        out3 = self.up4(out4) + in3
        out2 = self.up3(out3) + in2
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.concat_attention(fuse, [p5, p4, p3, p2])
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat((fuse, nn.functional.interpolate(binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BalanceBCELoss(nn.Module):
    """
    DiceLoss on binary.
    For SegDetector without adaptive module.
    """

    def __init__(self, eps=1e-06):
        super(BalanceBCELoss, self).__init__()
        self.loss = BalanceCrossEntropyLoss()

    def forward(self, pred, batch):
        loss = self.loss(pred['binary'], batch['gt'], batch['mask'])
        return loss, dict(dice_loss=loss)


class AdaptiveDiceLoss(nn.Module):
    """
    Integration of DiceLoss on both binary
        prediction and thresh prediction.
    """

    def __init__(self, eps=1e-06):
        super(AdaptiveDiceLoss, self).__init__()
        self.main_loss = DiceLoss(eps)
        self.thresh_loss = DiceLoss(eps)

    def forward(self, pred, batch):
        assert isinstance(pred, dict)
        assert 'binary' in pred
        assert 'thresh_binary' in pred
        binary = pred['binary']
        thresh_binary = pred['thresh_binary']
        gt = batch['gt']
        mask = batch['mask']
        main_loss = self.main_loss(binary, gt, mask)
        thresh_loss = self.thresh_loss(thresh_binary, gt, mask)
        loss = main_loss + thresh_loss
        return loss, dict(main_loss=main_loss, thresh_loss=thresh_loss)


class AdaptiveInstanceDiceLoss(nn.Module):
    """
    InstanceDiceLoss on both binary and thresh_bianry.
    """

    def __init__(self, iou_thresh=0.2, thresh=0.3):
        super(AdaptiveInstanceDiceLoss, self).__init__()
        self.main_loss = DiceLoss()
        self.main_instance_loss = InstanceDiceLoss()
        self.thresh_loss = DiceLoss()
        self.thresh_instance_loss = InstanceDiceLoss()
        self.weights = nn.ParameterDict(dict(main=nn.Parameter(torch.ones(1)), thresh=nn.Parameter(torch.ones(1)), main_instance=nn.Parameter(torch.ones(1)), thresh_instance=nn.Parameter(torch.ones(1))))

    def partial_loss(self, weight, loss):
        return loss / weight + torch.log(torch.sqrt(weight))

    def forward(self, pred, batch):
        main_loss = self.main_loss(pred['binary'], batch['gt'], batch['mask'])
        thresh_loss = self.thresh_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
        main_instance_loss = self.main_instance_loss(pred['binary'], batch['gt'], batch['mask'])
        thresh_instance_loss = self.thresh_instance_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
        loss = self.partial_loss(self.weights['main'], main_loss) + self.partial_loss(self.weights['thresh'], thresh_loss) + self.partial_loss(self.weights['main_instance'], main_instance_loss) + self.partial_loss(self.weights['thresh_instance'], thresh_instance_loss)
        metrics = dict(main_loss=main_loss, thresh_loss=thresh_loss, main_instance_loss=main_instance_loss, thresh_instance_loss=thresh_instance_loss)
        metrics.update(self.weights)
        return loss, metrics


class L1DiceLoss(nn.Module):
    """
    L1Loss on thresh, DiceLoss on thresh_binary and binary.
    """

    def __init__(self, eps=1e-06, l1_scale=10):
        super(L1DiceLoss, self).__init__()
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.l1_scale = l1_scale

    def forward(self, pred, batch):
        dice_loss, metrics = self.dice_loss(pred, batch)
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        loss = dice_loss + self.l1_scale * l1_loss
        metrics.update(**l1_metric)
        return loss, metrics


class FullL1DiceLoss(L1DiceLoss):
    """
    L1loss on thresh, pixels with topk losses in non-text regions are also counted.
    DiceLoss on thresh_binary and binary.
    """

    def __init__(self, eps=1e-06, l1_scale=10):
        nn.Module.__init__(self)
        self.dice_loss = AdaptiveDiceLoss(eps=eps)
        self.l1_loss = BalanceL1Loss()
        self.l1_scale = l1_scale


class L1BalanceCELoss(nn.Module):
    """
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    """

    def __init__(self, eps=1e-06, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        metrics = dict(bce_loss=bce_loss)
        if 'thresh' in pred:
            l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
            dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
            metrics['thresh_loss'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics


class L1BCEMiningLoss(nn.Module):
    """
    Basicly the same with L1BalanceCELoss, where the bce loss map is used as
        attention weigts for DiceLoss
    """

    def __init__(self, eps=1e-06, l1_scale=10, bce_scale=5):
        super(L1BCEMiningLoss, self).__init__()
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        bce_loss, bce_map = self.bce_loss(pred['binary'], batch['gt'], batch['mask'], return_origin=True)
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        bce_map = (bce_map - bce_map.min()) / (bce_map.max() - bce_map.min())
        dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'], weights=bce_map + 1)
        metrics = dict(bce_loss=bce_loss)
        metrics['thresh_loss'] = dice_loss
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics.update(**l1_metric)
        return loss, metrics


class L1LeakyDiceLoss(nn.Module):
    """
    LeakyDiceLoss on binary,
    MaskL1Loss on thresh,
    DiceLoss on thresh_binary.
    """

    def __init__(self, eps=1e-06, coverage_scale=5, l1_scale=10):
        super(L1LeakyDiceLoss, self).__init__()
        self.main_loss = LeakyDiceLoss(coverage_scale=coverage_scale)
        self.l1_loss = MaskL1Loss()
        self.thresh_loss = DiceLoss(eps=eps)
        self.l1_scale = l1_scale

    def forward(self, pred, batch):
        main_loss, metrics = self.main_loss(pred['binary'], batch['gt'], batch['mask'])
        thresh_loss = self.thresh_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        metrics.update(**l1_metric, thresh_loss=thresh_loss)
        loss = main_loss + thresh_loss + l1_loss * self.l1_scale
        return loss, metrics


class SimpleDetectionDecoder(nn.Module):

    def __init__(self, feature_channel=256):
        nn.Module.__init__(self)
        self.feature_channel = feature_channel
        self.head_layer = self.create_head_layer()
        self.pred_layers = nn.ModuleDict(self.create_pred_layers())

    def create_head_layer(self):
        return SimpleUpsampleHead(self.feature_channel, [self.feature_channel, self.feature_channel // 2, self.feature_channel // 4])

    def create_pred_layer(self, channels):
        return nn.Sequential(nn.Conv2d(self.feature_channel // 4, channels, kernel_size=1, stride=1, padding=0, bias=False))

    def create_pred_layers(self):
        return {}

    def postprocess_pred(self, pred):
        return pred

    def calculate_losses(self, preds, label):
        raise NotImplementedError()

    def forward(self, input, label, meta, train):
        feature = self.head_layer(input)
        pred = {}
        for name, pred_layer in self.pred_layers.items():
            pred[name] = pred_layer(feature)
        if train:
            losses = self.calculate_losses(pred, label)
            pred = self.postprocess_pred(pred)
            loss = sum(losses.values())
            return loss, pred, losses
        else:
            pred = self.postprocess_pred(pred)
            return pred


class SimpleSegDecoder(SimpleDetectionDecoder):

    def create_pred_layers(self):
        return {'heatmap': self.create_pred_layer(1)}

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        heatmap_pred = pred['heatmap']
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        return {'heatmap_loss': heatmap_loss}


class SimpleEASTDecoder(SimpleDetectionDecoder):

    def __init__(self, feature_channels=256, densebox_ratio=1000.0, densebox_rescale_factor=512):
        SimpleDetectionDecoder.__init__(self, feature_channels)
        self.densebox_ratio = densebox_ratio
        self.densebox_rescale_factor = densebox_rescale_factor

    def create_pred_layers(self):
        return {'heatmap': self.create_pred_layer(1), 'densebox': self.create_pred_layer(8)}

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        pred['densebox'] = pred['densebox'] * self.densebox_rescale_factor
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        densebox = label['densebox'] / self.densebox_rescale_factor
        densebox_weight = label['densebox_weight']
        heatmap_pred = pred['heatmap']
        densebox_pred = pred['densebox']
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        densebox_loss = F.mse_loss(densebox_pred, densebox, reduction='none')
        densebox_loss = (densebox_loss * densebox_weight).mean(dim=(1, 2, 3)) * self.densebox_ratio
        return {'heatmap_loss': heatmap_loss, 'densebox_loss': densebox_loss}


class SimpleTextsnakeDecoder(SimpleDetectionDecoder):

    def __init__(self, feature_channels=256, radius_ratio=10.0):
        SimpleDetectionDecoder.__init__(self, feature_channels)
        self.radius_ratio = radius_ratio

    def create_pred_layers(self):
        return {'heatmap': self.create_pred_layer(1), 'radius': self.create_pred_layer(1)}

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        pred['radius'] = torch.exp(pred['radius'])
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        radius = torch.log(label['radius'] + 1)
        radius_weight = label['radius_weight']
        heatmap_pred = pred['heatmap']
        radius_pred = pred['radius']
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        radius_loss = F.smooth_l1_loss(radius_pred, radius, reduction='none')
        radius_loss = (radius_loss * radius_weight).mean(dim=(1, 2, 3)) * self.radius_ratio
        return {'heatmap_loss': heatmap_loss, 'radius_loss': radius_loss}


class SimpleMSRDecoder(SimpleDetectionDecoder):

    def __init__(self, feature_channels=256, offset_ratio=1000.0, offset_rescale_factor=512):
        SimpleDetectionDecoder.__init__(self, feature_channels)
        self.offset_ratio = offset_ratio
        self.offset_rescale_factor = offset_rescale_factor

    def create_pred_layers(self):
        return {'heatmap': self.create_pred_layer(1), 'offset': self.create_pred_layer(2)}

    def postprocess_pred(self, pred):
        pred['heatmap'] = F.sigmoid(pred['heatmap'])
        pred['offset'] = pred['offset'] * self.offset_rescale_factor
        return pred

    def calculate_losses(self, pred, label):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        offset = label['offset'] / self.offset_rescale_factor
        offset_weight = label['offset_weight']
        heatmap_pred = pred['heatmap']
        offset_pred = pred['offset']
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        offset_loss = F.mse_loss(offset_pred, offset, reduction='none')
        offset_loss = (offset_loss * offset_weight).mean(dim=(1, 2, 3)) * self.offset_ratio
        return {'heatmap_loss': heatmap_loss, 'offset_loss': offset_loss}


class BasicModel(nn.Module):

    def __init__(self, args):
        nn.Module.__init__(self)
        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=[local_rank], find_unused_parameters=True)
    else:
        return nn.DataParallel(model)


class SegDetectorModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        super(SegDetectorModel, self).__init__()
        self.model = BasicModel(args)
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image']
        else:
            data = batch
        data = data.float()
        pred = self.model(data, training=self.training)
        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BalanceCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BalanceL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeakyDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MaskL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileBottleneck,
     lambda: ([], {'inp': 4, 'oup': 4, 'kernel': 3, 'stride': 1, 'exp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleChannelAttention,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleChannelSpatialAttention,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleFeatureSelection,
     lambda: ([], {'in_channels': 4, 'inter_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaleSpatialAttention,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MhLiao_DB(_paritybench_base):
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

