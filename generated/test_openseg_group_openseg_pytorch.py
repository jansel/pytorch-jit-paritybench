import sys
_module = sys.modules[__name__]
del sys
lib = _module
datasets = _module
data_loader = _module
loader = _module
ade20k_loader = _module
default_loader = _module
lip_loader = _module
preprocess = _module
ade20k = _module
ade20k_generator = _module
cityscapes = _module
cityscapes_generator = _module
cityscapes_instance_generator = _module
edge_generator = _module
instance_edge_generator = _module
lip = _module
mapillary_generator = _module
pascal_context_generator = _module
pascal_voc_generator = _module
tools = _module
collate = _module
cv2_aug_transforms = _module
pil_aug_transforms = _module
transforms = _module
extensions = _module
cc_attention = _module
_ext = _module
build = _module
functions = _module
dense_crf = _module
dcn = _module
deform_conv = _module
modulated_dcn = _module
build_modulated = _module
deform_conv = _module
modulated_dcn_func = _module
modules = _module
deform_conv = _module
modulated_dcn = _module
test = _module
test_modulated = _module
frn = _module
frn = _module
inplace_abn = _module
bn = _module
inplace_abn_1 = _module
bn = _module
misc = _module
pacnet = _module
pac = _module
paccrf = _module
test_pac = _module
parallel = _module
_functions = _module
data_container = _module
data_parallel = _module
distributed = _module
scatter_gather = _module
switchablenorms = _module
switchable_norm = _module
syncbn = _module
allreduce = _module
comm = _module
module = _module
loss = _module
loss_helper = _module
loss_manager = _module
F1_running_score = _module
metrics = _module
ade20k_evaluator = _module
evaluation = _module
csHelpers = _module
evalInstanceLevelSemanticLabeling = _module
evalPixelLevelSemanticLabeling = _module
instance = _module
instances2dict = _module
helpers = _module
annotation = _module
labels = _module
labels_cityPersons = _module
setup = _module
cityscapes_evaluator = _module
cocostuff_evaluator = _module
pascal_context_evaluator = _module
running_score = _module
running_score_mp = _module
models = _module
backbones = _module
backbone_selector = _module
hrnet = _module
hrnet_backbone = _module
hrnet_config = _module
resnet = _module
dcn_resnet_models = _module
resnest_models = _module
resnet_backbone = _module
resnet_models = _module
resnext_models = _module
wide_resnet_models = _module
wsl_resnext_models = _module
model_manager = _module
asp_oc_block = _module
base_oc_block = _module
decoder_block = _module
edge_block = _module
isa_block = _module
offset_block = _module
spatial_ocr_block = _module
nets = _module
ce2pnet = _module
fcnet = _module
hrnet = _module
ideal_ocrnet = _module
isanet = _module
ocnet = _module
ocrnet = _module
module_helper = _module
utils = _module
distributed = _module
dc_helper = _module
file_helper = _module
image_helper = _module
json_helper = _module
mask_helper = _module
video_helper = _module
average_meter = _module
configer = _module
logger = _module
timer = _module
vis = _module
attention_visualizer = _module
log_visualizer = _module
palette = _module
seg_parser = _module
seg_visualizer = _module
tensor_visualizer = _module
main = _module
segfix = _module
segfix_instance = _module
segmentor = _module
tester = _module
blob_helper = _module
cost_helper = _module
data_helper = _module
evaluator = _module
base = _module
standard = _module
tasks = _module
module_runner = _module
optim_scheduler = _module
trainer = _module

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


import random


import torch


import torch.nn.functional as F


from torch.utils.data.dataloader import default_collate


import torch.autograd as autograd


import torch.nn as nn


from torch.autograd.function import once_differentiable


from torch.autograd import Function


from torch.nn.modules.utils import _pair


import math


from torch.nn.modules.module import Module


from torch import nn


from torch.autograd import Variable


import time


from torch.autograd import gradcheck


import torch.nn.functional as functional


import torch.distributed as dist


from numbers import Number


from itertools import repeat


import numpy as np


from torch.autograd.function import Function


from torch.nn.parameter import Parameter


import torch as th


from torch.nn.parallel._functions import _get_stream


import functools


import torch.cuda.comm as comm


from torch.nn.parallel._functions import Broadcast


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel.scatter_gather import gather


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


from torch._utils import _take_tensors


from torch.nn.parallel._functions import Scatter as OrigScatter


import collections


from torch.nn.functional import batch_norm


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.utils.cpp_extension import load


import torch.utils.checkpoint as cp


from collections import OrderedDict


from torch.nn import Conv2d


from torch.nn import Module


from torch.nn import Linear


from torch.nn import ReLU


from functools import partial


from torch.nn import functional as F


import scipy.io as io


import scipy


from scipy import ndimage


from math import ceil


import torch.backends.cudnn as cudnn


from collections import Counter


from torch.nn.parallel.scatter_gather import gather as torch_gather


def _check_contiguous(*args):
    if not all([(mod is None or mod.is_contiguous()) for mod in args]):
        raise ValueError('Non-contiguous input')


class CA_Map(autograd.Function):

    @staticmethod
    def forward(ctx, weight, g):
        out = torch.zeros_like(g)
        _ext.ca_map_forward_cuda(weight, g, out)
        ctx.save_for_backward(weight, g)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        _ext.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
        _check_contiguous(dw, dg)
        return dw, dg


ca_map = CA_Map.apply


class CA_Weight(autograd.Function):

    @staticmethod
    def forward(ctx, t, f):
        n, c, h, w = t.size()
        size = n, h + w - 1, h, w
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t
            .device)
        _ext.ca_forward_cuda(t, f, weight)
        ctx.save_for_backward(t, f)
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        _ext.ca_backward_cuda(dw.contiguous(), t, f, dt, df)
        _check_contiguous(dt, df)
        return dt, df


ca_weight = CA_Weight.apply


class CrossAttention(nn.Module):

    def __init__(self, dim_in, dim_inner, dim_out):
        super(CrossAttention, self).__init__()
        self.t_func = nn.Conv2d(in_channels=dim_in, out_channels=dim_inner,
            kernel_size=1, stride=1, padding=0)
        self.f_func = nn.Conv2d(in_channels=dim_in, out_channels=dim_inner,
            kernel_size=1, stride=1, padding=0)
        self.g_func = nn.Conv2d(in_channels=dim_in, out_channels=dim_out,
            kernel_size=1, stride=1, padding=0)
        self.inc = nn.Conv2d(in_channels=dim_out, out_channels=dim_in,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.inc.weight, 0)
        nn.init.constant_(self.inc.bias, 0)

    def forward(self, x):
        t = self.t_func(x)
        f = self.f_func(x)
        g = self.g_func(x)
        w = ca_weight(t, f)
        w = F.softmax(w, 1)
        out = ca_map(w, g)
        x = x + self.inc(out)
        return x


class CrissCrossAttention(nn.Module):
    """ Pixel-wise attention module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x
        return out


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //
            8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
            kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height
            ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class DeformConvFunction(Function):

    def __init__(self, stride, padding, dilation, deformable_groups=1,
        im2col_step=64):
        super(DeformConvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)
        output = input.new(*self._output_size(input, weight))
        self.bufs_ = [input.new(), input.new()]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(input, torch.autograd.Variable):
                if not isinstance(input.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            elif not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert input.shape[0
                ] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv.deform_conv_forward_cuda(input, weight, offset,
                output, self.bufs_[0], self.bufs_[1], weight.size(3),
                weight.size(2), self.stride[1], self.stride[0], self.
                padding[1], self.padding[0], self.dilation[1], self.
                dilation[0], self.deformable_groups, cur_im2col_step)
        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if isinstance(grad_output, torch.autograd.Variable):
                if not isinstance(grad_output.data, torch.cuda.FloatTensor):
                    raise NotImplementedError
            elif not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert input.shape[0
                ] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if self.needs_input_grad[0] or self.needs_input_grad[1]:
                grad_input = input.new(*input.size()).zero_()
                grad_offset = offset.new(*offset.size()).zero_()
                deform_conv.deform_conv_backward_input_cuda(input, offset,
                    grad_output, grad_input, grad_offset, weight, self.
                    bufs_[0], weight.size(3), weight.size(2), self.stride[1
                    ], self.stride[0], self.padding[1], self.padding[0],
                    self.dilation[1], self.dilation[0], self.
                    deformable_groups, cur_im2col_step)
            if self.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                deform_conv.deform_conv_backward_parameters_cuda(input,
                    offset, grad_output, grad_weight, self.bufs_[0], self.
                    bufs_[1], weight.size(3), weight.size(2), self.stride[1
                    ], self.stride[0], self.padding[1], self.padding[0],
                    self.dilation[1], self.dilation[0], self.
                    deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be {})'.
                format('x'.join(map(str, output_size))))
        return output_size


def deform_conv_function(input, offset, weight, stride=1, padding=0,
    dilation=1, deform_groups=1, im2col_step=64):
    if input is not None and input.dim() != 4:
        raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'
            .format(input.dim()))
    f = DeformConvFunction(_pair(stride), _pair(padding), _pair(dilation),
        deform_groups, im2col_step)
    return f(input, offset, weight)


class DeformConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, num_deformable_groups=1):
        super(DeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.num_deformable_groups = num_deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv_function(input, offset, self.weight, self.stride,
            self.padding, self.dilation, self.num_deformable_groups)


class ModulatedDeformConvFunction(Function):

    def __init__(self, stride, padding, dilation=1, deformable_groups=1):
        super(ModulatedDeformConvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, mask, weight, bias):
        if not input.is_cuda:
            raise NotImplementedError
        if (weight.requires_grad or mask.requires_grad or offset.
            requires_grad or input.requires_grad):
            self.save_for_backward(input, offset, mask, weight, bias)
        output = input.new(*self._infer_shape(input, weight))
        self._bufs = [input.new(), input.new()]
        _backend.modulated_deform_conv_cuda_forward(input, weight, bias,
            self._bufs[0], offset, mask, output, self._bufs[1], weight.
            shape[2], weight.shape[3], self.stride, self.stride, self.
            padding, self.padding, self.dilation, self.dilation, self.
            deformable_groups)
        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = self.saved_tensors
        grad_input = input.new(*input.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()
        grad_mask = mask.new(*mask.size()).zero_()
        grad_weight = weight.new(*weight.size()).zero_()
        grad_bias = bias.new(*bias.size()).zero_()
        _backend.modulated_deform_conv_cuda_backward(input, weight, bias,
            self._bufs[0], offset, mask, self._bufs[1], grad_input,
            grad_weight, grad_bias, grad_offset, grad_mask, grad_output,
            weight.shape[2], weight.shape[3], self.stride, self.stride,
            self.padding, self.padding, self.dilation, self.dilation, self.
            deformable_groups)
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias

    def _infer_shape(self, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * self.padding - (self.dilation * (
            kernel_h - 1) + 1)) // self.stride + 1
        width_out = (width + 2 * self.padding - (self.dilation * (kernel_w -
            1) + 1)) // self.stride + 1
        return n, channels_out, height_out, width_out


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation=1, deformable_groups=1, no_bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.no_bias = no_bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()
        if self.no_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = ModulatedDeformConvFunction(self.stride, self.padding, self.
            dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class DeformRoIPoolingFunction(Function):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPoolingFunction, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        assert self.trans_std >= 0.0 and self.trans_std <= 1.0

    def forward(self, data, rois, offset):
        if not data.is_cuda:
            raise NotImplementedError
        output = data.new(*self._infer_shape(data, rois))
        output_count = data.new(*self._infer_shape(data, rois))
        _backend.deform_psroi_pooling_cuda_forward(data, rois, offset,
            output, output_count, self.no_trans, self.spatial_scale, self.
            output_dim, self.group_size, self.pooled_size, self.part_size,
            self.sample_per_part, self.trans_std)
        self.data = data
        self.rois = rois
        self.offset = offset
        self.output_count = output_count
        return output

    def backward(self, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data = self.data
        rois = self.rois
        offset = self.offset
        output_count = self.output_count
        grad_input = data.new(*data.size()).zero_()
        grad_offset = offset.new(*offset.size()).zero_()
        _backend.deform_psroi_pooling_cuda_backward(grad_output, data, rois,
            offset, output_count, grad_input, grad_offset, self.no_trans,
            self.spatial_scale, self.output_dim, self.group_size, self.
            pooled_size, self.part_size, self.sample_per_part, self.trans_std)
        return grad_input, torch.zeros(rois.shape).cuda(), grad_offset

    def _infer_shape(self, data, rois):
        c = data.shape[1]
        n = rois.shape[0]
        return n, self.output_dim, self.pooled_size, self.pooled_size


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans,
        group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        self.func = DeformRoIPoolingFunction(self.spatial_scale, self.
            pooled_size, self.output_dim, self.no_trans, self.group_size,
            self.part_size, self.sample_per_part, self.trans_std)

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new()
        return self.func(data, rois, offset)


class FilterResponseNormalization(nn.Module):

    def __init__(self, beta, gamma, tau, eps=1e-06):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """
        super(FilterResponseNormalization, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = torch.Tensor([eps])

    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]
            ) == (c, c, c)
        nu2 = torch.mean(x.pow(2), (2, 3), keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)


ACT_ELU = 'elu'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_RELU = 'relu'


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var,
            self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope,
                inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var,
            self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope,
                inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = (
            '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
            )
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class SingleGPU(nn.Module):

    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, input):
        return self.module(input)


def np_gaussian_2d(width, sigma=-1):
    """Truncated 2D Gaussian filter"""
    assert width % 2 == 1
    if sigma <= 0:
        sigma = float(width) / 4
    r = np.arange(-(width // 2), width // 2 + 1, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * r * r / (sigma * sigma))
    gaussian_2d = gaussian_1d.reshape(-1, 1) * gaussian_1d
    gaussian_2d /= gaussian_2d.sum()
    return gaussian_2d


class _PacConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, transposed, output_padding, bias, pool_only,
        kernel_type, smooth_kernel_type, channel_wise, normalize_kernel,
        shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([(k % 2 != 1) for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[
            smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError(
                'smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([(p > d * (k - 1) / 2) for p, d, k in zip(padding, dilation,
            kernel_size)]):
            pass
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)
                        ] = 0
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels,
                    out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels,
                    in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'
                        .format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format
                (kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(
                smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([0.25, 0.5, 0.25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(
                    1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0
                ).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) *
                smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(
                    1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0
                ).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'
                .format(smooth_kernel_type))
        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1.0 / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.
                    kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size,
                    effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.
                    kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) +
                        tuple(range(s, 0, -1)) + (0,) * p) / s).reshape((-1
                        ,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple((k - j - 1) // s + j // s + 
                            1.0 for j in range(k))).reshape((-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[(c), (c), :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0
                    ] == self.kernel_size[1
                    ] and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[
                    0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[(0), (0), :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(
                        n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data
                        ) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'
                    .format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter
            ):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel,
            Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.
                smooth_kernel.shape))

    def extra_repr(self):
        s = (
            '{in_channels}, {out_channels}, kernel_size={kernel_size}, kernel_type={kernel_type}'
            )
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)


def _ceil_pad_factor(sizes, factor):
    offs = tuple((factor - sz % factor) % factor for sz in sizes)
    pad = tuple((off + 1) // 2 for off in offs)
    return pad


class PacCRF(nn.Module):
    """
    Args:
        channels (int): number of categories.
        num_steps (int): number of mean-field update steps.
        final_output (str): 'log_softmax' | 'softmax' | 'log_Q'. Default: 'log_Q'
        perturbed_init (bool): whether to perturb initialization. Default: True
        native_impl (bool): Default: False
        fixed_weighting (bool): whether to use fixed weighting for unary/pairwise terms. Default: False
        unary_weight (float): Default: 1.0
        pairwise_kernels (dict or list): pairwise kernels, see add_pairwise_kernel() for details. Default: None
    """

    def __init__(self, channels, num_steps, final_output='log_Q',
        perturbed_init=True, native_impl=False, fixed_weighting=False,
        unary_weight=1.0, pairwise_kernels=None):
        super(PacCRF, self).__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.final_output = final_output
        self.perturbed_init = perturbed_init
        self.native_impl = native_impl
        self.fixed_weighting = fixed_weighting
        self.init_unary_weight = unary_weight
        self.messengers = nn.ModuleList()
        self.compat = nn.ModuleList()
        self.init_pairwise_weights = []
        self.pairwise_weights = nn.ParameterList()
        self._use_pairwise_weights = []
        self.unary_weight = (unary_weight if self.fixed_weighting else nn.
            Parameter(th.tensor(float(unary_weight))))
        self.blur = []
        self.pairwise_repr = []
        if pairwise_kernels is not None:
            if type(pairwise_kernels) == dict:
                self.add_pairwise_kernel(**pairwise_kernels)
            else:
                for k in pairwise_kernels:
                    self.add_pairwise_kernel(**k)

    def reset_parameters(self, pairwise_idx=None):
        if pairwise_idx is None:
            idxs = range(len(self.messengers))
            if not self.fixed_weighting:
                self.unary_weight.data.fill_(self.init_unary_weight)
        else:
            idxs = [pairwise_idx]
        for i in idxs:
            self.messengers[i].reset_parameters()
            if isinstance(self.messengers[i], nn.Conv2d):
                pass
            if self.compat[i] is not None:
                self.compat[i].weight.data[:, :, (0), (0)] = 1.0 - th.eye(self
                    .channels, dtype=th.float32)
                if self.perturbed_init:
                    perturb_range = 0.001
                    self.compat[i].weight.data.add_((th.rand_like(self.
                        compat[i].weight.data) - 0.5) * perturb_range)
            self.pairwise_weights[i].data = th.ones_like(self.
                pairwise_weights[i]) * self.init_pairwise_weights[i]

    def extra_repr(self):
        s = (
            'categories={channels}, num_steps={num_steps}, final_output={final_output}'
            )
        if self.perturbed_init:
            s += ', perturbed_init=True'
        if self.fixed_weighting:
            s += ', fixed_weighting=True'
        if self.pairwise_repr:
            s += ', pairwise_kernels=({})'.format(', '.join(self.pairwise_repr)
                )
        return s.format(**self.__dict__)

    def add_pairwise_kernel(self, kernel_size=3, dilation=1, blur=1,
        compat_type='4d', spatial_filter=True, pairwise_weight=1.0):
        assert kernel_size % 2 == 1
        self.pairwise_repr.append('{}{}_{}_{}_{}'.format('0d' if 
            compat_type == 'potts' else compat_type, 's' if spatial_filter else
            '', kernel_size, dilation, blur))
        if compat_type == 'potts':
            pairwise_weight *= -1.0
        if (compat_type == 'potts' and not spatial_filter and not self.
            fixed_weighting):
            self._use_pairwise_weights.append(True)
        else:
            self._use_pairwise_weights.append(False)
        self.pairwise_weights.append(nn.Parameter(th.tensor(pairwise_weight,
            dtype=th.float32)))
        self.init_pairwise_weights.append(pairwise_weight)
        self.blur.append(blur)
        self.compat.append(nn.Conv2d(self.channels, self.channels,
            kernel_size=1, bias=False) if compat_type == '2d' else None)
        pad = int(kernel_size // 2) * dilation
        if compat_type == 'na':
            messenger = nn.Conv2d(self.channels, self.channels, kernel_size,
                padding=pad, dilation=dilation, bias=False)
        elif compat_type == '4d':
            messenger = pac.PacConv2d(self.channels, self.channels,
                kernel_size, padding=pad, dilation=dilation, bias=False,
                shared_filters=False, native_impl=self.native_impl, filler=
                'crf_perturbed' if self.perturbed_init else 'crf')
        elif spatial_filter:
            messenger = pac.PacConv2d(self.channels, self.channels,
                kernel_size, padding=pad, dilation=dilation, bias=False,
                shared_filters=True, native_impl=self.native_impl, filler=
                'crf_perturbed' if self.perturbed_init else 'crf')
        else:
            messenger = pac.PacConv2d(self.channels, self.channels,
                kernel_size, padding=pad, dilation=dilation, bias=False,
                shared_filters=True, native_impl=self.native_impl, filler=
                'crf_pool')
        self.messengers.append(messenger)
        self.reset_parameters(-1)

    def num_pairwise_kernels(self):
        return len(self.messengers)

    def forward(self, unary, edge_feat, edge_kernel=None, logQ=None):
        n_kernels = len(self.messengers)
        edge_kernel = [edge_kernel] * n_kernels if isinstance(edge_kernel,
            th.Tensor) else edge_kernel
        if edge_kernel is None:
            edge_kernel = [None] * n_kernels
            _shared = isinstance(edge_feat, th.Tensor)
            if _shared:
                edge_feat = {(1): edge_feat}
            for i in range(n_kernels):
                if isinstance(self.messengers[i], nn.Conv2d):
                    continue
                if _shared and self.blur[i] in edge_feat:
                    feat = edge_feat[self.blur[i]]
                elif self.blur[i] == 1:
                    feat = edge_feat[i]
                else:
                    feat = edge_feat[1] if _shared else edge_feat[i]
                    pad = _ceil_pad_factor(feat.shape[2:], self.blur[i])
                    feat = F.avg_pool2d(feat, kernel_size=self.blur[i],
                        padding=pad, count_include_pad=False)
                    if _shared:
                        edge_feat[self.blur[i]] = feat
                edge_kernel[i], _ = self.messengers[i].compute_kernel(feat)
                del feat
            del edge_feat
        if logQ is None:
            logQ = unary
        for step in range(self.num_steps):
            Q = F.softmax(logQ, dim=1)
            Q_blur = {(1): Q}
            logQ = unary * self.unary_weight
            for i in range(n_kernels):
                pad = _ceil_pad_factor(Q.shape[2:], self.blur[i])
                if self.blur[i] not in Q_blur:
                    Q_blur[self.blur[i]] = F.avg_pool2d(Q, kernel_size=self
                        .blur[i], padding=pad, count_include_pad=False)
                if isinstance(self.messengers[i], nn.Conv2d):
                    msg = self.messengers[i](Q_blur[self.blur[i]])
                else:
                    msg = self.messengers[i](Q_blur[self.blur[i]], None,
                        edge_kernel[i])
                if self.compat[i] is not None:
                    msg = self.compat[i](msg)
                if self.blur[i] > 1:
                    msg = F.interpolate(msg, scale_factor=self.blur[i],
                        mode='bilinear', align_corners=False)
                    msg = msg[:, :, pad[0]:pad[0] + unary.shape[2], pad[1]:
                        pad[1] + unary.shape[3]].contiguous()
                pw = self.pairwise_weights[i] if self._use_pairwise_weights[i
                    ] else self.init_pairwise_weights[i]
                logQ = logQ - msg * pw
        if self.final_output == 'softmax':
            out = F.softmax(logQ, dim=1)
        elif self.final_output == 'log_softmax':
            out = F.log_softmax(logQ, dim=1)
        elif self.final_output == 'log_Q':
            out = logQ
        else:
            raise ValueError('Unknown value for final_output: {}'.format(
                self.final_output))
        return out


class PacCRFLoose(nn.Module):

    def __init__(self, channels, num_steps, final_output='log_Q',
        perturbed_init=True, native_impl=False, fixed_weighting=False,
        unary_weight=1.0, pairwise_kernels=None):
        super(PacCRFLoose, self).__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.final_output = final_output
        self.steps = nn.ModuleList()
        for i in range(num_steps):
            self.steps.append(PacCRF(channels, 1, 'log_Q', perturbed_init,
                native_impl, fixed_weighting, unary_weight, pairwise_kernels))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_steps):
            self.steps[i].reset_parameters()

    def extra_repr(self):
        s = (
            'categories={channels}, num_steps={num_steps}, final_output={final_output}'
            )
        return s.format(**self.__dict__)

    def add_pairwise_kernel(self, kernel_size=3, dilation=1, blur=1,
        compat_type='4d', spatial_filter=True, pairwise_weight=1.0):
        for i in range(self.num_steps):
            self.steps[i].add_pairwise_kernel(kernel_size, dilation, blur,
                compat_type, spatial_filter, pairwise_weight)

    def num_pairwise_kernels(self):
        return self.steps[0].num_pairwise_kernels()

    def forward(self, unary, edge_feat, edge_kernel=None):
        n_kernels = self.num_pairwise_kernels()
        edge_kernel = [edge_kernel] * n_kernels if isinstance(edge_kernel,
            th.Tensor) else edge_kernel
        blurs = self.steps[0].blur
        if edge_kernel is None:
            edge_kernel = [None] * n_kernels
            _shared = isinstance(edge_feat, th.Tensor)
            if _shared:
                edge_feat = {(1): edge_feat}
            for i in range(n_kernels):
                if _shared and blurs[i] in edge_feat:
                    feat = edge_feat[blurs[i]]
                elif blurs[i] == 1:
                    feat = edge_feat[i]
                else:
                    feat = edge_feat[1] if _shared else edge_feat[i]
                    pad = _ceil_pad_factor(feat.shape[2:], blurs[i])
                    feat = F.avg_pool2d(feat, kernel_size=blurs[i], padding
                        =pad, count_include_pad=False)
                    if _shared:
                        edge_feat[blurs[i]] = feat
                edge_kernel[i], _ = self.steps[0].messengers[i].compute_kernel(
                    feat)
                del feat
            del edge_feat
        logQ = unary
        for step in self.steps:
            logQ = step(unary, None, edge_kernel, logQ)
        if self.final_output == 'softmax':
            out = F.softmax(logQ, dim=1)
        elif self.final_output == 'log_softmax':
            out = F.log_softmax(logQ, dim=1)
        elif self.final_output == 'log_Q':
            out = logQ
        else:
            raise ValueError('Unknown value for final_output: {}'.format(
                self.final_output))
        return out


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.
                format(args[0].__class__.__name__, func.__name__, args[0].
                datatype))
        return func(*args, **kwargs)
    return wrapper


class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()

    @assert_tensor_type
    def numel(self):
        return self.data.numel()


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                    [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception('Unknown type {}.'.format(type(output)))


class Scatter(object):

    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            streams = [_get_stream(device) for device in target_gpus]
        outputs = scatter(input, target_gpus, streams)
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)
        return tuple(outputs)


def scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


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
        Amit Agrawal. "Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0,
        gather_=True):
        super(DataParallelModel, self).__init__(module, device_ids,
            output_device, dim)
        self.gather_ = gather_

    def gather(self, outputs, output_device):
        if self.gather_:
            return gather(outputs, output_device, dim=self.dim)
        return outputs

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


torch_ver = torch.__version__[:3]


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    if torch_ver != '0.3':
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != '0.3':
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0]
            )
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.
    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.
    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelCriterion, self).__init__(module, device_ids,
            output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, inputs, *targets, gathered=True, **kwargs):
        if gathered:
            if isinstance(inputs, (list, tuple)):
                inputs, _ = self.scatter(inputs, kwargs, self.device_ids)
            else:
                inputs, _ = self.scatter([inputs], kwargs, self.device_ids)
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs[0], *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


class MMDistributedDataParallel(nn.Module):

    def __init__(self, module, dim=0, broadcast_buffers=True, bucket_cap_mb=25
        ):
        super(MMDistributedDataParallel, self).__init__()
        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers
        self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024
        self._sync_params()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(tensors, _unflatten_dense_tensors(
                flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states, self.
                broadcast_bucket_size)
        if self.broadcast_buffers:
            buffers = [b.data for b in self.module._all_buffers()]
            if len(buffers) > 0:
                self._dist_broadcast_coalesced(buffers, self.
                    broadcast_bucket_size)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.current_device()])
        return self.module(*inputs[0], **kwargs[0])


class SwitchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.997,
        using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(
                input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)
        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class SwitchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.997,
        using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1,
                num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1)
                )
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1
                ] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[
                2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SwitchNorm3d(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.997,
        using_moving_average=True, using_bn=True, last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1,
                num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1)
                )
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)
        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1
                ] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[
                2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


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


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


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
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
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
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
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


build_path = '/tmp/bulid/syncbn'


class _batchnormtrain(Function):

    @staticmethod
    def forward(ctx, input, mean, std, gamma, beta):
        ctx.save_for_backward(input, mean, std, gamma, beta)
        if input.is_cuda:
            output = syncbn.batchnorm_forward(input, mean, std, gamma, beta)
        else:
            raise NotImplemented
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, mean, std, gamma, beta = ctx.saved_variables
        if gradOutput.is_cuda:
            gradInput, gradMean, gradStd, gradGamma, gradBeta = (syncbn.
                batchnorm_backward(gradOutput, input, mean, std, gamma,
                beta, True))
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
            xsum, xsqusum = syncbn.sumsquare_forward(input)
        else:
            raise NotImplemented
        return xsum, xsqusum

    @staticmethod
    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_variables
        if input.is_cuda:
            gradInput = syncbn.sumsquare_backward(input, gradSum, gradSquare)
        else:
            raise NotImplemented
        return gradInput


def sum_square(input):
    """Calculate sum of elements and sum of squares for Batch Normalization"""
    return _sum_square.apply(input)


class _SyncBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(_SyncBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not self.training:
            return batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input_shape[0], self.num_features, -1)
        N = input.size(0) * input.size(2)
        xsum, xsqsum = sum_square(input)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(xsum,
                xsqsum, N))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(xsum,
                xsqsum, N))
        return batchnormtrain(input, mean, 1.0 / inv_std, self.weight, self
            .bias).view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.data
        return mean, (bias_var + self.eps) ** -0.5


class WeightedFSOhemCELoss(nn.Module):

    def __init__(self, configer):
        super().__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params'
            ) and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction'
                ]

    def forward(self, predict, target, min_kept=1, weight=None,
        ignore_index=-1, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1)[mask].contiguous(
            ).sort()
        min_threshold = sort_prob[min(min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matrix = F.cross_entropy(predict, target, weight=weight,
            ignore_index=ignore_index, reduction='none').contiguous().view(-1)
        sort_loss_matrix = loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class FSCELoss(nn.Module):

    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        if self.configer.exists('loss', 'params'
            ) and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight)
        reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params'
            ) and 'ce_reduction' in self.configer.get('loss', 'params'):
            reduction = self.configer.get('loss', 'params')['ce_reduction']
        ignore_index = -1
        if self.configer.exists('loss', 'params'
            ) and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=
            ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)
            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size
                        (2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size
                        (2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs
                .size(3)))
            loss = self.ce_loss(inputs, target)
        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class FSOhemCELoss(nn.Module):

    def __init__(self, configer):
        super(FSOhemCELoss, self).__init__()
        self.configer = configer
        self.thresh = self.configer.get('loss', 'params')['ohem_thresh']
        self.min_kept = max(1, self.configer.get('loss', 'params')[
            'ohem_minkeep'])
        weight = None
        if self.configer.exists('loss', 'params'
            ) and 'ce_weight' in self.configer.get('loss', 'params'):
            weight = self.configer.get('loss', 'params')['ce_weight']
            weight = torch.FloatTensor(weight)
        self.reduction = 'elementwise_mean'
        if self.configer.exists('loss', 'params'
            ) and 'ce_reduction' in self.configer.get('loss', 'params'):
            self.reduction = self.configer.get('loss', 'params')['ce_reduction'
                ]
        ignore_index = -1
        if self.configer.exists('loss', 'params'
            ) and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        self.ignore_label = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=
            ignore_index, reduction='none')

    def forward(self, predict, target, **kwargs):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1) != self.ignore_label
        sort_prob, sort_indices = prob.contiguous().view(-1)[mask].contiguous(
            ).sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1)
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class FSAuxOhemCELoss(nn.Module):

    def __init__(self, configer=None):
        super(FSAuxOhemCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        if self.configer.get('loss', 'loss_type') == 'fs_auxohemce_loss':
            self.ohem_ce_loss = FSOhemCELoss(self.configer)
        else:
            assert self.configer.get('loss', 'loss_type'
                ) == 'fs_auxslowohemce_loss'
            self.ohem_ce_loss = FSSlowOhemCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ohem_ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'
            ] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'
            ] * aux_loss
        return loss


class FSAuxCELoss(nn.Module):

    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)

    def forward(self, inputs, targets, **kwargs):
        aux_out, seg_out = inputs
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        loss = self.configer.get('network', 'loss_weights')['seg_loss'
            ] * seg_loss
        loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'
            ] * aux_loss
        return loss


class ModuleHelper(object):

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type == 'torchbn':
            return nn.Sequential(nn.BatchNorm2d(num_features, **kwargs), nn
                .ReLU())
        elif bn_type == 'torchsyncbn':
            return nn.Sequential(nn.SyncBatchNorm(num_features, **kwargs),
                nn.ReLU())
        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return nn.Sequential(BatchNorm2d(num_features, **kwargs), nn.ReLU()
                )
        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return nn.Sequential(SwitchNorm2d(num_features, **kwargs), nn.
                ReLU())
        elif bn_type == 'gn':
            return nn.Sequential(nn.GroupNorm(num_groups=8, num_channels=
                num_features, **kwargs), nn.ReLU())
        elif bn_type == 'fn':
            Log.error('Not support Filter-Response-Normalization: {}.'.
                format(bn_type))
            exit(1)
        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver == '1.0':
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                return InPlaceABNSync(num_features, **kwargs)
        else:
            Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def BatchNorm2d(bn_type='torch', ret_cls=False):
        if bn_type == 'torchbn':
            return nn.BatchNorm2d
        elif bn_type == 'torchsyncbn':
            return nn.SyncBatchNorm
        elif bn_type == 'syncbn':
            from lib.extensions.syncbn.module import BatchNorm2d
            return BatchNorm2d
        elif bn_type == 'sn':
            from lib.extensions.switchablenorms.switchable_norm import SwitchNorm2d
            return SwitchNorm2d
        elif bn_type == 'gn':
            return functools.partial(nn.GroupNorm, num_groups=32)
        elif bn_type == 'inplace_abn':
            torch_ver = torch.__version__[:3]
            if torch_ver == '0.4':
                from lib.extensions.inplace_abn.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync
                return functools.partial(InPlaceABNSync, activation='none')
            elif torch_ver == '1.0':
                from lib.extensions.inplace_abn_1.bn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync
                return functools.partial(InPlaceABNSync, activation='none')
            elif torch_ver == '1.2':
                from inplace_abn import InPlaceABNSync
                if ret_cls:
                    return InPlaceABNSync
                return functools.partial(InPlaceABNSync, activation='identity')
        else:
            Log.error('Not support BN type: {}.'.format(bn_type))
            exit(1)

    @staticmethod
    def load_model(model, pretrained=None, all_match=True, network='resnet101'
        ):
        if pretrained is None:
            return model
        if all_match:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = model.state_dict()
            load_dict = dict()
            for k, v in pretrained_dict.items():
                if 'resinit.{}'.format(k) in model_dict:
                    load_dict['resinit.{}'.format(k)] = v
                else:
                    load_dict[k] = v
            model.load_state_dict(load_dict)
        else:
            Log.info('Loading pretrained model:{}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            if network == 'wide_resnet':
                pretrained_dict = pretrained_dict['state_dict']
            model_dict = model.state_dict()
            if network == 'hrnet_plus':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in
                    model_dict.keys()}
            elif network == 'hrnet' or network == 'xception' or network == 'resnest':
                load_dict = {k: v for k, v in pretrained_dict.items() if k in
                    model_dict.keys()}
                Log.info('Missing keys: {}'.format(list(set(model_dict) -
                    set(load_dict))))
            elif network == 'dcnet' or network == 'resnext':
                load_dict = dict()
                for k, v in pretrained_dict.items():
                    if 'resinit.{}'.format(k) in model_dict:
                        load_dict['resinit.{}'.format(k)] = v
                    elif k in model_dict:
                        load_dict[k] = v
                    else:
                        pass
            elif network == 'wide_resnet':
                load_dict = {'.'.join(k.split('.')[1:]): v for k, v in
                    pretrained_dict.items() if '.'.join(k.split('.')[1:]) in
                    model_dict}
            else:
                load_dict = {'.'.join(k.split('.')[1:]): v for k, v in
                    pretrained_dict.items() if '.'.join(k.split('.')[1:]) in
                    model_dict}
            Log.info('Matched Keys: {}'.format(load_dict.keys()))
            model_dict.update(load_dict)
            model.load_state_dict(model_dict)
        return model

    @staticmethod
    def load_url(url, map_location=None):
        model_dir = os.path.join('~', '.PyTorchCV', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        filename = url.split('/')[-1]
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            Log.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        Log.info('Loading pretrained model:{}'.format(cached_file))
        return torch.load(cached_file, map_location=map_location)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module, mode='fan_in', nonlinearity='leaky_relu', bias
        =0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity
                =nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=
                nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type
        =None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes,
            momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes,
            momentum=bn_momentum)
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
        out = out + residual
        out = self.relu_in(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type
        =None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes,
            momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes,
            momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * 4,
            momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
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
        out = out + residual
        out = self.relu_in(out)
        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
        num_channels, fuse_method, multi_scale_output=True, bn_type=None,
        bn_momentum=0.1):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks,
            num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks,
            num_blocks, num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.fuse_layers = self._make_fuse_layers(bn_type=bn_type,
            bn_momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            Log.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            Log.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            Log.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks,
        num_channels, stride=1, bn_type=None, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[
            branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[
                branch_index], num_channels[branch_index] * block.expansion,
                kernel_size=1, stride=stride, bias=False), ModuleHelper.
                BatchNorm2d(bn_type=bn_type)(num_channels[branch_index] *
                block.expansion, momentum=bn_momentum))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels
            [branch_index], stride, downsample, bn_type=bn_type,
            bn_momentum=bn_momentum))
        self.num_inchannels[branch_index] = num_channels[branch_index
            ] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                num_channels[branch_index], bn_type=bn_type, bn_momentum=
                bn_momentum))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels,
        bn_type, bn_momentum=0.1):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks,
                num_channels, bn_type=bn_type, bn_momentum=bn_momentum))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, bn_type, bn_momentum=0.1):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(
                        num_inchannels[j], num_inchannels[i], 1, 1, 0, bias
                        =False), ModuleHelper.BatchNorm2d(bn_type=bn_type)(
                        num_inchannels[i], momentum=bn_momentum)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), ModuleHelper.
                                BatchNorm2d(bn_type=bn_type)(
                                num_outchannels_conv3x3, momentum=bn_momentum))
                                )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(
                                num_inchannels[j], num_outchannels_conv3x3,
                                3, 2, 1, bias=False), ModuleHelper.
                                BatchNorm2d(bn_type=bn_type)(
                                num_outchannels_conv3x3, momentum=
                                bn_momentum), nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output], mode='bilinear',
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, bn_type, bn_momentum, **kwargs):
        self.inplanes = 64
        super(HighResolutionNet, self).__init__()
        if os.environ.get('full_res_stem'):
            Log.info('using full-resolution stem with stride=1')
            stem_stride = 1
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride,
                padding=1, bias=False)
            self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(64,
                momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=False)
            self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, bn_type=
                bn_type, bn_momentum=bn_momentum)
        else:
            stem_stride = 2
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride,
                padding=1, bias=False)
            self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(64,
                momentum=bn_momentum)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=
                stem_stride, padding=1, bias=False)
            self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(64,
                momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=False)
            self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, bn_type=
                bn_type, bn_momentum=bn_momentum)
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels,
            bn_type=bn_type, bn_momentum=bn_momentum)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
            num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
            num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
            num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(
            len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
            num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg,
            num_channels, multi_scale_output=True, bn_type=bn_type,
            bn_momentum=bn_momentum)
        if os.environ.get('keep_imagenet_head'):
            self.incre_modules, self.downsamp_modules, self.final_layer = (self
                ._make_head(pre_stage_channels, bn_type=bn_type,
                bn_momentum=bn_momentum))

    def _make_head(self, pre_stage_channels, bn_type, bn_momentum):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        Log.info('pre_stage_channels: {}'.format(pre_stage_channels))
        Log.info('head_channels: {}'.format(head_channels))
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels,
                head_channels[i], 1, bn_type=bn_type, bn_momentum=bn_momentum)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(nn.Conv2d(in_channels=
                in_channels, out_channels=out_channels, kernel_size=3,
                stride=2, padding=1), ModuleHelper.BatchNorm2d(bn_type=
                bn_type)(out_channels, momentum=bn_momentum), nn.ReLU(
                inplace=False))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)
        final_layer = nn.Sequential(nn.Conv2d(in_channels=head_channels[3] *
            head_block.expansion, out_channels=2048, kernel_size=1, stride=
            1, padding=0), ModuleHelper.BatchNorm2d(bn_type=bn_type)(2048,
            momentum=bn_momentum), nn.ReLU(inplace=False))
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer,
        num_channels_cur_layer, bn_type, bn_momentum):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(
                        num_channels_pre_layer[i], num_channels_cur_layer[i
                        ], 3, 1, 1, bias=False), ModuleHelper.BatchNorm2d(
                        bn_type=bn_type)(num_channels_cur_layer[i],
                        momentum=bn_momentum), nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i
                        ] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels,
                        outchannels, 3, 2, 1, bias=False), ModuleHelper.
                        BatchNorm2d(bn_type=bn_type)(outchannels, momentum=
                        bn_momentum), nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1,
        bn_type=None, bn_momentum=0.1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.
                expansion, kernel_size=1, stride=stride, bias=False),
                ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * block.
                expansion, momentum=bn_momentum))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, bn_type=
            bn_type, bn_momentum=bn_momentum))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, bn_type=bn_type,
                bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=
        True, bn_type=None, bn_momentum=0.1):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block,
                num_blocks, num_inchannels, num_channels, fuse_method,
                reset_multi_scale_output, bn_type, bn_momentum))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        if os.environ.get('full_res_stem'):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        if os.environ.get('drop_stage4'):
            return y_list
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        if os.environ.get('keep_imagenet_head'):
            x_list = []
            y = self.incre_modules[0](y_list[0])
            x_list.append(y)
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]
                    ) + self.downsamp_modules[i](y)
                x_list.append(y)
            y = self.final_layer(y)
            del x_list[-1]
            x_list.append(y)
            return x_list
        return y_list


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, style='pytorch', with_cp=False, bn_type=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu_in(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, style='pytorch', with_cp=False, with_dcn=False,
        num_deformable_groups=1, dcn_offset_lr_mult=0.1,
        use_regular_conv_on_stride=False, use_modulated_dcn=False, bn_type=None
        ):
        """Bottleneck block.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        conv1_stride = 1
        conv2_stride = stride
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            conv1_stride, bias=False)
        self.with_dcn = with_dcn
        self.use_modulated_dcn = use_modulated_dcn
        if use_regular_conv_on_stride and stride > 1:
            self.with_dcn = False
        if self.with_dcn:
            None
            if use_modulated_dcn:
                self.conv_offset_mask = nn.Conv2d(planes, 
                    num_deformable_groups * 27, kernel_size=3, stride=
                    conv2_stride, padding=dilation, dilation=dilation)
                self.conv_offset_mask.lr_mult = dcn_offset_lr_mult
                self.conv_offset_mask.zero_init = True
                self.conv2 = ModulatedDeformConv(planes, planes, 3, stride=
                    conv2_stride, padding=dilation, dilation=dilation,
                    deformable_groups=num_deformable_groups, no_bias=True)
            else:
                self.conv2_offset = nn.Conv2d(planes, num_deformable_groups *
                    18, kernel_size=3, stride=conv2_stride, padding=
                    dilation, dilation=dilation)
                self.conv2_offset.lr_mult = dcn_offset_lr_mult
                self.conv2_offset.zero_init = True
                self.conv2 = DeformConv(planes, planes, (3, 3), stride=
                    conv2_stride, padding=dilation, dilation=dilation,
                    num_deformable_groups=num_deformable_groups)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=
                conv2_stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size
            =1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * self.
            expansion)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            if self.with_dcn:
                if self.use_modulated_dcn:
                    offset_mask = self.conv_offset_mask(out)
                    offset1, offset2, mask_raw = torch.chunk(offset_mask, 3,
                        dim=1)
                    offset = torch.cat((offset1, offset2), dim=1)
                    mask = torch.sigmoid(mask_raw)
                    out = self.conv2(out, offset, mask)
                else:
                    offset = self.conv2_offset(out)
                    dilation = self.conv2.dilation[0]
                    bias_w = torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1,
                        0, 1]]) * (dilation - 1)
                    bias_h = bias_w.permute(1, 0)
                    bias_w.requires_grad = False
                    bias_h.requires_grad = False
                    offset += torch.cat([bias_h.reshape(-1), bias_w.reshape
                        (-1)]).view(1, -1, 1, 1)
                    out = self.conv2(out, offset)
            else:
                out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu_in(out)
        return out


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1,
    style='pytorch', with_cp=False, with_dcn=False, dcn_offset_lr_mult=0.1,
    use_regular_conv_on_stride=False, use_modulated_dcn=False, bn_type=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.
            expansion, kernel_size=1, stride=stride, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * block.expansion)
            )
    layers = []
    layers.append(block(inplanes, planes, stride, dilation, downsample,
        style=style, with_cp=with_cp, with_dcn=with_dcn, dcn_offset_lr_mult
        =dcn_offset_lr_mult, use_regular_conv_on_stride=
        use_regular_conv_on_stride, use_modulated_dcn=use_modulated_dcn,
        bn_type=bn_type))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, 1, dilation, style=style,
            with_cp=with_cp, with_dcn=with_dcn, dcn_offset_lr_mult=
            dcn_offset_lr_mult, use_regular_conv_on_stride=
            use_regular_conv_on_stride, use_modulated_dcn=use_modulated_dcn,
            bn_type=bn_type))
    return nn.Sequential(*layers)


class DCNResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self, block, layers, deep_base=True, bn_type=None):
        super(DCNResNet, self).__init__()
        self.style = 'pytorch'
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,
                64, kernel_size=3, stride=2, padding=1, bias=False)), (
                'bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)), (
                'relu1', nn.ReLU(inplace=False)), ('conv2', nn.Conv2d(64, 
                64, kernel_size=3, stride=1, padding=1, bias=False)), (
                'bn2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)), (
                'relu2', nn.ReLU(inplace=False)), ('conv3', nn.Conv2d(64, 
                128, kernel_size=3, stride=1, padding=1, bias=False)), (
                'bn3', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.
                inplanes)), ('relu3', nn.ReLU(inplace=False))]))
        else:
            self.resinit = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,
                64, kernel_size=7, stride=2, padding=3, bias=False)), (
                'bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.
                inplanes)), ('relu1', nn.ReLU(inplace=False))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_res_layer(block, self.inplanes, 64, layers[0],
            style=self.style, with_dcn=False, use_modulated_dcn=False,
            bn_type=bn_type)
        self.layer2 = make_res_layer(block, 256, 128, layers[1], stride=2,
            style=self.style, with_dcn=False, use_modulated_dcn=False,
            bn_type=bn_type)
        self.layer3 = make_res_layer(block, 512, 256, layers[2], stride=2,
            style=self.style, with_dcn=True, use_modulated_dcn=False,
            bn_type=bn_type)
        self.layer4 = make_res_layer(block, 1024, 512, layers[3], stride=2,
            style=self.style, with_dcn=True, use_modulated_dcn=False,
            bn_type=bn_type)

    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
        padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2,
        reduction_factor=4, rectify=False, rectify_avg=False, bn_type=None,
        dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                **kwargs)
        self.use_bn = bn_type is not None
        self.bn0 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(channels * radix)
        self.relu = ReLU(inplace=False)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.
            cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1,
        cardinality=1, bottleneck_width=64, avd=False, avd_first=False,
        dilation=1, is_first=False, rectified_conv=False, rectify_avg=False,
        bn_type=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False
            )
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)
        if radix > 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=
                3, stride=stride, padding=dilation, dilation=dilation,
                groups=cardinality, bias=False, radix=radix, rectify=
                rectified_conv, rectify_avg=rectify_avg, bn_type=bn_type,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            self.conv2 = RFConv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False, average_mode=rectify_avg)
            self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(group_width)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False)
            self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(group_width)
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias
            =False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * 4)
        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu_in(out)
        return out


class ResNeSt(nn.Module):

    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=
        64, num_classes=1000, dilated=False, dilation=1, deep_stem=False,
        stem_width=64, avg_down=False, rectified_conv=False, rectify_avg=
        False, avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0,
        last_gamma=False, bn_type=None):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNeSt, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width,
                kernel_size=3, stride=2, padding=1, bias=False, **
                conv_kwargs), ModuleHelper.BatchNorm2d(bn_type=bn_type)(
                stem_width), nn.ReLU(inplace=False), conv_layer(stem_width,
                stem_width, kernel_size=3, stride=1, padding=1, bias=False,
                **conv_kwargs), ModuleHelper.BatchNorm2d(bn_type=bn_type)(
                stem_width), nn.ReLU(inplace=False), conv_layer(stem_width,
                stem_width * 2, kernel_size=3, stride=1, padding=1, bias=
                False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding
                =3, bias=False, **conv_kwargs)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_type=
            bn_type, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            bn_type=bn_type)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2, bn_type=bn_type, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4, bn_type=bn_type, dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                dilation=1, bn_type=bn_type, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=2, bn_type=bn_type, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                bn_type=bn_type, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                bn_type=bn_type, dropblock_prob=dropblock_prob)
        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, ModuleHelper.BatchNorm2d(bn_type=bn_type,
                ret_cls=True)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        bn_type=None, dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride,
                        stride=stride, ceil_mode=True, count_include_pad=False)
                        )
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                        ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.
                    expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.
                    expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(ModuleHelper.BatchNorm2d(bn_type=bn_type)(
                planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=
                downsample, radix=self.radix, cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width, avd=self.avd,
                avd_first=self.avd_first, dilation=1, is_first=is_first,
                rectified_conv=self.rectified_conv, rectify_avg=self.
                rectify_avg, bn_type=bn_type, dropblock_prob=dropblock_prob,
                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=
                downsample, radix=self.radix, cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width, avd=self.avd,
                avd_first=self.avd_first, dilation=2, is_first=is_first,
                rectified_conv=self.rectified_conv, rectify_avg=self.
                rectify_avg, bn_type=bn_type, dropblock_prob=dropblock_prob,
                last_gamma=self.last_gamma))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix,
                cardinality=self.cardinality, bottleneck_width=self.
                bottleneck_width, avd=self.avd, avd_first=self.avd_first,
                dilation=dilation, rectified_conv=self.rectified_conv,
                rectify_avg=self.rectify_avg, bn_type=bn_type,
                dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        return nn.Sequential(*layers)

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)
        return tuple_features


class NormalResnetBackbone(nn.Module):

    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()
        self.num_features = 2048
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)
        return tuple_features


class DilatedResnetBackbone(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnetBackbone, self).__init__()
        self.num_features = 2048
        from functools import partial
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate,
                    dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self.
                        _nostride_dilate, dilate=int(4 * r)))
        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate,
                    dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self.
                        _nostride_dilate, dilate=int(2 * r)))
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = 1, 1
                if m.kernel_size == (3, 3):
                    m.dilation = dilate // 2, dilate // 2
                    m.padding = dilate // 2, dilate // 2
            elif m.kernel_size == (3, 3):
                m.dilation = dilate, dilate
                m.padding = dilate, dilate

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)
        return tuple_features


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type
        =None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
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
        out = out + residual
        out = self.relu_in(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type
        =None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
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
        out = out + residual
        out = self.relu_in(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False,
        bn_type=None):
        super(ResNet, self).__init__()
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,
                64, kernel_size=3, stride=2, padding=1, bias=False)), (
                'bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)), (
                'relu1', nn.ReLU(inplace=False)), ('conv2', nn.Conv2d(64, 
                64, kernel_size=3, stride=1, padding=1, bias=False)), (
                'bn2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)), (
                'relu2', nn.ReLU(inplace=False)), ('conv3', nn.Conv2d(64, 
                128, kernel_size=3, stride=1, padding=1, bias=False)), (
                'bn3', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.
                inplanes)), ('relu3', nn.ReLU(inplace=False))]))
        else:
            self.resinit = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,
                64, kernel_size=7, stride=2, padding=3, bias=False)), (
                'bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.
                inplanes)), ('relu1', nn.ReLU(inplace=False))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_type=bn_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            bn_type=bn_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            bn_type=bn_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            bn_type=bn_type)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, ModuleHelper.BatchNorm2d(bn_type=bn_type,
                ret_cls=True)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, bn_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * block.
                expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            bn_type=bn_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_type=bn_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, bn_type=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu_in(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, bn_type=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * self.
            expansion)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu_in(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, groups=1, width_per_group=64, replace_stride_with_dilation=
        None, bn_type=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.resinit = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3,
            self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes
            )), ('relu1', nn.ReLU(inplace=False))]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_type=bn_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0], bn_type=bn_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1], bn_type=bn_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2], bn_type=bn_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
        bn_type=None):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), ModuleHelper.BatchNorm2d(bn_type=
                bn_type)(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, bn_type=bn_type))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation, bn_type
                =bn_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=
        1, bn_type=None, dropout=None):
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
        bn_type : callable
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
        self.bn1 = ModuleHelper.BNReLU(in_channels, bn_type=bn_type)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3,
                stride=stride, padding=dilation, bias=False, dilation=
                dilation)), ('bn2', ModuleHelper.BNReLU(channels[0],
                bn_type=bn_type)), ('conv2', nn.Conv2d(channels[0],
                channels[1], 3, stride=1, padding=dilation, bias=False,
                dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1,
                stride=stride, padding=0, bias=False)), ('bn2',
                ModuleHelper.BNReLU(channels[0], bn_type=bn_type)), (
                'conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1,
                padding=dilation, bias=False, groups=groups, dilation=
                dilation)), ('bn3', ModuleHelper.BNReLU(channels[1],
                bn_type=bn_type)), ('conv3', nn.Conv2d(channels[1],
                channels[2], 1, stride=1, padding=0, bias=False))]
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


class WiderResNetA2(nn.Module):

    def __init__(self, structure=[3, 3, 6, 3, 1, 1], bn_type=None, classes=
        0, dilation=True):
        """Wider ResNet with pre-activation (identity mapping) blocks

        This variant uses down-sampling by max-pooling in the first two blocks and by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        bn_type : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation
        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')
        self.mod1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3,
            stride=1, padding=1, bias=False))]))
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 
            1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1
                if mod_id == 4:
                    drop = None
                elif mod_id == 5:
                    drop = None
                else:
                    drop = None
                blocks.append(('block%d' % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id],
                    bn_type=bn_type, stride=stride, dilation=dil, dropout=
                    drop)))
                in_channels = channels[mod_id][-1]
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), nn.MaxPool2d(3,
                    stride=2, padding=1, ceil_mode=True))
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(
                OrderedDict(blocks)))
        self.bn_out = ModuleHelper.BNReLU(in_channels, bn_type=bn_type)

    def forward(self, img):
        tuple_features = list()
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        tuple_features.append(out)
        out = self.mod5(out)
        tuple_features.append(out)
        out = self.mod6(out)
        tuple_features.append(out)
        out = self.mod7(out)
        out = self.bn_out(out)
        tuple_features.append(out)
        return tuple_features


class ASP_OC_Module(nn.Module):

    def __init__(self, features, out_features=256, dilations=(12, 24, 36),
        bn_type=None, dropout=0.1):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=1, dilation=1, bias=True), ModuleHelper.
            BNReLU(out_features, bn_type=bn_type), BaseOC_Context_Module(
            in_channels=out_features, out_channels=out_features,
            key_channels=out_features // 2, value_channels=out_features // 
            2, dropout=0, sizes=[2], bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=1, padding=0, dilation=1, bias=False), ModuleHelper
            .BNReLU(out_features, bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=dilations[0], dilation=dilations[0],
            bias=False), ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=dilations[1], dilation=dilations[1],
            bias=False), ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features,
            kernel_size=3, padding=dilations[2], dilation=dilations[2],
            bias=False), ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(out_features * 5, 
            out_features * 2, kernel_size=1, padding=0, dilation=1, bias=
            False), ModuleHelper.BNReLU(out_features * 2, bn_type=bn_type),
            nn.Dropout2d(dropout))

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i],
                feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')
        output = self.conv_bn_dropout(out)
        return output


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1, bn_type=None):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), ModuleHelper.BNReLU(self.key_channels, bn_type=
            bn_type), nn.Conv2d(in_channels=self.key_channels, out_channels
            =self.key_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type))
        self.f_query = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), ModuleHelper.BNReLU(self.key_channels, bn_type=
            bn_type), nn.Conv2d(in_channels=self.key_channels, out_channels
            =self.key_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type))
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels
            =self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=
            self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode=
                'bilinear', align_corners=True)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
            key_channels, value_channels, out_channels, scale, bn_type)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, dropout, sizes=[1], bn_type=None):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels,
            in_channels, key_channels, value_channels, size, bn_type) for
            size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(2 * in_channels,
            out_channels, kernel_size=1, padding=0), ModuleHelper.BNReLU(
            out_channels, bn_type=bn_type), nn.Dropout2d(dropout))

    def _make_stage(self, in_channels, output_channels, key_channels,
        value_channels, size, bn_type):
        return SelfAttentionBlock2D(in_channels, key_channels,
            value_channels, output_channels, size, bn_type=bn_type)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, dropout=0, sizes=[1], bn_type=None):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels,
            out_channels, key_channels, value_channels, size, bn_type) for
            size in sizes])
        self.conv_bn_dropout = nn.Sequential(ModuleHelper.BNReLU(
            out_channels, bn_type=bn_type), nn.Dropout2d(dropout))

    def _make_stage(self, in_channels, output_channels, key_channels,
        value_channels, size, bn_type):
        return SelfAttentionBlock2D(in_channels, key_channels,
            value_channels, output_channels, size, bn_type=bn_type)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class Decoder_Module(nn.Module):

    def __init__(self, bn_type=None, inplane1=512, inplane2=256, outplane=128):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplane1, 256, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(256,
            bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(inplane2, 48, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(48,
            bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(304, outplane, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(
            outplane, bn_type=bn_type), nn.Conv2d(outplane, outplane,
            kernel_size=1, padding=0, dilation=1, bias=False), ModuleHelper
            .BNReLU(outplane, bn_type=bn_type))

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True
            )
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        return x


class CE2P_Decoder_Module(nn.Module):

    def __init__(self, num_classes, dropout=0, bn_type=None, inplane1=512,
        inplane2=256):
        super(CE2P_Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplane1, 256, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(256,
            bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(inplane2, 48, kernel_size=1,
            stride=1, padding=0, dilation=1, bias=False), ModuleHelper.
            BNReLU(48, bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(256,
            bn_type=bn_type), nn.Conv2d(256, 256, kernel_size=1, padding=0,
            dilation=1, bias=False), ModuleHelper.BNReLU(256, bn_type=
            bn_type), nn.Dropout2d(dropout))
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0,
            dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear',
            align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


class Edge_Module(nn.Module):

    def __init__(self, mid_fea, out_fea, bn_type=None, factor=1):
        super(Edge_Module, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(factor * 256, mid_fea,
            kernel_size=1, padding=0, dilation=1, bias=False), ModuleHelper
            .BNReLU(mid_fea, bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(factor * 512, mid_fea,
            kernel_size=1, padding=0, dilation=1, bias=False), ModuleHelper
            .BNReLU(mid_fea, bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(factor * 1024, mid_fea,
            kernel_size=1, padding=0, dilation=1, bias=False), ModuleHelper
            .BNReLU(mid_fea, bn_type=bn_type))
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1,
            dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding
            =0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)
        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear',
            align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear',
            align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear',
            align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear',
            align_corners=True)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge = self.conv5(edge)
        return edge, edge_fea


class SelfAttentionBlock2D(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(nn.Conv2d(self.in_channels, self.
            key_channels, kernel_size=1, bias=False), ModuleHelper.BNReLU(
            self.key_channels, bn_type=bn_type), nn.Conv2d(self.
            key_channels, self.key_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type))
        self.f_query = nn.Sequential(nn.Conv2d(self.in_channels, self.
            key_channels, kernel_size=1, bias=False), ModuleHelper.BNReLU(
            self.key_channels, bn_type=bn_type), nn.Conv2d(self.
            key_channels, self.key_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type))
        self.f_value = nn.Conv2d(self.in_channels, self.value_channels,
            kernel_size=1, bias=False)
        self.W = nn.Sequential(nn.Conv2d(self.value_channels, self.
            out_channels, kernel_size=1, bias=False), ModuleHelper.BNReLU(
            self.out_channels, bn_type=bn_type))

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)
        return context


class ISA_Block(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels, down_factor=[8, 8], bn_type=None):
        super(ISA_Block, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.long_range_sa = SelfAttentionBlock2D(in_channels, key_channels,
            value_channels, out_channels, bn_type=bn_type)
        self.short_range_sa = SelfAttentionBlock2D(out_channels,
            key_channels, value_channels, out_channels, bn_type=bn_type)

    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:
            feats = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, 
                pad_h - pad_h // 2))
        else:
            feats = x
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c,
            out_h, out_w)
        feats = self.long_range_sa(feats)
        c = self.out_channels
        feats = feats.view(n, dh, dw, c, out_h, out_w)
        feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw
            )
        feats = self.short_range_sa(feats)
        feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5
            )
        feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w //
                2 + w]
        return feats


class ISA_Module(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels, down_factors=[[8, 8]], dropout=0, bn_type=None):
        super(ISA_Module, self).__init__()
        assert isinstance(down_factors, (tuple, list))
        self.down_factors = down_factors
        self.stages = nn.ModuleList([ISA_Block(in_channels, key_channels,
            value_channels, out_channels, d, bn_type) for d in down_factors])
        concat_channels = in_channels + out_channels
        if len(self.down_factors) > 1:
            self.up_conv = nn.Sequential(nn.Conv2d(in_channels, len(self.
                down_factors) * out_channels, kernel_size=1, padding=0,
                bias=False), ModuleHelper.BNReLU(len(self.down_factors) *
                out_channels, bn_type=bn_type))
            concat_channels = out_channels * len(self.down_factors) * 2
        self.conv_bn = nn.Sequential(nn.Conv2d(concat_channels,
            out_channels, kernel_size=1, bias=False), ModuleHelper.BNReLU(
            out_channels, bn_type=bn_type), nn.Dropout2d(dropout))

    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        if len(self.down_factors) == 1:
            context = priors[0]
        else:
            context = torch.cat(priors, dim=1)
            x = self.up_conv(x)
        return self.conv_bn(torch.cat([x, context], dim=1))


class OffsetBlock(nn.Module):
    """
    This module takes relative offset as input and outputs feature at each position (coordinate + offset)
    """

    def __init__(self):
        super(OffsetBlock, self).__init__()
        self.coord_map = None
        self.norm_factor = None

    def _gen_coord_map(self, H, W):
        coord_vecs = [torch.arange(length, dtype=torch.float) for length in
            (H, W)]
        coord_h, coord_w = torch.meshgrid(coord_vecs)
        return coord_h, coord_w

    def forward(self, x, offset_map):
        n, c, h, w = x.size()
        if self.coord_map is None or self.coord_map[0].size(
            ) != offset_map.size()[2:]:
            self.coord_map = self._gen_coord_map(h, w)
            self.norm_factor = torch.FloatTensor([(w - 1) / 2, (h - 1) / 2])
        grid_h = offset_map[:, (0)] + self.coord_map[0]
        grid_w = offset_map[:, (1)] + self.coord_map[1]
        grid = torch.stack([grid_w, grid_h], dim=-1) / self.norm_factor - 1.0
        feats = F.grid_sample(x, grid, padding_mode='border')
        return feats


class OffsetModule(nn.Module):

    def __init__(self):
        super(OffsetModule, self).__init__()
        self.offset_block = OffsetBlock()

    def forward(self, x, offset):
        x_out = self.offset_block(x, offset)
        return x_out


def label_to_onehot(gt, num_classes, ignore_index=-1):
    """
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    """
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)
    return onehot.permute(0, 3, 1, 2)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.
                LongTensor), probs.size(1))
            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1
                ), gt_probs.size(2), gt_probs.size(3)
            gt_probs = gt_probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1)
            gt_probs = F.normalize(gt_probs, p=1, dim=2)
            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1
                ).unsqueeze(3)
            return ocr_context
        else:
            batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2
                ), probs.size(3)
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1)
            probs = F.softmax(self.scale * probs, dim=2)
            ocr_context = torch.matmul(probs, feats).permute(0, 2, 1
                ).unsqueeze(3)
            return ocr_context


class PyramidSpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scales=[1, 2, 4]):
        super(PyramidSpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scales = scales
        self.relu = nn.ReLU(inplace=True)

    def _compute_single_scale(self, feats, probs, dh, dw):
        batch_size, k, h, w = probs.size(0), probs.size(1), probs.size(2
            ), probs.size(3)
        c = feats.size(1)
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:
            feats = F.pad(feats, (pad_w // 2, pad_w - pad_w // 2, pad_h // 
                2, pad_h - pad_h // 2))
            probs = F.pad(probs, (pad_w // 2, pad_w - pad_w // 2, pad_h // 
                2, pad_h - pad_h // 2))
        feats = feats.view(batch_size, c, out_h, dh, out_w, dw).permute(0, 
            3, 5, 1, 2, 4)
        feats = feats.contiguous().view(batch_size, dh * dw, c, out_h, out_w)
        probs = probs.view(batch_size, k, out_h, dh, out_w, dw).permute(0, 
            3, 5, 1, 2, 4)
        probs = probs.contiguous().view(batch_size, dh * dw, k, out_h, out_w)
        feats = feats.view(batch_size, dh * dw, c, -1)
        probs = probs.view(batch_size, dh * dw, k, -1)
        feats = feats.permute(0, 1, 3, 2)
        probs = F.softmax(probs, dim=3)
        cc = torch.matmul(probs, feats).view(batch_size, -1, c)
        return cc.permute(0, 2, 1).unsqueeze(3)

    def forward(self, feats, probs):
        ocr_list = []
        for scale in self.scales:
            ocr_tmp = self._compute_single_scale(feats, probs, scale, scale)
            ocr_list.append(ocr_tmp)
        pyramid_ocr = torch.cat(ocr_list, 2)
        return pyramid_ocr


class _ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1, use_gt=False,
        use_bg=False, fetch_attention=False, bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), ModuleHelper.BNReLU(self.key_channels, bn_type=
            bn_type), nn.Conv2d(in_channels=self.key_channels, out_channels
            =self.key_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type))
        self.f_object = nn.Sequential(nn.Conv2d(in_channels=self.
            in_channels, out_channels=self.key_channels, kernel_size=1,
            stride=1, padding=0), ModuleHelper.BNReLU(self.key_channels,
            bn_type=bn_type), nn.Conv2d(in_channels=self.key_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type)
            )
        self.f_down = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type)
            )
        self.f_up = nn.Sequential(nn.Conv2d(in_channels=self.key_channels,
            out_channels=self.in_channels, kernel_size=1, stride=1, padding
            =0), ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type))

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.
                LongTensor), proxy.size(2) - 1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size,
                h * w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = self.key_channels ** -0.5 * sim_map
            sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode=
                'bilinear', align_corners=True)
        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.
                size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode=
                'bilinear', align_corners=True)
            return context, bg_context
        elif self.fetch_attention:
            return context, sim_map
        else:
            return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):

    def __init__(self, in_channels, key_channels, scale=1, use_gt=False,
        use_bg=False, fetch_attention=False, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
            key_channels, scale, use_gt, use_bg, fetch_attention, bn_type=
            bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """

    def __init__(self, in_channels, key_channels, out_channels, scale=1,
        dropout=0.1, use_gt=False, use_bg=False, use_oc=True,
        fetch_attention=False, bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
            key_channels, scale, use_gt, use_bg, fetch_attention, bn_type)
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(_in_channels,
            out_channels, kernel_size=1, padding=0), ModuleHelper.BNReLU(
            out_channels, bn_type=bn_type), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats,
                    proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats,
                    gt_label)
        elif self.fetch_attention:
            context, sim_map = self.object_context_block(feats, proxy_feats)
        else:
            context = self.object_context_block(feats, proxy_feats)
        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context,
                    bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1)
                    )
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        if self.fetch_attention:
            return output, sim_map
        else:
            return output


class SpatialOCR_Context(nn.Module):
    """
    Implementation of the FastOC module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, scale=1, dropout=0,
        bn_type=None):
        super(SpatialOCR_Context, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
            key_channels, scale, bn_type=bn_type)

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return context


class SpatialOCR_ASP_Module(nn.Module):

    def __init__(self, features, hidden_features=256, out_features=512,
        dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(SpatialOCR_ASP_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features,
            kernel_size=3, padding=1, dilation=1, bias=True), ModuleHelper.
            BNReLU(hidden_features, bn_type=bn_type), SpatialOCR_Context(
            in_channels=hidden_features, key_channels=hidden_features // 2,
            scale=1, bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features,
            kernel_size=1, padding=0, dilation=1, bias=True), ModuleHelper.
            BNReLU(hidden_features, bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features,
            kernel_size=3, padding=dilations[0], dilation=dilations[0],
            bias=True), ModuleHelper.BNReLU(hidden_features, bn_type=bn_type))
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features,
            kernel_size=3, padding=dilations[1], dilation=dilations[1],
            bias=True), ModuleHelper.BNReLU(hidden_features, bn_type=bn_type))
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features,
            kernel_size=3, padding=dilations[2], dilation=dilations[2],
            bias=True), ModuleHelper.BNReLU(hidden_features, bn_type=bn_type))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(hidden_features * 5,
            out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type), nn.
            Dropout2d(dropout))
        self.object_head = SpatialGather_Module(num_classes)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i],
                feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        feat1 = self.context[2](feat1, proxy_feats)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')
        output = self.conv_bn_dropout(out)
        return output


class HRNetBackbone(object):

    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        from lib.models.backbones.hrnet.hrnet_config import MODEL_CONFIGS
        if arch == 'hrnet18':
            arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet18'], bn_type=
                'inplace_abn', bn_momentum=0.1)
            arch_net = ModuleHelper.load_model(arch_net, pretrained=self.
                configer.get('network', 'pretrained'), all_match=False,
                network='hrnet')
        elif arch == 'hrnet32':
            arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet32'], bn_type=
                'inplace_abn', bn_momentum=0.1)
            arch_net = ModuleHelper.load_model(arch_net, pretrained=self.
                configer.get('network', 'pretrained'), all_match=False,
                network='hrnet')
        elif arch == 'hrnet48':
            arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet48'], bn_type=
                'inplace_abn', bn_momentum=0.1)
            arch_net = ModuleHelper.load_model(arch_net, pretrained=self.
                configer.get('network', 'pretrained'), all_match=False,
                network='hrnet')
        elif arch == 'hrnet64':
            arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet64'], bn_type=
                'inplace_abn', bn_momentum=0.1)
            arch_net = ModuleHelper.load_model(arch_net, pretrained=self.
                configer.get('network', 'pretrained'), all_match=False,
                network='hrnet')
        else:
            raise Exception('Architecture undefined!')
        return arch_net


class ResNeStModels(object):

    def __init__(self, configer):
        self.configer = configer

    def resnest50(self, **kwargs):
        model = ResNeSt(Bottleneck, [3, 4, 6, 3], radix=2, groups=1,
            bottleneck_width=64, dilated=True, dilation=4, deep_stem=False,
            stem_width=32, avg_down=True, avd=True, avd_first=False,
            bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnest')
        return model

    def deepbase_resnest50(self, **kwargs):
        model = ResNeSt(Bottleneck, [3, 4, 6, 3], radix=2, groups=1,
            bottleneck_width=64, dilated=True, dilation=4, deep_stem=True,
            stem_width=32, avg_down=True, avd=True, avd_first=False,
            bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnest')
        return model

    def resnest101(self, **kwargs):
        model = ResNeSt(Bottleneck, [3, 4, 23, 3], radix=2, groups=1,
            bottleneck_width=64, dilated=True, dilation=4, deep_stem=False,
            stem_width=64, avg_down=True, avd=True, avd_first=False,
            bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnest')
        return model

    def deepbase_resnest101(self, **kwargs):
        model = ResNeSt(Bottleneck, [3, 4, 23, 3], radix=2, groups=1,
            bottleneck_width=64, dilated=True, dilation=4, deep_stem=True,
            stem_width=64, avg_down=True, avd=True, avd_first=False,
            bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnest')
        return model

    def deepbase_resnest200(self, **kwargs):
        model = ResNeSt(Bottleneck, [3, 24, 36, 3], radix=2, groups=1,
            bottleneck_width=64, dilated=True, dilation=4, deep_stem=True,
            stem_width=64, avg_down=True, avd=True, avd_first=False,
            bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnest')
        return model

    def deepbase_resnest269(self, **kwargs):
        model = ResNeSt(Bottleneck, [3, 30, 48, 8], radix=2, groups=1,
            bottleneck_width=64, dilated=True, dilation=4, deep_stem=True,
            stem_width=64, avg_down=True, avd=True, avd_first=False,
            bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnest')
        return model


class ResNetModels(object):

    def __init__(self, configer):
        self.configer = configer

    def resnet18(self, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], deep_base=False, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def deepbase_resnet18(self, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], deep_base=True, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], deep_base=False, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def deepbase_resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], deep_base=True, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=False, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def deepbase_resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=True, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=False, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def deepbase_resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=True, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'))
        return model

    def resnet152(self, **kwargs):
        """Constructs a ResNet-152 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=False, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, all_match=False, pretrained=
            self.configer.get('network', 'pretrained'), network='resnet152')
        return model

    def deepbase_resnet152(self, **kwargs):
        """Constructs a ResNet-152 model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=True, bn_type=
            self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, all_match=False, pretrained=
            self.configer.get('network', 'pretrained'), network='resnet152')
        return model

    def wide_resnet16(self, **kwargs):
        """Constructs a WideResNet-16 model.
        """
        model = WiderResNetA2([1, 1, 1, 1, 1, 1], bn_type=self.configer.get
            ('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='wide_resnet')
        return model

    def wide_resnet20(self, **kwargs):
        """Constructs a WideResNet-20 model.
        """
        model = WiderResNetA2([1, 1, 1, 3, 1, 1], bn_type=self.configer.get
            ('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='wide_resnet')
        return model

    def wide_resnet38(self, **kwargs):
        """Constructs a WideResNet-38 model.
        """
        model = WiderResNetA2([3, 3, 6, 3, 1, 1], bn_type=self.configer.get
            ('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='wide_resnet')
        return model


model_urls = {'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'}


def ResNext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=
            progress)
        model.load_state_dict(state_dict)
    return model


class ResNextModels(object):

    def __init__(self, configer):
        self.configer = configer

    def resnext101_32x8d(self, **kwargs):
        """Constructs a ResNeXt-101 32x8d model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        pretrained = False
        progress = False
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        model = ResNext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
            pretrained, progress, bn_type=self.configer.get('network',
            'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnext')
        return model

    def resnext101_32x16d(self, **kwargs):
        """Constructs a ResNeXt-101 32x16d model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        pretrained = False
        progress = False
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 16
        model = ResNext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3],
            pretrained, progress, bn_type=self.configer.get('network',
            'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnext')
        return model

    def resnext101_32x32d(self, **kwargs):
        """Constructs a ResNeXt-101 32x32d model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        pretrained = False
        progress = False
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 32
        model = ResNext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3],
            pretrained, progress, bn_type=self.configer.get('network',
            'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnext')
        return model

    def resnext101_32x48d(self, **kwargs):
        """Constructs a ResNeXt-101 32x48d model.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        pretrained = False
        progress = False
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 48
        model = ResNext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3],
            pretrained, progress, bn_type=self.configer.get('network',
            'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, pretrained=self.configer.get
            ('network', 'pretrained'), all_match=False, network='resnext')
        return model


class ResNetBackbone(object):

    def __init__(self, configer):
        self.configer = configer
        self.resnet_models = ResNetModels(self.configer)
        self.resnext_models = ResNextModels(self.configer)
        self.resnest_models = ResNeStModels(self.configer)

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        multi_grid = None
        if self.configer.exists('network', 'multi_grid'):
            multi_grid = self.configer.get('network', 'multi_grid')
        if arch == 'deepbase_resnet18':
            orig_resnet = self.resnet_models.deepbase_resnet18()
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512
        elif arch == 'deepbase_resnet18_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet18()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
            arch_net.num_features = 512
        elif arch == 'deepbase_resnet18_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet18()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
            arch_net.num_features = 512
        elif arch == 'resnet34':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512
        elif arch == 'resnet34_dilated8':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
            arch_net.num_features = 512
        elif arch == 'resnet34_dilated16':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
            arch_net.num_features = 512
        elif arch == 'resnet50':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = NormalResnetBackbone(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'resnet50_dilated16':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
        elif arch == 'deepbase_resnet50':
            orig_resnet = self.resnet_models.deepbase_resnet50()
            arch_net = NormalResnetBackbone(orig_resnet)
        elif arch == 'deepbase_resnet50_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'deepbase_resnet50_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
        elif arch == 'resnet101':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = NormalResnetBackbone(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'resnet101_dilated16':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
        elif arch == 'deepbase_resnet101':
            orig_resnet = self.resnet_models.deepbase_resnet101()
            arch_net = NormalResnetBackbone(orig_resnet)
        elif arch == 'deepbase_resnet101_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'deepbase_resnet101_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
        elif arch == 'deepbase_resnet152_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet152()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'deepbase_resnet152_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet152()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16,
                multi_grid=multi_grid)
        elif arch == 'resnext101_32x8d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x8d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'resnext101_32x16d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x16d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'resnext101_32x32d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x32d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'resnext101_32x48d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x48d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8,
                multi_grid=multi_grid)
        elif arch == 'wide_resnet16_dilated8':
            arch_net = self.resnet_models.wide_resnet16()
        elif arch == 'wide_resnet20_dilated8':
            arch_net = self.resnet_models.wide_resnet20()
        elif arch == 'wide_resnet38_dilated8':
            arch_net = self.resnet_models.wide_resnet38()
        elif arch == 'deepbase_resnest50_dilated8':
            arch_net = self.resnest_models.deepbase_resnest50()
        elif arch == 'deepbase_resnest101_dilated8':
            arch_net = self.resnest_models.deepbase_resnest101()
        elif arch == 'deepbase_resnest200_dilated8':
            arch_net = self.resnest_models.deepbase_resnest200()
        elif arch == 'deepbase_resnest269_dilated8':
            arch_net = self.resnest_models.deepbase_resnest269()
        else:
            raise Exception('Architecture undefined!')
        return arch_net


class BackboneSelector(object):

    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get('network', 'backbone')
        model = None
        if ('resnet' in backbone or 'resnext' in backbone or 'resnest' in
            backbone) and 'senet' not in backbone:
            model = ResNetBackbone(self.configer)(**params)
        elif 'hrnet' in backbone:
            model = HRNetBackbone(self.configer)(**params)
        else:
            Log.error('Backbone {} is invalid.'.format(backbone))
            exit(1)
        return model


class CE2P_ASPOCR(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(CE2P_ASPOCR, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get(
                'network', 'bn_type'), factor=2)
            self.decoder = CE2P_Decoder_Module(self.num_classes, dropout=
                0.1, bn_type=self.configer.get('network', 'bn_type'),
                inplane1=512, inplane2=512)
        else:
            in_channels = [1024, 2048]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get(
                'network', 'bn_type'), factor=1)
            self.decoder = CE2P_Decoder_Module(self.num_classes, dropout=
                0.1, bn_type=self.configer.get('network', 'bn_type'),
                inplane1=512, inplane2=256)
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=2048,
            hidden_features=256, out_features=512, dilations=(6, 12, 18),
            num_classes=self.num_classes, bn_type=self.configer.get(
            'network', 'bn_type'))
        self.cls = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(256,
            bn_type=self.configer.get('network', 'bn_type')), nn.Conv2d(256,
            self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
        self.dsn = nn.Sequential(nn.Conv2d(in_channels[0], 512, kernel_size
            =3, stride=1, padding=1, bias=False), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        seg_dsn = self.dsn(x[-2])
        edge_out, edge_fea = self.edgelayer(x[-4], x[-3], x[-2])
        x5 = x[-1]
        x_hr = self.asp_ocr_head(x5, seg_dsn)
        seg_out1, x_hr = self.decoder(x_hr, x[-4])
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)
        seg_dsn = F.interpolate(seg_dsn, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        seg_out2 = F.interpolate(seg_out2, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        seg_out1 = F.interpolate(seg_out1, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        edge_out = F.interpolate(edge_out, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        return seg_out1, edge_out, seg_dsn, seg_out2


class CE2P_OCRNet(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(CE2P_OCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get(
                'network', 'bn_type'), factor=2)
            self.decoder = Decoder_Module(self.num_classes, dropout=0.1,
                bn_type=self.configer.get('network', 'bn_type'), inplane1=
                512, inplane2=512)
        else:
            in_channels = [1024, 2048]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get(
                'network', 'bn_type'), factor=1)
            self.decoder = Decoder_Module(self.num_classes, dropout=0.1,
                bn_type=self.configer.get('network', 'bn_type'), inplane1=
                512, inplane2=256)
        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=2048,
            key_channels=256, out_channels=512, scale=1, dropout=0, bn_type
            =self.configer.get('network', 'bn_type'))
        self.cls = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(256,
            bn_type=self.configer.get('network', 'bn_type')), nn.Conv2d(256,
            self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
        self.dsn = nn.Sequential(nn.Conv2d(in_channels[0], 512, kernel_size
            =3, stride=1, padding=1, bias=False), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        seg_dsn = self.dsn(x[-2])
        edge_out, edge_fea = self.edgelayer(x[-4], x[-3], x[-2])
        x5 = x[-1]
        context = self.spatial_context_head(x5, seg_dsn)
        x_hr = self.spatial_ocr_head(x5, context)
        seg_out1, x_hr = self.decoder(x_hr, x[-4])
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)
        seg_dsn = F.interpolate(seg_dsn, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        seg_out2 = F.interpolate(seg_out2, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        seg_out1 = F.interpolate(seg_out1, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        edge_out = F.interpolate(edge_out, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        return seg_out1, edge_out, seg_dsn, seg_out2


class CE2P_IdealOCRNet(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(CE2P_IdealOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get(
                'network', 'bn_type'), factor=2)
            self.decoder = Decoder_Module(self.num_classes, dropout=0.1,
                bn_type=self.configer.get('network', 'bn_type'), inplane1=
                512, inplane2=512)
        else:
            in_channels = [1024, 2048]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get(
                'network', 'bn_type'), factor=1)
            self.decoder = Decoder_Module(self.num_classes, dropout=0.1,
                bn_type=self.configer.get('network', 'bn_type'), inplane1=
                512, inplane2=256)
        self.spatial_context_head = SpatialGather_Module(self.num_classes,
            use_gt=True)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=2048,
            key_channels=256, out_channels=512, scale=1, dropout=0, use_gt=
            True, bn_type=self.configer.get('network', 'bn_type'))
        self.cls = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1,
            padding=0, dilation=1, bias=False), ModuleHelper.BNReLU(256,
            bn_type=self.configer.get('network', 'bn_type')), nn.Conv2d(256,
            self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
        self.dsn = nn.Sequential(nn.Conv2d(in_channels[0], 512, kernel_size
            =3, stride=1, padding=1, bias=False), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_, label_):
        x = self.backbone(x_)
        seg_dsn = self.dsn(x[-2])
        edge_out, edge_fea = self.edgelayer(x[-4], x[-3], x[-2])
        x5 = x[-1]
        label = F.interpolate(input=label_.unsqueeze(1).type(torch.
            FloatTensor), size=(x5.size(2), x5.size(3)), mode='nearest')
        context = self.spatial_context_head(x5, seg_dsn, label)
        x_hr = self.spatial_ocr_head(x5, context, label)
        seg_out1, x_hr = self.decoder(x_hr, x[-4])
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)
        seg_dsn = F.interpolate(seg_dsn, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        seg_out2 = F.interpolate(seg_out2, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        seg_out1 = F.interpolate(seg_out1, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        edge_out = F.interpolate(edge_out, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        return seg_out1, edge_out, seg_dsn, seg_out2


class FcnNet(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(FcnNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        elif 'mobilenetv2' in self.configer.get('network', 'backbone'):
            in_channels = [160, 320]
        else:
            in_channels = [1024, 2048]
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=False))
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=False))
        if 'mobilenetv2' in self.configer.get('network', 'backbone'):
            self.cls_head = nn.Sequential(nn.Conv2d(in_channels[1], 256,
                kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(
                256, bn_type=self.configer.get('network', 'bn_type')), nn.
                Dropout2d(0.1), nn.Conv2d(256, self.num_classes,
                kernel_size=1, stride=1, padding=0, bias=False))
            self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 128,
                kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(
                128, bn_type=self.configer.get('network', 'bn_type')), nn.
                Dropout2d(0.1), nn.Conv2d(128, self.num_classes,
                kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn_head(x[-2])
        x = self.cls_head(x[-1])
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return aux_x, x


class FcnNet_wo_dsn(nn.Module):

    def __init__(self, configer):
        self.inplanes = 128
        super(FcnNet_wo_dsn, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        elif 'mobilenetv2' in self.configer.get('network', 'backbone'):
            in_channels = [160, 320]
        else:
            in_channels = [1024, 2048]
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))
        if 'mobilenetv2' in self.configer.get('network', 'backbone'):
            self.cls_head = nn.Sequential(nn.Conv2d(in_channels[1], 256,
                kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(
                256, bn_type=self.configer.get('network', 'bn_type')), nn.
                Dropout2d(0.1), nn.Conv2d(256, self.num_classes,
                kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_):
        x = self.backbone(x_)
        x = self.cls_head(x[-1])
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x


class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        in_channels = 720
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels, in_channels,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(
            in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1), nn.Conv2d(in_channels, self.num_classes,
            kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode='bilinear',
            align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode='bilinear',
            align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode='bilinear',
            align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        return out


class HRNet_W48_ASPOCR(nn.Module):

    def __init__(self, configer):
        super(HRNet_W48_ASPOCR, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        in_channels = 720
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=720,
            hidden_features=256, out_features=256, dilations=(24, 48, 72),
            num_classes=self.num_classes, bn_type=self.configer.get(
            'network', 'bn_type'))
        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.aux_head = nn.Sequential(nn.Conv2d(in_channels, 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Conv2d(512,
            self.num_classes, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode='bilinear',
            align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode='bilinear',
            align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode='bilinear',
            align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)
        feats = self.asp_ocr_head(feats, out_aux)
        out = self.cls_head(feats)
        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        return out_aux, out


class HRNet_W48_OCR(nn.Module):

    def __init__(self, configer):
        super(HRNet_W48_OCR, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        in_channels = 720
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1,
            stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(nn.Conv2d(in_channels, in_channels,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(
            in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode='bilinear',
            align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode='bilinear',
            align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode='bilinear',
            align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)
        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        return out_aux, out


class HRNet_W48_OCR_B(nn.Module):
    """
    Considering that the 3x3 convolution on the 4x resolution feature map is expensive,
    we can decrease the intermediate channels from 512 to 256 w/o performance loss.
    """

    def __init__(self, configer):
        super(HRNet_W48_OCR_B, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        in_channels = 720
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels, 256,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(256,
            bn_type=self.configer.get('network', 'bn_type')))
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=256,
            key_channels=128, out_channels=256, scale=1, dropout=0.05,
            bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1,
            stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(nn.Conv2d(in_channels, 256,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(256,
            bn_type=self.configer.get('network', 'bn_type')), nn.Conv2d(256,
            self.num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode='bilinear',
            align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode='bilinear',
            align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode='bilinear',
            align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)
        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)),
            mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        return out_aux, out


class IdealSpatialOCRNet(nn.Module):
    """
    augment the representations with the ground-truth object context.
    """

    def __init__(self, configer):
        super(IdealSpatialOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.spatial_context_head = SpatialGather_Module(self.num_classes,
            use_gt=True)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            use_gt=True, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.05), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_, label_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        label = F.interpolate(input=label_.unsqueeze(1).type(torch.
            FloatTensor), size=(x.size(2), x.size(3)), mode='nearest')
        context = self.spatial_context_head(x, x_dsn, label)
        x = self.spatial_ocr_head(x, context, label)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class IdealSpatialOCRNetB(nn.Module):
    """
    augment the representations with both the ground-truth background context and object context.
    """

    def __init__(self, configer):
        super(IdealSpatialOCRNetB, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.spatial_context_head = SpatialGather_Module(self.num_classes,
            use_gt=True)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            use_gt=True, use_bg=True, bn_type=self.configer.get('network',
            'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.05), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_, label_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        label = F.interpolate(input=label_.unsqueeze(1).type(torch.
            FloatTensor), size=(x.size(2), x.size(3)), mode='nearest')
        context = self.spatial_context_head(x, x_dsn, label)
        x = self.spatial_ocr_head(x, context, label)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class IdealSpatialOCRNetC(nn.Module):
    """
    augment the representations with only the ground-truth background context.
    """

    def __init__(self, configer):
        super(IdealSpatialOCRNetC, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.spatial_context_head = SpatialGather_Module(self.num_classes,
            use_gt=True)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            use_gt=True, use_bg=True, use_oc=False, bn_type=self.configer.
            get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.05), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_, label_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        label = F.interpolate(input=label_.unsqueeze(1).type(torch.
            FloatTensor), size=(x.size(2), x.size(3)), mode='nearest')
        context = self.spatial_context_head(x, x_dsn, label)
        x = self.spatial_ocr_head(x, context, label)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class IdealGatherOCRNet(nn.Module):

    def __init__(self, configer):
        super(IdealGatherOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.spatial_context_head = SpatialGather_Module(self.num_classes,
            use_gt=True)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            use_gt=False, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.05), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_, label_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        label = F.interpolate(input=label_.unsqueeze(1).type(torch.
            FloatTensor), size=(x.size(2), x.size(3)), mode='nearest')
        context = self.spatial_context_head(x, x_dsn, label)
        x = self.spatial_ocr_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class IdealDistributeOCRNet(nn.Module):

    def __init__(self, configer):
        super(IdealDistributeOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.spatial_context_head = SpatialGather_Module(self.num_classes,
            use_gt=False)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            use_gt=True, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.05), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_, label_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        label = F.interpolate(input=label_.unsqueeze(1).type(torch.
            FloatTensor), size=(x.size(2), x.size(3)), mode='nearest')
        context = self.spatial_context_head(x, x_dsn)
        x = self.spatial_ocr_head(x, context, label)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class ISANet(nn.Module):
    """
    Interlaced Sparse Self-Attention for Semantic Segmentation
    """

    def __init__(self, configer):
        self.inplanes = 128
        super(ISANet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        bn_type = self.configer.get('network', 'bn_type')
        factors = self.configer.get('network', 'factors')
        self.isa_head = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3,
            stride=1, padding=1, bias=False), ModuleHelper.BNReLU(512,
            bn_type=bn_type), ISA_Module(in_channels=512, key_channels=256,
            value_channels=512, out_channels=512, down_factors=factors,
            dropout=0.05, bn_type=bn_type))
        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1,
            stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3,
            stride=1, padding=1, bias=False), ModuleHelper.BNReLU(512,
            bn_type=bn_type), nn.Dropout2d(0.05), nn.Conv2d(512, self.
            num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.isa_head(x[-1])
        x = self.cls_head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class BaseOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """

    def __init__(self, configer):
        self.inplanes = 128
        super(BaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.oc_module = BaseOC_Module(in_channels=512, out_channels=512,
            key_channels=256, value_channels=256, dropout=0.05, sizes=[1],
            bn_type=self.configer.get('network', 'bn_type'))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.dsn = nn.Sequential(nn.Conv2d(in_channels[0], 512, kernel_size
            =3, stride=1, padding=1), ModuleHelper.BNReLU(512, bn_type=self
            .configer.get('network', 'bn_type')), nn.Conv2d(512, self.
            num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x = self.oc_module_pre(x[-1])
        x = self.oc_module(x)
        x = self.cls(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class AspOCNet(nn.Module):
    """
    OCNet: Object Context Network for Scene Parsing
    """

    def __init__(self, configer):
        self.inplanes = 128
        super(AspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.context = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), ASP_OC_Module
            (512, 256, bn_type=self.configer.get('network', 'bn_type')))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True)
        self.dsn = nn.Sequential(nn.Conv2d(in_channels[0], 512, kernel_size
            =3, stride=1, padding=1), ModuleHelper.BNReLU(512, bn_type=self
            .configer.get('network', 'bn_type')), nn.Conv2d(512, self.
            num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.context(x[-1])
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return aux_x, x


class SpatialOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """

    def __init__(self, configer):
        self.inplanes = 128
        super(SpatialOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(nn.Conv2d(in_channels[1], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')))
        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
            key_channels=256, out_channels=512, scale=1, dropout=0.05,
            bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.05), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        context = self.spatial_context_head(x, x_dsn)
        x = self.spatial_ocr_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


class ASPOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """

    def __init__(self, configer):
        self.inplanes = 128
        super(ASPOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        if 'wide_resnet38' in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=2048,
            hidden_features=256, out_features=256, num_classes=self.
            num_classes, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=
            1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(nn.Conv2d(in_channels[0], 512,
            kernel_size=3, stride=1, padding=1), ModuleHelper.BNReLU(512,
            bn_type=self.configer.get('network', 'bn_type')), nn.Dropout2d(
            0.1), nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1,
            padding=0, bias=True))

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.asp_ocr_head(x[-1], x_dsn)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode=
            'bilinear', align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode='bilinear',
            align_corners=True)
        return x_dsn, x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_openseg_group_openseg_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ABN(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DataParallelModel(*[], **{'module': _mock_layer()}), [], {'input': torch.rand([4, 4])})

    def test_002(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(OffsetBlock(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(OffsetModule(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(PAM_Module(*[], **{'in_dim': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_006(self):
        self._check(PacCRF(*[], **{'channels': 4, 'num_steps': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(PacCRFLoose(*[], **{'channels': 4, 'num_steps': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(PyramidSpatialGather_Module(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(SingleGPU(*[], **{'module': _mock_layer()}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(SpatialGather_Module(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(SwitchNorm1d(*[], **{'num_features': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_012(self):
        self._check(SwitchNorm2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(rSoftMax(*[], **{'radix': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

