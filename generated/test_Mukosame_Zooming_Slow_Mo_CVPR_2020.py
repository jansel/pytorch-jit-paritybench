import sys
_module = sys.modules[__name__]
del sys
Vimeo7_dataset = _module
data = _module
data_sampler = _module
util = _module
create_lmdb_mp = _module
generate_mod_LR_bic = _module
sep_vimeo_list = _module
VideoSR_base_model = _module
models = _module
base_model = _module
lr_scheduler = _module
DCNv2 = _module
dcn_v2 = _module
setup = _module
test = _module
Sakuya_arch = _module
modules = _module
convlstm = _module
loss = _module
module_util = _module
networks = _module
options = _module
test = _module
train = _module
utils = _module
make_video = _module
util = _module
video_to_zsm = _module

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


import random


import logging


import numpy as np


import torch


import torch.utils.data as data


import torch.utils.data


import math


from torch.utils.data.sampler import Sampler


import torch.distributed as dist


from collections import OrderedDict


import torch.nn as nn


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from collections import Counter


from collections import defaultdict


from torch.optim.lr_scheduler import _LRScheduler


from torch import nn


from torch.autograd import Function


from torch.nn.modules.utils import _pair


from torch.autograd.function import once_differentiable


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import time


from torch.autograd import gradcheck


import functools


import torch.nn.functional as F


from torch.autograd import Variable


import torch.nn.functional as fnn


import torch.nn.init as init


import torch.multiprocessing as mp


from torchvision.utils import make_grid


import re


class _DCNv2(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding, dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(input, weight, bias, offset, mask, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1], ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = _backend.dcn_v2_backward(input, weight, bias, offset, mask, grad_output, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0], ctx.padding[1], ctx.dilation[0], ctx.dilation[1], ctx.deformable_groups)
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None


dcn_v2_conv = _DCNv2.apply


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == mask.shape[1]
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)


logger = logging.getLogger('base')


class DCN_sep(DCNv2):
    """Use other features to generate offsets and masks"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCN_sep, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea):
        """input: input features for deformable conv
        fea: other features used for generating offsets and mask"""
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.deformable_groups)


class _DCNv2Pooling(Function):

    @staticmethod
    def forward(ctx, input, rois, offset, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        ctx.spatial_scale = spatial_scale
        ctx.no_trans = int(no_trans)
        ctx.output_dim = output_dim
        ctx.group_size = group_size
        ctx.pooled_size = pooled_size
        ctx.part_size = pooled_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        output, output_count = _backend.dcn_v2_psroi_pooling_forward(input, rois, offset, ctx.no_trans, ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        ctx.save_for_backward(input, rois, offset, output_count)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset, output_count = ctx.saved_tensors
        grad_input, grad_offset = _backend.dcn_v2_psroi_pooling_backward(grad_output, input, rois, offset, output_count, ctx.no_trans, ctx.spatial_scale, ctx.output_dim, ctx.group_size, ctx.pooled_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, None, grad_offset, None, None, None, None, None, None, None, None


dcn_v2_pooling = _DCNv2Pooling.apply


class DCNv2Pooling(nn.Module):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            offset = input.new()
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class DCNPooling(DCNv2Pooling):

    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, deform_fc_dim=1024):
        super(DCNPooling, self).__init__(spatial_scale, pooled_size, output_dim, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.deform_fc_dim = deform_fc_dim
        if not no_trans:
            self.offset_mask_fc = nn.Sequential(nn.Linear(self.pooled_size * self.pooled_size * self.output_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.deform_fc_dim), nn.ReLU(inplace=True), nn.Linear(self.deform_fc_dim, self.pooled_size * self.pooled_size * 3))
            self.offset_mask_fc[4].weight.data.zero_()
            self.offset_mask_fc[4].bias.data.zero_()

    def forward(self, input, rois):
        offset = input.new()
        if not self.no_trans:
            n = rois.shape[0]
            roi = dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = torch.sigmoid(mask)
            return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask
        return dcn_v2_pooling(input, rois, offset, self.spatial_scale, self.pooled_size, self.output_dim, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class PCD_Align(nn.Module):
    """ Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    """

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        """align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        """
        y = []
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        y = torch.cat(y, dim=1)
        return y


class Easy_PCD(nn.Module):

    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        fea1 = [L1_fea[:, (0), :, :, :].clone(), L2_fea[:, (0), :, :, :].clone(), L3_fea[:, (0), :, :, :].clone()]
        fea2 = [L1_fea[:, (1), :, :, :].clone(), L2_fea[:, (1), :, :, :].clone(), L3_fea[:, (1), :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea)
        return fusion_fea


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return Variable(torch.zeros(batch_size, self.hidden_dim, height, width)), Variable(torch.zeros(batch_size, self.hidden_dim, height, width))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width), input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = input_tensor.size(3), input_tensor.size(4)
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, (t), :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, tensor_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size])):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class DeformableConvLSTM(ConvLSTM):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups, batch_first=False, bias=True, return_all_layers=False):
        ConvLSTM.__init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        nf = input_dim
        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)
        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width), input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_tensor, hidden_state=None):
        """        
        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            None. 
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = input_tensor.size(3), input_tensor.size(4)
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, (t), :, :, :]
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor, cur_state=[h_temp, c_temp])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        return super()._init_hidden(batch_size, tensor_size)


class BiDeformableConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups, batch_first=False, bias=True, return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers, front_RBs=front_RBs, groups=groups, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2 * input_dim, input_dim, 1, 1, bias=True)

    def forward(self, x):
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, (reversed_idx), (...)]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        rev_rev = out_rev[0][:, (reversed_idx), (...)]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B * N, -1, H, W)
        result = self.conv_1x1(result)
        return result.view(B, -1, C, H, W)


class LunaTokis(nn.Module):

    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10):
        super(LunaTokis, self).__init__()
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        p_size = 48
        patch_size = p_size, p_size
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)
        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=patch_size, input_dim=nf, hidden_dim=hidden_dim, kernel_size=(3, 3), num_layers=1, batch_first=True, front_RBs=front_RBs, groups=groups)
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        to_lstm_fea = []
        """
        0: + fea1, fusion_fea, fea2
        1: + ...    ...        ...  fusion_fea, fea2
        2: + ...    ...        ...    ...       ...   fusion_fea, fea2
        """
        for idx in range(N - 1):
            fea1 = [L1_fea[:, (idx), :, :, :].clone(), L2_fea[:, (idx), :, :, :].clone(), L3_fea[:, (idx), :, :, :].clone()]
            fea2 = [L1_fea[:, (idx + 1), :, :, :].clone(), L2_fea[:, (idx + 1), :, :, :].clone(), L3_fea[:, (idx + 1), :, :, :].clone()]
            aligned_fea = self.pcd_align(fea1, fea2)
            fusion_fea = self.fusion(aligned_fea)
            if idx == 0:
                to_lstm_fea.append(fea1[0])
            to_lstm_fea.append(fusion_fea)
            to_lstm_fea.append(fea2[0])
        lstm_feats = torch.stack(to_lstm_fea, dim=1)
        feats = self.ConvBLSTM(lstm_feats)
        B, T, C, H, W = feats.size()
        feats = feats.view(B * T, C, H, W)
        out = self.recon_trunk(feats)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, K, G = out.size()
        outs = out.view(B, T, -1, K, G)
        return outs


class ConvBLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(input_size, input_dim, hidden_dims // 2, kernel_size, num_layers, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        self.reverse_net = ConvLSTM(input_size, input_dim, hidden_dims // 2, kernel_size, num_layers, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """
        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)
        if not self.return_all_layers:
            y_out_fwd = y_out_fwd[-1]
            y_out_rev = y_out_rev[-1]
        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, (reversed_idx), (...)]
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)
        return ycat


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-06):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError('kernel size must be uneven')
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, (None), :, :])
    if cuda:
        kernel = kernel
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)
    pyr.append(current)
    return pyr


class LapLoss(nn.Module):

    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, input, target):
        if len(input.shape) == 5:
            B, N, C, H, W = input.size()
            input = input.view(-1, C, H, W)
            target = target.view(-1, C, H, W)
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma, n_channels=input.shape[1], cuda=input.is_cuda)
        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CharbonnierLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLSTMCell,
     lambda: ([], {'input_size': [4, 4], 'input_dim': 4, 'hidden_dim': 4, 'kernel_size': [4, 4], 'bias': 4}),
     lambda: ([torch.rand([4, 4, 64, 64]), (torch.rand([4, 4, 64, 64]), torch.rand([4, 4, 65, 65]))], {}),
     False),
    (LapLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 512, 512]), torch.rand([4, 4, 512, 512])], {}),
     False),
    (ResidualBlock_noBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
]

class Test_Mukosame_Zooming_Slow_Mo_CVPR_2020(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

