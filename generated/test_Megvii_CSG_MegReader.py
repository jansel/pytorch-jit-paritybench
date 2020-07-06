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
base = _module
crnn = _module
feature_pyramid = _module
fpn_top_down = _module
ppm = _module
resnet = _module
resnet_dilated = _module
resnet_fpn = _module
resnet_ppm = _module
upsample_head = _module
concern = _module
average_meter = _module
box2seg = _module
charset_tool = _module
charsets = _module
config = _module
convert = _module
cv = _module
distributed = _module
icdar2015_eval = _module
detection = _module
deteval = _module
icdar2013 = _module
iou = _module
mtwi2018 = _module
log = _module
nori_reader = _module
redis_meta = _module
signal_monitor = _module
tensorboard = _module
test_log = _module
textsnake = _module
visualizer = _module
webcv2 = _module
manager = _module
server = _module
data = _module
augmenter = _module
crop_file_dataset = _module
data_loader = _module
dataset = _module
east = _module
file_dataset = _module
list_dataset = _module
lmdb_dataset = _module
local_csv = _module
meta = _module
meta_loader = _module
meta_loaders = _module
charbox_meta_loader = _module
data_id_meta_loader = _module
detection_meta_loader = _module
json_meta_loader = _module
lmdb_meta_loader = _module
meta_cache = _module
nori_meta_loader = _module
recognition_meta_loader = _module
text_lines_meta_loader = _module
mingle_dataset = _module
mnist = _module
nori_dataset = _module
processes = _module
augment_data = _module
charboxes_from_textlines = _module
data_process = _module
extract_detetion_data = _module
filter_keys = _module
make_border_map = _module
make_center_distance_map = _module
make_center_map = _module
make_center_points = _module
make_decouple_map = _module
make_density_map = _module
make_icdar_data = _module
make_keypoint_map = _module
make_recognition_label = _module
make_seg_detection_data = _module
make_seg_recognition_label = _module
normalize_image = _module
random_crop_data = _module
resize_image = _module
serialize_box = _module
quad = _module
simple_detection = _module
text_lines = _module
unpack_msgpack_data = _module
decoders = _module
attention_decoder = _module
balance_cross_entropy_loss = _module
classification = _module
crnn = _module
ctc_decoder = _module
ctc_decoder2d = _module
ctc_loss = _module
ctc_loss2d = _module
dice_loss = _module
east = _module
l1_loss = _module
pss_loss = _module
seg_detector = _module
seg_detector_loss = _module
seg_recognizer = _module
simple_detection = _module
textsnake = _module
eval = _module
experiment = _module
ops = _module
ctc_loss_2d = _module
setup = _module
json_to_lmdb = _module
nori_to_lmdb = _module
structure = _module
builder = _module
ensemble_model = _module
measurers = _module
classification_measurer = _module
grid_sampling_measurer = _module
icdar_detection_measurer = _module
quad_measurer = _module
sequence_recognition_measurer = _module
model = _module
maskrcnn_benchmark = _module
representers = _module
classification_representer = _module
ctc_representer = _module
ctc_representer2d = _module
ensemble_ctc_representer = _module
integral_regression_representer = _module
mask_rcnn = _module
seg_detector_representer = _module
seg_recognition_representer = _module
sequence_recognition_representer = _module
visualizer = _module
visualizers = _module
ctc_visualizer2d = _module
seg_detector_visualizer = _module
seg_recognition_visualizer = _module
sequence_recognition_visualizer = _module
textsnake = _module
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


import functools


import torch.distributed as dist


from collections import defaultdict


import torch.utils.data as data


import time


import numpy as np


from torch.utils.data import Sampler


from torch.utils.data import ConcatDataset


from torch.utils.data import BatchSampler


from torch.utils.data import Dataset as TorchDataset


from collections import OrderedDict


import warnings


from scipy import ndimage


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


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


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.kernels = [3, 3, 3, 3, 3, 3, 2]
        self.paddings = [1, 1, 1, 1, 1, 1, 0]
        self.strides = [1, 1, 1, 1, 1, 1, 1]
        self.channels = [64, 128, 256, 256, 512, 512, 512, nc]
        conv0 = nn.Sequential(self._make_layer(0), nn.MaxPool2d((2, 2)))
        conv1 = nn.Sequential(self._make_layer(1), nn.MaxPool2d((2, 2)))
        conv2 = self._make_layer(2, True)
        conv3 = nn.Sequential(self._make_layer(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv4 = self._make_layer(4, True)
        conv5 = nn.Sequential(self._make_layer(5), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv6 = self._make_layer(6, True)
        self.cnn = nn.Sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6)

    def _make_layer(self, i, batch_normalization=False):
        in_channel = self.channels[i - 1]
        out_channel = self.channels[i]
        layer = list()
        layer.append(nn.Conv2d(in_channel, out_channel, self.kernels[i], self.strides[i], self.paddings[i]))
        if batch_normalization:
            layer.append(nn.BatchNorm2d(out_channel))
        else:
            layer.append(nn.ReLU())
        return nn.Sequential(*layer)

    def forward(self, input):
        return self.cnn(input)


class FeaturePyramid(nn.Module):

    def __init__(self, bottom_up, top_down):
        nn.Module.__init__(self)
        self.bottom_up = bottom_up
        self.top_down = top_down

    def forward(self, feature):
        pyramid_features = self.bottom_up(feature)
        feature = self.top_down(pyramid_features[::-1])
        return feature


class FPNTopDown(nn.Module):

    def __init__(self, pyramid_channels, feature_channel):
        nn.Module.__init__(self)
        self.reduction_layers = nn.ModuleList()
        for pyramid_channel in pyramid_channels:
            reduction_layer = nn.Conv2d(pyramid_channel, feature_channel, kernel_size=1, stride=1, padding=0, bias=False)
            self.reduction_layers.append(reduction_layer)
        self.merge_layer = nn.Conv2d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pyramid_features):
        feature = None
        for pyramid_feature, reduction_layer in zip(pyramid_features, self.reduction_layers):
            pyramid_feature = reduction_layer(pyramid_feature)
            if feature is None:
                feature = pyramid_feature
            else:
                feature = self.upsample_add(feature, pyramid_feature)
        feature = self.merge_layer(feature)
        return feature


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class PPMDeepsup(nn.Module):

    def __init__(self, inner_channels=256, fc_dim=2048, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, inner_channels, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        return x


def bn(*args, **kwargs):
    if config.sync_bn:
        return apex.parallel.SyncBatchNorm(*args, **kwargs)
    else:
        return nn.BatchNorm2d(*args, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn(planes)
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
        self.bn2 = bn(planes)
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
        self.bn1 = bn(planes)
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
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4)
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

    def __init__(self, block, layers, num_classes=1000, dcn=None, stage_with_dcn=(False, False, False, False), dilations=[1, 1, 1, 1]):
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = bn(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = bn(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = bn(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn, dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn, dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn, dilation=dilations[3])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, dilation=dilation), bn(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5


class ResnetDilated(nn.Module):

    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
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

    def forward(self, x, return_feature_maps=True):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return x


class Attn(nn.Module):

    def __init__(self, method, hidden_dims, embed_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_dims = hidden_dims
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_dims + embed_size, hidden_dims)
        self.v = nn.Parameter(torch.rand(hidden_dims))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(0)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class AttentionRNNCell(nn.Module):

    def __init__(self, hidden_dims, embedded_dims, nr_classes, n_layers=1, dropout_p=0, bidirectional=False):
        super(AttentionRNNCell, self).__init__()
        self.hidden_dims = hidden_dims
        self.embedded_dims = embedded_dims
        self.nr_classes = nr_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(nr_classes, nr_classes)
        self.embedding.weight.data = torch.eye(nr_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(nr_classes, hidden_dims)
        self.attn = Attn('concat', hidden_dims, embedded_dims)
        self.rnn = nn.GRUCell(2 * hidden_dims + embedded_dims, hidden_dims)
        self.out = nn.Linear(hidden_dims, nr_classes)

    def forward(self, word_input, last_hidden, encoder_outputs, train=True):
        """
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        """
        batch_size = word_input.size(0)
        word_embedded_onehot = self.embedding(word_input.to(last_hidden.device).type(torch.long)).view(1, batch_size, -1)
        word_embedded = self.word_linear(word_embedded_onehot)
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_input = torch.cat([word_embedded, context], 2)
        last_hidden = last_hidden.view(batch_size, -1)
        rnn_input = rnn_input.view(batch_size, -1)
        hidden = self.rnn(rnn_input, last_hidden)
        if train:
            output = nn.functional.log_softmax(self.out(hidden), dim=1)
        else:
            output = nn.functional.softmax(self.out(hidden), dim=1)
        return output, hidden, attn_weights


class State:

    def __init__(self, cmd_key=None, autoload=True, default=None):
        self.autoload = autoload
        self.default = default
        self.cmd_key = cmd_key


class StateMeta(type):

    def __new__(mcs, name, bases, attrs):
        current_states = []
        for key, value in attrs.items():
            if isinstance(value, State):
                current_states.append((key, value))
        current_states.sort(key=lambda x: x[0])
        attrs['states'] = OrderedDict(current_states)
        new_class = super(StateMeta, mcs).__new__(mcs, name, bases, attrs)
        states = OrderedDict()
        for base in reversed(new_class.__mro__):
            if hasattr(base, 'states'):
                states.update(base.states)
        new_class.states = states
        for key, value in states.items():
            setattr(new_class, key, value.default)
        return new_class


class Configurable(metaclass=StateMeta):

    def __init__(self, *args, cmd={}, **kwargs):
        self.load_all(cmd=cmd, **kwargs)

    @staticmethod
    def construct_class_from_config(args):
        cls = Configurable.extract_class_from_args(args)
        return cls(**args)

    @staticmethod
    def extract_class_from_args(args):
        cls = args.copy().pop('class')
        package, cls = cls.rsplit('.', 1)
        module = importlib.import_module(package)
        cls = getattr(module, cls)
        return cls

    def load_all(self, cmd={}, **kwargs):
        for name, state in self.states.items():
            if state.cmd_key is not None and name in cmd:
                setattr(self, name, cmd[name])
            elif state.autoload:
                self.load(name, cmd=cmd, **kwargs)

    def load(self, state_name, **kwargs):
        cmd = kwargs.pop('cmd', dict())
        if state_name in kwargs:
            setattr(self, state_name, self.create_member_from_config((kwargs[state_name], cmd)))
        else:
            setattr(self, state_name, self.states[state_name].default)

    def create_member_from_config(self, conf):
        args, cmd = conf
        if args is None or isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, (list, tuple)):
            return [self.create_member_from_config((subargs, cmd)) for subargs in args]
        elif isinstance(args, dict):
            if 'class' in args:
                cls = self.extract_class_from_args(args)
                return cls(**args, cmd=cmd)
            return {key: self.create_member_from_config((subargs, cmd)) for key, subargs in args.items()}
        else:
            return args

    def dump(self):
        state = {}
        state['class'] = self.__class__.__module__ + '.' + self.__class__.__name__
        for name, value in self.states.items():
            obj = getattr(self, name)
            state[name] = self.dump_obj(obj)
        return state

    def dump_obj(self, obj):
        if obj is None:
            return None
        elif hasattr(obj, 'dump'):
            return obj.dump()
        elif isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.dump_obj(value) for value in obj]
        elif isinstance(obj, dict):
            return {key: self.dump_obj(value) for key, value in obj.items()}
        else:
            return str(obj)


class Charset(Configurable):
    corups = State(default=string.hexdigits)
    blank = State(default=0)
    unknown = State(default=1)
    blank_char = State('\t')
    unknown_char = State('\n')
    case_sensitive = State(default=False)

    def __init__(self, corups=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self._corpus = SortedSet(self._filter_corpus(corups))
        if self.blank_char in self._corpus:
            self._corpus.remove(self.blank_char)
        if self.unknown_char in self._corpus:
            self._corpus.remove(self.unknown_char)
        self._charset = list(self._corpus)
        self._charset.insert(self.blank, self.blank_char)
        self._charset.insert(self.unknown, self.unknown_char)
        self._charset_lut = {char: index for index, char in enumerate(self._charset)}

    def _filter_corpus(self, corups):
        return corups

    def __getitem__(self, index):
        return self._charset[index]

    def index(self, x):
        target = x
        if not self.case_sensitive:
            target = target.upper()
        return self._charset_lut.get(target, self.unknown)

    def is_empty(self, index):
        return index == self.blank or index == self.unknown

    def is_empty_char(self, x):
        return x == self.blank_char or x == self.unknown_char

    def __len__(self):
        return len(self._charset)

    def string_to_label(self, string_input, max_size=32):
        length = max(max_size, len(string_input))
        target = np.zeros((length,), dtype=np.int32)
        for index, c in enumerate(string_input):
            value = self.index(c)
            target[index] = value
        return target

    def label_to_string(self, label):
        ingnore = [self.unknown, self.blank]
        return ''.join([self._charset[i] for i in label if i not in ingnore])


class EnglishCharset(Charset):

    def __init__(self, cmd={}, **kwargs):
        corups = string.digits + string.ascii_uppercase
        super().__init__(corups, cmd, **kwargs)


DefaultCharset = EnglishCharset


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
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')[:, (0), :, :]
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        if return_origin:
            return balance_loss, loss
        return balance_loss


class ClassificationDecoder(nn.Module):

    def __init__(self):
        super(ClassificationDecoder, self).__init__()
        self.fc = torch.nn.Linear(256, 10)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, feature_map, targets=None, train=False):
        x = torch.max(torch.max(feature_map, dim=3)[0], dim=2)[0]
        x = self.fc(x)
        pred = x
        if train:
            loss = self.criterion(pred, targets)
            return loss, pred
        else:
            return pred


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


class CTCLoss(nn.Module):

    def __init__(self, blank=0, reduction='mean'):
        """The Connectionist Temporal Classification loss.

	Args:
	    blank (int, optional): blank label. Default :math:`0`.
	    reduction (string, optional): Specifies the reduction to apply to the output:
		'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
		'mean': the output losses will be divided by the target lengths and
		then the mean over the batch is taken. Default: 'mean'

	Inputs:
	    log_probs: Tensor of size :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
		`T = input length`, and `N = batch size`.
		The logarithmized probabilities of the outputs
		(e.g. obtained with :func:`torch.nn.functional.log_softmax`).
	    targets: Tensor of size :math:`(N, S)` or `(sum(target_lengths))`.
		Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
	    input_lengths: Tuple or tensor of size :math:`(N)`.
		Lengths of the inputs (must each be :math:`\\leq T`)
	    target_lengths: Tuple or tensor of size  :math:`(N)`.
		Lengths of the targets


	Example::

	    >>> ctc_loss = CTCLoss()
	    >>> log_probs = torch.randn(12, 16, 20).log_softmax(2).detach().requires_grad_()
	    >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
	    >>> input_lengths = torch.full((16,), 12, dtype=torch.long)
	    >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
	    >>> loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
	    >>> loss.backward()

	Reference:
	    A. Graves et al.: Connectionist Temporal Classification:
	    Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
	    https://www.cs.toronto.edu/~graves/icml_2006.pdf
	"""
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction

    def expand_with_blank(self, targets):
        N, S = targets.shape
        blank = torch.tensor([self.blank], dtype=torch.long).repeat(targets.shape)
        expanded_targets = torch.cat([blank.unsqueeze(-1), targets.unsqueeze(-1)], -1)
        expanded_targets = expanded_targets.view(N, -1)
        expanded_targets = torch.cat([expanded_targets, blank[:, 0:1]], dim=-1)
        return expanded_targets

    def log_add(self, a, b):
        x, y = torch.max(a, b), torch.min(a, b)
        return x + torch.log1p(torch.exp(y - x))

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
        log_probs: :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`[NOT IMPLEMENTED YET].
            Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
	"""
        targets = targets.type(torch.long)
        expanded_targets = self.expand_with_blank(targets)
        N, S = expanded_targets.shape
        T = log_probs.shape[0]
        tiny = torch.finfo().tiny
        probability = torch.log(torch.zeros(S, N) + tiny)
        probability[0] = log_probs[(0), :, (self.blank)]
        batch_indices = torch.linspace(0, N - 1, N).type(torch.long) * log_probs.shape[-1]
        indices = batch_indices + expanded_targets[:, (1)]
        probability[1] = log_probs[0].take(indices)
        mask_skipping = torch.ne(expanded_targets[:, 2:], expanded_targets[:, :-2]).transpose(0, 1)
        mask_skipping = mask_skipping.type(torch.float)
        mask_not_skipping = 1 - mask_skipping
        for timestep in range(1, T):
            new_probability1 = self.log_add(probability[1:], probability[:-1])
            new_probability2 = self.log_add(new_probability1[1:], probability[:-2]) * mask_skipping + new_probability1[1:] * mask_not_skipping
            new_probability = torch.cat([probability[:1], new_probability1[:1], new_probability2], dim=0)
            probability = new_probability + log_probs[timestep].gather(1, expanded_targets).transpose(0, 1)
            """
            probability[2:] = torch.log(torch.exp(probability[2:]) +                    torch.exp(probability[1:-1]) +                    torch.exp(probability[:-2]) * mask_skipping)
            probability[1] = torch.log(torch.exp(probability[0]) + torch.exp(probability[1]) + tiny)
            probability = probability + log_probs[timestep].gather(1, expanded_targets).transpose(0, 1)
            """
        lengths = (target_lengths * 2 + 1).unsqueeze(0)
        loss = self.log_add(probability.gather(0, lengths - 1), probability.gather(0, lengths - 2))
        loss = loss.squeeze(0)
        if self.reduction == 'mean':
            return -(loss / target_lengths.type(torch.float))
        return -loss


class CTCLoss2D(nn.Module):

    def __init__(self, blank=0, reduction='mean'):
        """The python-implementation of 2D-CTC loss.
        NOTICE: This class is only for the useage of understanding the principle of 2D-CTC.
            Please use `ops.ctc_loss_2d` for practice.
        Args:
            blank (int, optional): blank label. Default :math:`0`.
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the output losses will be divided by the target lengths and
                then the mean over the batch is taken. Default: 'mean'

        Inputs:
            mask: Tensor of size :math:`(T, H, N)` where `H = height`,
                `T = input length`, and `N = batch size`.
                The logarithmized path transition probabilities.
                (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
            classify: Tensor of size :math:`(T, H, N, C)` where `C = number of classes`, `H = height`, `T = input length`, and `N = batch size`.
                The logarithmized character classification probabilities at all possible path pixels.
                (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
            targets: Tensor of size :math:`(N, S)` or `(sum(target_lengths))`.
                Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
            input_lengths: Tuple or tensor of size :math:`(N)`.
                Lengths of the inputs (must each be :math:`\\leq T`)
            target_lengths: Tuple or tensor of size  :math:`(N)`.
                Lengths of the targets

        Example::

            >>> ctc_loss = CTCLoss2D()
            >>> N, H, T, C = 16, 8, 32, 20
            >>> mask = torch.randn(T, H, N).log_softmax(1).detach().requires_grad_()
            >>> classify = torch.randn(T, H, N, C).log_softmax(3).detach().requires_grad_()
            >>> targets = torch.randint(1, C, (N, C), dtype=torch.long)
            >>> input_lengths = torch.full((N,), T, dtype=torch.long)
            >>> target_lengths = torch.randint(10, 31, (N,), dtype=torch.long)
            >>> loss = ctc_loss(mask, classify, targets, input_lengths, target_lengths)
            >>> loss.backward()

        Reference:
            2D-CTC for Scene Text Recognition, https://arxiv.org/abs/1907.09705.
        """
        super(CTCLoss2D, self).__init__()
        warnings.warn('NOTICE: This class is only for the useage of understanding the principle of 2D-CTC.Please use `ops.ctc_loss_2d` for practice.')
        self.blank = blank
        self.reduction = reduction
        self.register_buffer('tiny', torch.tensor(torch.finfo().tiny, requires_grad=False))
        self.register_buffer('blank_buffer', torch.tensor([self.blank], dtype=torch.long))
        self.register_buffer('zeros', torch.log(self.tiny))
        self.registered = False

    def expand_with_blank(self, targets):
        N, S = targets.shape
        blank = self.blank_buffer.repeat(targets.shape)
        expanded_targets = torch.cat([blank.unsqueeze(-1), targets.unsqueeze(-1)], -1)
        expanded_targets = expanded_targets.view(N, -1)
        expanded_targets = torch.cat([expanded_targets, blank[:, 0:1]], dim=-1)
        return expanded_targets

    def log_add(self, a, b):
        x, y = torch.max(a, b), torch.min(a, b)
        return x + torch.log1p(torch.exp(y - x))

    def log_sum(self, x, dim, keepdim=False):
        tiny = self.tiny
        return torch.log(torch.max(torch.sum(torch.exp(x), dim=dim, keepdim=keepdim), tiny))

    def safe_log_sum(self, x, keepdim=False):
        result = x[:, (0)]
        for i in range(1, x.size(1)):
            result = self.log_add(result, x[:, (i)])
        if keepdim:
            result = result.unsqueeze(1)
        return result

    def forward(self, mask, classify, targets, input_lengths, target_lengths):
        """
        mask: Tensor of size :math:`(T, H, N)` where `H = height`,
            `T = input length`, and `N = batch size`.
            The logarithmized path transition probabilities.
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        classify: Tensor of size :math:`(T, H, N, C)` where `C = number of classes`, `H = height`, `T = input length`, and `N = batch size`.
            The logarithmized character classification probabilities at all possible path pixels.
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`[NOT IMPLEMENTED YET].
            Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
        """
        device = classify.device
        targets = targets.type(torch.long)
        expanded_targets = self.expand_with_blank(targets)
        N, S = expanded_targets.shape
        T, H = classify.shape[:2]
        targets_indices = expanded_targets.repeat(H, 1, 1)
        tiny = self.tiny
        probability = torch.log((torch.zeros(S, H, N) + tiny) / H)
        probability[0] = classify[(0), :, :, (self.blank)]
        probability[1] = classify[0].gather(2, targets_indices[:, :, 1:2]).permute(2, 0, 1)
        mask_skipping = torch.ne(expanded_targets[:, 2:], expanded_targets[:, :-2]).transpose(0, 1)
        mask_skipping = mask_skipping.unsqueeze(1).type(torch.float)
        mask_not_skipping = 1 - mask_skipping
        length_indices = torch.linspace(0, S - 1, S).repeat(N, 1, 1).transpose(0, 2)
        zeros = self.zeros.repeat(S, 1, N).view(S, 1, N)
        count_computable = torch.cat([mask_skipping[0:1] + 1, mask_skipping[0:1] + 1, mask_skipping + 1], dim=0)
        count_computable = torch.cumsum(count_computable, dim=0)
        for timestep in range(1, T):
            mask_uncomputed = (length_indices > count_computable[timestep]).type(torch.float)
            height_summed = self.log_sum(probability + mask[timestep - 1].unsqueeze(0), dim=1, keepdim=True)
            height_summed = height_summed * (1 - mask_uncomputed) + zeros * mask_uncomputed
            new_probability1 = self.log_add(height_summed[1:], height_summed[:-1])
            new_probability2 = self.log_add(new_probability1[1:], height_summed[:-2]) * mask_skipping + new_probability1[1:] * mask_not_skipping
            new_probability = torch.cat([height_summed[:1], new_probability1[:1], new_probability2], dim=0)
            probability = new_probability + classify[timestep].gather(2, targets_indices).permute(2, 0, 1)
        probability = self.safe_log_sum(probability + mask[T - 1].unsqueeze(0))
        lengths = (target_lengths * 2 + 1).unsqueeze(0)
        loss = self.log_add(probability.gather(0, lengths - 1), probability.gather(0, lengths - 2))
        loss = loss.squeeze(0)
        if self.reduction == 'mean':
            return -(loss / target_lengths.type(torch.float))
        elif self.reduction == 'sum':
            return -loss.sum()
        return -loss


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
            pred = pred[:, (0), :, :]
            gt = gt[:, (0), :, :]
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


class EASTDecoder(nn.Module):

    def __init__(self, channels=256, heatmap_ratio=1.0, densebox_ratio=0.01, densebox_rescale_factor=512):
        nn.Module.__init__(self)
        self.heatmap_ratio = heatmap_ratio
        self.densebox_ratio = densebox_ratio
        self.densebox_rescale_factor = densebox_rescale_factor
        self.head_layer = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=2, stride=2, padding=0))
        self.heatmap_pred_layer = nn.Sequential(nn.Conv2d(channels // 4, 1, kernel_size=1, stride=1, padding=0))
        self.densebox_pred_layer = nn.Sequential(nn.Conv2d(channels // 4, 8, kernel_size=1, stride=1, padding=0))

    def forward(self, input, label, meta, train):
        heatmap = label['heatmap']
        heatmap_weight = label['heatmap_weight']
        densebox = label['densebox']
        densebox_weight = label['densebox_weight']
        feature = self.head_layer(input)
        heatmap_pred = self.heatmap_pred_layer(feature)
        densebox_pred = self.densebox_pred_layer(feature) * self.densebox_rescale_factor
        heatmap_loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap, reduction='none')
        heatmap_loss = (heatmap_loss * heatmap_weight).mean(dim=(1, 2, 3))
        densebox_loss = F.mse_loss(densebox_pred, densebox, reduction='none')
        densebox_loss = (densebox_loss * densebox_weight).mean(dim=(1, 2, 3))
        loss = heatmap_loss * self.heatmap_ratio + densebox_loss * self.densebox_ratio
        pred = {'heatmap': F.sigmoid(heatmap_pred), 'densebox': densebox_pred}
        metrics = {'heatmap_loss': heatmap_loss, 'densebox_loss': densebox_loss}
        if train:
            return loss, pred, metrics
        else:
            return pred


class MaskL1Loss(nn.Module):

    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred[:, (0)] - gt) * mask).sum() / mask.sum()
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
        loss = torch.abs(pred[:, (0)] - gt)
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
            ipdb.set_trace()
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
            g_g = gt[:, (4), :, :]
            g_p, _ = torch.max(pred, 1)
            loss += self.criterion(g_p, g_g, mask)
            return loss
        elif gt_type == 'both':
            pss_loss = self.get_loss(pred[:, :4, :, :], gt[:, :4, :, :], mask)
            g_g = gt[:, (4), :, :]
            g_p, _ = torch.max(pred, 1)
            pss_loss += self.criterion(g_p, g_g, mask)
            shrink_loss = self.criterion(pred[:, (4), :, :], gt[:, (5), :, :], mask)
            return pss_loss, shrink_loss
        else:
            return NotImplementedError('gt_type [%s] is not implemented', gt_type)

    def get_loss(self, pred, gt, mask):
        loss = torch.tensor(0.0)
        for ind in range(pred.size(1)):
            loss += self.criterion(pred[:, (ind), :, :], gt[:, (ind), :, :], mask)
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
        self.binarize = nn.Sequential(nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2), nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
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
        self.thresh = nn.Sequential(nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias), nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True), self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias), nn.Sigmoid())
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
        result = OrderedDict(binary=binary)
        if self.adaptive:
            if self.serial:
                fuse = torch.cat((fuse, nn.functional.interpolate(binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


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
        l1_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
        dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
        metrics['thresh_loss'] = dice_loss
        loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        metrics.update(**l1_metric)
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


def SimpleUpsampleHead(feature_channel, layer_channels):
    modules = []
    modules.append(nn.Conv2d(feature_channel, layer_channels[0], kernel_size=3, stride=1, padding=1, bias=False))
    for layer_index in range(len(layer_channels) - 1):
        modules.extend([nn.BatchNorm2d(layer_channels[layer_index]), nn.ReLU(inplace=True), nn.ConvTranspose2d(layer_channels[layer_index], layer_channels[layer_index + 1], kernel_size=2, stride=2, padding=0, bias=False)])
    return nn.Sequential(*modules)


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


class TextsnakeDecoder(nn.Module):

    def __init__(self, channels=256):
        nn.Module.__init__(self)
        self.head_layer = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2, padding=0), nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=2, stride=2, padding=0))
        self.pred_layer = nn.Sequential(nn.Conv2d(channels // 4, 7, kernel_size=1, stride=1, padding=0))

    @staticmethod
    def ohem(predict, target, train_mask, negative_ratio=3.0):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()
        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        loss_pos = F.cross_entropy(predict, target, reduction='none')[pos].sum()
        loss_neg = F.cross_entropy(predict, target, reduction='none')[neg]
        loss_neg, _ = torch.topk(loss_neg, n_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, label, meta, train):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param label: (dict)
            :param tr_mask: (Variable), TR target, (BS, H, W)
            :param tcl_mask: (Variable), TCL target, (BS, H, W)
            :param sin_map: (Variable), sin target, (BS, H, W)
            :param cos_map: (Variable), cos target, (BS, H, W)
            :param radius_map: (Variable), radius target, (BS, H, W)
            :param train_mask: (Variable), training mask, (BS, H, W)
            :return: loss_tr, loss_tcl, loss_radius, loss_sin, loss_cos
        """
        tr_mask = label['tr_mask']
        tcl_mask = label['tcl_mask']
        sin_map = label['sin_map']
        cos_map = label['cos_map']
        radius_map = label['radius_map']
        train_mask = label['train_mask']
        feature = self.head_layer(input)
        pred = self.pred_layer(feature)
        tr_out = pred[:, :2]
        tcl_out = pred[:, 2:4]
        sin_out = pred[:, (4)]
        cos_out = pred[:, (5)]
        radius_out = pred[:, (6)]
        tr_pred = tr_out.permute(0, 2, 3, 1).reshape(-1, 2)
        tcl_pred = tcl_out.permute(0, 2, 3, 1).reshape(-1, 2)
        sin_pred = sin_out.reshape(-1)
        cos_pred = cos_out.reshape(-1)
        radius_pred = radius_out.reshape(-1)
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale
        train_mask = train_mask.view(-1)
        tr_mask = tr_mask.reshape(-1)
        tcl_mask = tcl_mask.reshape(-1)
        radius_map = radius_map.reshape(-1)
        sin_map = sin_map.reshape(-1)
        cos_map = cos_map.reshape(-1)
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())
        loss_tcl = F.cross_entropy(tcl_pred, tcl_mask.long(), reduction='none')[train_mask * tr_mask].mean()
        ones = radius_map.new(radius_pred[tcl_mask].size()).fill_(1.0).float()
        loss_radius = F.smooth_l1_loss(radius_pred[tcl_mask] / radius_map[tcl_mask], ones)
        loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
        loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])
        loss = loss_tr + loss_tcl + loss_radius + loss_sin + loss_cos
        pred = {'tr_pred': F.softmax(tr_out, dim=1)[:, (1)], 'tcl_pred': F.softmax(tcl_out, dim=1)[:, (1)], 'sin_pred': sin_out, 'cos_pred': cos_out, 'radius_pred': radius_out}
        metrics = {'loss_tr': loss_tr, 'loss_tcl': loss_tcl, 'loss_radius': loss_radius, 'loss_sin': loss_sin, 'loss_cos': loss_cos}
        if train:
            return loss, pred, metrics
        else:
            return pred


class EnsembleModel(nn.Module):

    def __init__(self, models, *args, **kwargs):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleDict(models)

    def forward(self, batch, select_key=None, training=False):
        pred = dict()
        for key, module in self.models.items():
            if select_key is not None and key != select_key:
                continue
            pred[key] = module(batch, training)
        return pred


class BasicModel(nn.Module):

    def __init__(self, args):
        nn.Module.__init__(self)
        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return apex.parallel.DistributedDataParallel(model)
    else:
        return nn.DataParallel(model)


class ClassificationModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return args['backbone'] + '-' + args['decoder']

    def forward(self, batch, training=True):
        data, label = batch
        data = data
        label = label
        if training:
            loss, pred = self.model(data, targets=label, train=True)
            return loss, pred
        else:
            return self.model(data, train=False)


class DetectionModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return args['backbone'] + '-' + args['decoder']

    def forward(self, batch, training=True):
        data, label, meta = batch
        data = data
        data = data.float() / 255.0
        for key, value in label.items():
            label[key] = value
        if training:
            loss, pred, metrics = self.model(data, label, meta, train=True)
            loss = loss.mean()
            return loss, pred, metrics
        else:
            return self.model(data, label, meta, train=False)


class DetectionEnsembleModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.sizes = args['sizes']
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return args['backbone'] + '-' + args['decoder']

    def forward(self, batch, training=True):
        assert not training
        data, label, meta = batch
        data = data
        data = data.float() / 255.0
        for key, value in label.items():
            label[key] = value
        size = data.shape[2], data.shape[3]
        heatmaps = []
        for size0 in self.sizes:
            data0 = F.interpolate(data, size0, mode='bilinear')
            pred0 = self.model(data0, label, meta, train=False)
            heatmap0 = F.interpolate(pred0['heatmap'], size, mode='bilinear')
            heatmaps.append(heatmap0)
        heatmap = sum(heatmaps) / len(heatmaps)
        return {'heatmap': heatmap}


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
        data = batch['image']
        for key, value in batch.items():
            if value is not None:
                if hasattr(value, 'to'):
                    batch[key] = value
        data = data.float()
        pred = self.model(data, training=training)
        if self.training:
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred


class SequenceRecognitionModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        images = batch['image']
        if self.training:
            labels = batch['label']
            lengths = batch['length'].type(torch.long)
            loss, pred = self.model(images, targets=labels, lengths=lengths, train=True)
            return loss, pred
        else:
            return self.model(images, train=False)


class SegRecognitionModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        images = batch['image']
        if self.training:
            mask = batch['mask']
            classify = batch['classify'].type(torch.long)
            return self.model(images, mask=mask, classify=classify, train=True)
        else:
            return self.model(images, train=False)


class IntegralRegressionRecognitionModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        args = dict()
        for key in batch.keys():
            args[key] = batch[key]
        images = args.pop('image')
        return self.model(images, **args)


class GridSamplingModel(nn.Module):

    def __init__(self, args, device, distributed: bool=False, local_rank: int=0):
        nn.Module.__init__(self)
        self.model = parallelize(BasicModel(args), distributed, local_rank)
        self.device = device
        self

    @staticmethod
    def model_name(args):
        return '/' + args['backbone'] + '/' + args['decoder']

    def forward(self, batch, training=True):
        args = dict()
        for key in batch.keys():
            args[key] = batch[key]
        images = args.pop('image')
        return self.model(images, **args)


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
     False),
    (BidirectionalLSTM,
     lambda: ([], {'nIn': 4, 'nHidden': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CRNN,
     lambda: ([], {'imgH': 16, 'nc': 4, 'nclass': 4, 'nh': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (ClassificationDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 4, 256])], {}),
     False),
    (FPNTopDown,
     lambda: ([], {'pyramid_channels': [4, 4], 'feature_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 64, 64])], {}),
     False),
    (LeakyDiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (MaskL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PPMDeepsup,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 2048, 64, 64])], {}),
     False),
    (SimpleDetectionDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (SimpleEASTDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (SimpleMSRDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (SimpleSegDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
    (SimpleTextsnakeDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4]), torch.rand([4, 4]), 0], {}),
     False),
]

class Test_Megvii_CSG_MegReader(_paritybench_base):
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

