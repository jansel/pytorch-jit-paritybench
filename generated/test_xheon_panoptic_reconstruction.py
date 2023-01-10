import sys
_module = sys.modules[__name__]
del sys
lib = _module
config = _module
defaults = _module
paths_catalog = _module
setup = _module
data = _module
build = _module
collate = _module
datasets = _module
front3d = _module
matterport = _module
io = _module
samplers = _module
transforms2d = _module
transforms3d = _module
engine = _module
trainer = _module
layers = _module
batch_norm = _module
misc = _module
nms = _module
roi_align = _module
smooth_l1_loss = _module
metrics = _module
absolute_error = _module
accuracy = _module
instance_intersection_over_union = _module
intersection_over_union = _module
masked_absolute_error = _module
masked_intersection_over_union = _module
masked_scalar = _module
masked_semantic_intersection_over_union = _module
mean_average_precision = _module
metric = _module
panoptic_quality = _module
panoptic_reconstruction_quality = _module
scalar = _module
semantic_intersection_over_union = _module
voxel_accuracy = _module
modeling = _module
backbone = _module
make_layers = _module
model_serialization = _module
model_zoo = _module
multitask_heads_sparse = _module
resnet = _module
resnet3d = _module
resnet_encoder = _module
resnet_fb = _module
resnet_sparse = _module
unet_sparse = _module
utils = _module
depth = _module
depth_prediction = _module
sobel = _module
detector = _module
balanced_positive_negative_sampler = _module
box_coder = _module
generalized_rcnn = _module
matcher = _module
poolers = _module
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
inference = _module
loss = _module
rpn = _module
utils = _module
frustum = _module
frustum_completion = _module
post_process = _module
model_serialization = _module
panoptic_reconstruction = _module
projection = _module
sparse_projection = _module
utils = _module
solver = _module
lr_scheduler = _module
structures = _module
bounding_box = _module
boxlist_ops = _module
depth_map = _module
field_list = _module
frustum = _module
segmentation_mask = _module
c2_model_loading = _module
checkpoint = _module
cv2_util = _module
debugger = _module
environment = _module
imports = _module
intrinsics = _module
logger = _module
metric_logger = _module
registry = _module
transform = _module
visualize = _module
helpers = _module
image = _module
io = _module
mesh = _module
pointcloud = _module
fix_checkpoint_names = _module
tools = _module
evaluate_net = _module
test_net_single_image = _module
train_net = _module

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


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils import data


import random


from typing import Dict


from typing import Union


from typing import List


from typing import Tuple


import numpy as np


import torch.utils.data


from torchvision.transforms import ColorJitter


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from torchvision import transforms as T


from typing import Optional


from typing import Any


from torch.nn import functional as F


from scipy import ndimage


from scipy.spatial.transform.rotation import Rotation as R


import time


from collections import OrderedDict


from torch import nn


import math


from torch.nn.modules.utils import _ntuple


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from collections import defaultdict


from collections import namedtuple


import logging


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


from matplotlib import pyplot as plt


import copy


from torch.utils.collect_env import get_pretty_env_info


from collections import deque


from matplotlib import patches


from matplotlib.figure import Figure


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


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Linear(torch.nn.Linear):

    def forward(self, x):
        if x.numel() > 0:
            return super(Linear, self).forward(x)
        output_shape = [x.shape[0], self.out_features]
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv1d(torch.nn.Conv1d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv1d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-1:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv2d(torch.nn.Conv2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv3d(torch.nn.Conv3d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv3d, self).forward(x)
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-3:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        output_shape = [((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op) for i, p, di, k, d, op in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride, self.output_padding)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose3d(torch.nn.ConvTranspose3d):

    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose3d, self).forward(x)
        output_shape = [((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op) for i, p, di, k, d, op in zip(x.shape[-3:], self.padding, self.dilation, self.kernel_size, self.stride, self.output_padding)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm2d(torch.nn.BatchNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm3d(torch.nn.BatchNorm3d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm3d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm1d(torch.nn.BatchNorm1d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm1d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class InstanceNorm1d(torch.nn.InstanceNorm1d):

    def forward(self, x):
        if x.numel() > 0:
            return super(InstanceNorm1d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class InstanceNorm2d(torch.nn.InstanceNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(InstanceNorm2d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class InstanceNorm3d(torch.nn.InstanceNorm3d):

    def forward(self, x):
        if x.numel() > 0:
            return super(InstanceNorm3d, self).forward(x)
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class DFConv2d(nn.Module):
    """Deformable convolutional layer"""

    def __init__(self, in_channels, out_channels, with_modulated_dcn=True, kernel_size=3, stride=1, groups=1, dilation=1, deformable_groups=1, bias=False):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2
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
        self.offset = Conv2d(in_channels, deformable_groups * offset_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, dilation=dilation)
        for l in [self.offset]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.0)
        self.conv = conv_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class _ROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(grad_output, rois, spatial_scale, output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ProxyCompletionHeadSparse(nn.Module):

    def __init__(self, channel_in: int, channel_out: int, truncation: int) ->None:
        super().__init__()
        self.truncation = truncation
        self.network = nn.Sequential(Me.MinkowskiInstanceNorm(channel_in), Me.MinkowskiReLU(), Me.MinkowskiLinear(channel_in, channel_out))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        output = self.network(x)
        return output


class GeometryHeadSparse(nn.Module):

    def __init__(self, channel_in: int, channel_out: int, truncation: int, num_blocks: int) ->None:
        super().__init__()
        self.truncation = truncation
        self.network = [Me.MinkowskiInstanceNorm(channel_in), Me.MinkowskiReLU()]
        for _ in range(num_blocks):
            self.network.append(SparseBasicBlock(channel_in, channel_in, dimension=3))
        self.network.extend([Me.MinkowskiConvolution(channel_in, channel_out, kernel_size=3, stride=1, bias=True, dimension=3)])
        self.network = nn.Sequential(*self.network)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        output = self.network(x)
        return output


class ClassificationHeadSparse(nn.Module):

    def __init__(self, channel_in: int, channel_out: int, num_blocks: int) ->None:
        super().__init__()
        self.network = [Me.MinkowskiInstanceNorm(channel_in), Me.MinkowskiReLU()]
        for _ in range(num_blocks):
            self.network.append(SparseBasicBlock(channel_in, channel_in, dimension=3))
        self.network.extend([Me.MinkowskiConvolution(channel_in, channel_out, kernel_size=3, stride=1, bias=True, dimension=3)])
        self.network = nn.Sequential(*self.network)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.network(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0
        self.conv1 = Me.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = Me.MinkowskiInstanceNorm(planes)
        self.conv2 = Me.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = Me.MinkowskiInstanceNorm(planes)
        self.relu = Me.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = Me.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = Me.MinkowskiInstanceNorm(planes)
        self.conv2 = Me.MinkowskiConvolution(planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = Me.MinkowskiInstanceNorm(planes)
        self.conv3 = Me.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = Me.MinkowskiInstanceNorm(planes * self.expansion)
        self.relu = Me.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


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


StageSpec = namedtuple('StageSpec', ['index', 'block_count', 'return_features'])


ResNet101FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True)))


ResNet101StagesTo4 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 23, True)))


ResNet101StagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True)))


ResNet152FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True)))


ResNet50FPNStagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True)))


ResNet50StagesTo4 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 6, True)))


ResNet50StagesTo5 = tuple(StageSpec(index=i, block_count=c, return_features=r) for i, c, r in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True)))


_STAGE_SPECS = Registry({'R-50-C4': ResNet50StagesTo4, 'R-50-C5': ResNet50StagesTo5, 'R-101-C4': ResNet101StagesTo4, 'R-101-C5': ResNet101StagesTo5, 'R-50-FPN': ResNet50FPNStagesTo5, 'R-50-FPN-RETINANET': ResNet50FPNStagesTo5, 'R-101-FPN': ResNet101FPNStagesTo5, 'R-101-FPN-RETINANET': ResNet101FPNStagesTo5, 'R-152-FPN': ResNet152FPNStagesTo5})


class BaseStem(nn.Module):

    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_func(out_channels)
        for l in [self.conv1]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class StemWithFixedBatchNorm(BaseStem):

    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(cfg, norm_func=FrozenBatchNorm2d)


_STEM_MODULES = Registry({'StemWithFixedBatchNorm': StemWithFixedBatchNorm, 'BaseStem': BaseStem})


class BottleneckWithFixedBatchNorm(Bottleneck):

    def __init__(self, in_channels, bottleneck_channels, out_channels, num_groups=1, stride_in_1x1=True, stride=1, dilation=1, dcn_config={}):
        super(BottleneckWithFixedBatchNorm, self).__init__(in_channels=in_channels, bottleneck_channels=bottleneck_channels, out_channels=out_channels, num_groups=num_groups, stride_in_1x1=stride_in_1x1, stride=stride, dilation=dilation, norm_func=FrozenBatchNorm2d, dcn_config=dcn_config)


_TRANSFORMATION_MODULES = Registry({'BottleneckWithFixedBatchNorm': BottleneckWithFixedBatchNorm, 'Bottleneck': Bottleneck})


def _make_stage(transformation_module, in_channels, bottleneck_channels, out_channels, block_count, num_groups, stride_in_1x1, first_stride, dilation=1, dcn_config={}):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(transformation_module(in_channels, bottleneck_channels, out_channels, num_groups, stride_in_1x1, stride, dilation=dilation, dcn_config=dcn_config))
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class ResNet(nn.Module):

    def __init__(self, cfg):
        super(ResNet, self).__init__()
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
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
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index - 1]
            module = _make_stage(transformation_module, in_channels, bottleneck_channels, out_channels, stage_spec.block_count, num_groups, cfg.MODEL.RESNETS.STRIDE_IN_1X1, first_stride=int(stage_spec.index > 1) + 1, dcn_config={'stage_with_dcn': stage_with_dcn, 'with_modulated_dcn': cfg.MODEL.RESNETS.WITH_MODULATED_DCN, 'deformable_groups': cfg.MODEL.RESNETS.DEFORMABLE_GROUPS})
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


class ResNetBlock3d(nn.Module):

    def __init__(self, in_features, out_features, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_features, out_features, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_features, out_features, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_features)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetDownsample(nn.Module):

    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_features, out_features, 1, stride, bias=False)
        self.norm = nn.InstanceNorm3d(out_features)

    def forward(self, x):
        return self.norm(self.conv(x))


class ResNetBlockTranspose3d(nn.Module):

    def __init__(self, in_features, out_features, stride=1, upsample=None):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_features, out_features, 3, stride, 1, 1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_features, out_features, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_features)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetTranspose(nn.Module):

    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_features, out_features, 3, stride, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


ModuleResult = Tuple[Dict, Dict]


class ResNetEncoder(nn.Module):

    def __init__(self, backbone: nn.Module) ->None:
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = [self.layer1[-1].out_channels, self.layer2[-1].out_channels, self.layer3[-1].out_channels, self.layer4[-1].out_channels]

    def forward(self, x: torch.Tensor) ->ModuleResult:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)
        return {}, {'blocks': [x_block1, x_block2, x_block3, x_block4]}

    def test(self, x: torch.Tensor) ->Dict:
        _, result = self.forward(x)
        return result


class ResNetHead(nn.Module):

    def __init__(self, block_module, stages, num_groups=1, width_per_group=64, stride_in_1x1=True, stride_init=None, res2_out_channels=256, dilation=1, dcn_config={}):
        super(ResNetHead, self).__init__()
        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
        block_module = _TRANSFORMATION_MODULES[block_module]
        self.stages = []
        stride = stride_init
        for stage in stages:
            name = 'layer' + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(block_module, in_channels, bottleneck_channels, out_channels, stage.block_count, num_groups, stride_in_1x1, first_stride=stride, dilation=dilation, dcn_config=dcn_config)
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


logger = logging.getLogger('trainer')


class DepthMap(object):

    def __init__(self, depth_map=None, intrinsic_matrix=None):
        if isinstance(depth_map, str):
            depth_map = torch.from_numpy(np.array(Image.open(depth_map))).float() / 1000.0
        self.depth_map = depth_map
        self.intrinsic_matrix = intrinsic_matrix

    def load_from(self, filename):
        depth_image = torch.from_numpy(np.array(Image.open(filename))).float()
        self.depth_map = depth_image / 1000.0

    def get_tensor(self):
        return self.depth_map.clone()

    def set_intrinsic(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix

    def get_intrinsic(self):
        return self.intrinsic_matrix.clone()

    def save(self, filename):
        plt.imsave(filename, self.depth_map.numpy(), cmap='rainbow')

    def mask_out(self, mask):
        self.depth_map = self.depth_map * mask

    def to_pointcloud(self, filename):
        pointcloud, _ = self.compute_pointcloud()
        write_pointcloud(pointcloud, None, filename)

    def to_pointcloud_with_colors(self, colors, filename):
        pointcloud, coords = self.compute_pointcloud()
        color_values = colors[coords[:, 0], coords[:, 1]]
        write_pointcloud(pointcloud, color_values, filename)

    def compute_pointcloud(self):
        coords2d = self.depth_map.nonzero(as_tuple=False)
        depth_map = self.depth_map[coords2d[:, 0], coords2d[:, 1]].reshape(-1)
        yv = coords2d[:, 0].reshape(-1).float() * depth_map.float()
        xv = coords2d[:, 1].reshape(-1).float() * depth_map.float()
        coords3d = torch.stack([xv, yv, depth_map.float(), torch.ones_like(depth_map).float()])
        pointcloud = torch.mm(torch.inverse(self.intrinsic_matrix.float()), coords3d.float()).t()[:, :3]
        return pointcloud, coords2d

    def compute_normal(self):
        width = self.depth_map.shape[1]
        height = self.depth_map.shape[0]
        depth_map = self.depth_map.reshape(-1).float()
        yv, xv = torch.meshgrid([torch.arange(height), torch.arange(width)])
        yv = yv.reshape(-1).float() * depth_map.float()
        xv = xv.reshape(-1).float() * depth_map.float()
        coords3d = torch.stack([xv, yv, depth_map.float(), torch.ones_like(depth_map).float()])
        pointcloud = torch.mm(torch.inverse(self.intrinsic_matrix.float()), coords3d.float()).t()[:, :3]
        """
           MC
        CM-CC-CP
           PC
        """
        output_normals = torch.zeros((3, height, width))
        y, x = torch.meshgrid([torch.arange(1, height - 1), torch.arange(1, width - 1)])
        y = y.reshape(-1)
        x = x.reshape(-1)
        CC = pointcloud[(y + 0) * width + (x + 0)]
        PC = pointcloud[(y + 1) * width + (x + 0)]
        CP = pointcloud[(y + 0) * width + (x + 1)]
        MC = pointcloud[(y - 1) * width + (x + 0)]
        CM = pointcloud[(y + 0) * width + (x - 1)]
        n = torch.cross(PC - MC, CP - CM).transpose(1, 0)
        l = torch.norm(n, dim=0)
        output_normals[:, y, x] = n / -l
        zeros = (self.depth_map == 0).nonzero()
        output_normals[:, zeros[:, 0], zeros[:, 1]] = 0
        zeros_height_lower = zeros.clone()
        zeros_height_lower[:, 0] -= 1
        zeros_height_lower[:, 0] = torch.clamp(zeros_height_lower[:, 0], min=0, max=height - 1)
        output_normals[:, zeros_height_lower[:, 0], zeros_height_lower[:, 1]] = 0
        zeros_height_upper = zeros.clone()
        zeros_height_upper[:, 0] += 1
        zeros_height_upper[:, 0] = torch.clamp(zeros_height_upper[:, 0], min=0, max=height - 1)
        output_normals[:, zeros_height_upper[:, 0], zeros_height_upper[:, 1]] = 0
        zeros_width_lower = zeros.clone()
        zeros_width_lower[:, 1] -= 1
        zeros_width_lower[:, 1] = torch.clamp(zeros_width_lower[:, 1], min=0, max=width - 1)
        output_normals[:, zeros_width_lower[:, 0], zeros_width_lower[:, 1]] = 0
        zeros_width_upper = zeros.clone()
        zeros_width_upper[:, 1] += 1
        zeros_width_upper[:, 1] = torch.clamp(zeros_width_upper[:, 1], min=0, max=width - 1)
        output_normals[:, zeros_width_upper[:, 0], zeros_width_upper[:, 1]] = 0
        return output_normals


class _UpProjection(nn.Module):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm_func(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = norm_func(num_output_features)
        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = norm_func(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out


class D(nn.Module):

    def __init__(self, num_features=2048):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = norm_func(num_features)
        self.up1 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        return x_d4


class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d
        self.up1 = _UpProjection(num_input_features=block_channel[0], num_output_features=16)
        self.up2 = _UpProjection(num_input_features=block_channel[1], num_output_features=16)
        self.up3 = _UpProjection(num_input_features=block_channel[2], num_output_features=16)
        self.up4 = _UpProjection(num_input_features=block_channel[3], num_output_features=16)
        self.conv = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = norm_func(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)
        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)
        return x


class DepthPredictionBackbone(nn.Module):

    def __init__(self, num_features, block_channel):
        super().__init__()
        self.D = D(num_features)
        self.MFF = MFF(block_channel)
        self.R = R(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = x
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))
        return out, torch.cat((x_decoder, x_mff), 1)


class Sobel(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))
        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out


class DepthPrediction(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        block_channel = self.get_block_channel_list()
        self.model = DepthPredictionBackbone(num_features=block_channel[-1], block_channel=block_channel)
        self.cos_loss = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel()
        self.criterionL1 = F.l1_loss

    @staticmethod
    def get_block_channel_list():
        block_channel_map = {'50': [256, 512, 1024, 2048], '34': [64, 128, 256, 512], '18': [64, 128, 256, 512]}
        identifier = config.MODEL.BACKBONE.CONV_BODY.split('-')[1]
        block_channels = block_channel_map[identifier]
        return block_channels

    def forward(self, features, depth_target) ->ModuleResult:
        depth_pred, depth_feature = self.model(features)
        depth_return = [DepthMap(p_[0].cpu(), t_.get_intrinsic()) for p_, t_ in zip(depth_pred, depth_target)]
        depth_target = torch.stack([target.get_tensor() for target in depth_target]).float().unsqueeze(1)
        results = {'prediction': depth_pred, 'return': depth_return, 'features': depth_feature}
        losses = {}
        if self.training:
            valid_masks = torch.stack([(depth.depth_map != 0.0).bool() for depth in depth_target], dim=0)
            valid_masks.unsqueeze_(1)
            grad_target = self.get_gradient(depth_target)
            grad_pred = self.get_gradient(depth_pred)
            grad_target_dx = grad_target[:, 0, :, :].contiguous().view_as(depth_target)
            grad_target_dy = grad_target[:, 1, :, :].contiguous().view_as(depth_target)
            grad_pred_dx = grad_pred[:, 0, :, :].contiguous().view_as(depth_target)
            grad_pred_dy = grad_pred[:, 1, :, :].contiguous().view_as(depth_target)
            ones = torch.ones(depth_target.size(0), 1, depth_target.size(2), depth_target.size(3)).float()
            normal_target = torch.cat((-grad_target_dx, -grad_target_dy, ones), 1)
            normal_pred = torch.cat((-grad_pred_dx, -grad_pred_dy, ones), 1)
            loss_depth = torch.log(torch.abs(depth_target - depth_pred) + 0.5)[valid_masks].mean()
            loss_dx = torch.log(torch.abs(grad_target_dx - grad_pred_dx) + 0.5)[valid_masks].mean()
            loss_dy = torch.log(torch.abs(grad_target_dy - grad_pred_dy) + 0.5)[valid_masks].mean()
            loss_gradient = loss_dx + loss_dy
            loss_normal = torch.abs(1 - self.cos_loss(normal_pred, normal_target))[valid_masks.squeeze(1)].mean()
            loss_weight = config.MODEL.DEPTH2D.LOSS_WEIGHT
            losses = {'depth': loss_weight * loss_depth, 'normal': loss_weight * loss_normal, 'gradient': loss_weight * loss_gradient}
        return losses, results

    def inference(self, features):
        depth_pred, depth_feature = self.model(features)
        return depth_pred, depth_feature


class R(nn.Module):

    def __init__(self, block_channel):
        super().__init__()
        norm_func = FrozenBatchNorm2d if config.MODEL.FIXNORM else nn.BatchNorm2d
        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = norm_func(num_features)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = norm_func(num_features)
        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        return x2


class FieldList:
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, image_size, mode='xyxy'):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.size = image_size
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def update(self, dictionary):
        for k, v in dictionary.items():
            self.extra_fields[k] = v

    def copy_with_fields(self, fields, skip_missing=False):
        field_list = FieldList(self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                field_list.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return field_list

    def __len__(self):
        return len(self.extra_fields)

    def __getitem__(self, item):
        field_list = FieldList(self.size, self.mode)
        for k, v in self.extra_fields.items():
            field_list.add_field(k, v[item])
        return field_list

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


FLIP_TOP_BOTTOM = 1


FLIP_LEFT_RIGHT = 0


class PolygonInstance:
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size):
        """
            Arguments:
                a list of lists of numbers.
                The first level refers to all the polygons that compose the
                object, and the second level to the polygon coordinates.
        """
        if isinstance(polygons, (list, tuple)):
            valid_polygons = []
            for p in polygons:
                p = torch.as_tensor(p, dtype=torch.float32)
                if len(p) >= 6:
                    valid_polygons.append(p)
            polygons = valid_polygons
        elif isinstance(polygons, PolygonInstance):
            polygons = copy.copy(polygons.polygons)
        else:
            RuntimeError('Type of argument `polygons` is not allowed:%s' % type(polygons))
        """ This crashes the training way too many times...
        for p in polygons:
            assert p[::2].min() >= 0
            assert p[::2].max() < size[0]
            assert p[1::2].min() >= 0
            assert p[1::2].max() , size[1]
        """
        self.polygons = polygons
        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1
        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)
        return PolygonInstance(flipped_polygons, size=self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = map(float, box)
        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)
        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)
        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)
        w, h = xmax - xmin, ymax - ymin
        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - xmin
            p[1::2] = p[1::2] - ymin
            cropped_polygons.append(p)
        return PolygonInstance(cropped_polygons, size=(w, h))

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [(p * ratio) for p in self.polygons]
            return PolygonInstance(scaled_polys, size)
        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)
        return PolygonInstance(scaled_polygons, size=size)

    def convert_to_binarymask(self):
        width, height = self.size
        polygons = [p.numpy() for p in self.polygons]
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        mask = torch.from_numpy(mask)
        return mask

    def __len__(self):
        return len(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_groups={}, '.format(len(self.polygons))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


class PolygonList:
    """
    This class handles PolygonInstances for all objects in the image
    """

    def __init__(self, polygons, size):
        """
        Arguments:
            polygons:
                a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.

                OR

                a list of PolygonInstances.

                OR

                a PolygonList

            size: absolute image size

        """
        if isinstance(polygons, (list, tuple)):
            if len(polygons) == 0:
                polygons = [[[]]]
            if isinstance(polygons[0], (list, tuple)):
                assert isinstance(polygons[0][0], (list, tuple)), str(type(polygons[0][0]))
            else:
                assert isinstance(polygons[0], PolygonInstance), str(type(polygons[0]))
        elif isinstance(polygons, PolygonList):
            size = polygons.size
            polygons = polygons.polygons
        else:
            RuntimeError('Type of argument `polygons` is not allowed:%s' % type(polygons))
        assert isinstance(size, (list, tuple)), str(type(size))
        self.polygons = []
        for p in polygons:
            p = PolygonInstance(p, size)
            if len(p) > 0:
                self.polygons.append(p)
        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
        flipped_polygons = []
        for polygon in self.polygons:
            flipped_polygons.append(polygon.transpose(method))
        return PolygonList(flipped_polygons, size=self.size)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_polygons = []
        for polygon in self.polygons:
            cropped_polygons.append(polygon.crop(box))
        cropped_size = w, h
        return PolygonList(cropped_polygons, cropped_size)

    def resize(self, size):
        resized_polygons = []
        for polygon in self.polygons:
            resized_polygons.append(polygon.resize(size))
        resized_size = size
        return PolygonList(resized_polygons, resized_size)

    def to(self, *args, **kwargs):
        return self

    def convert_to_binarymask(self):
        if len(self) > 0:
            masks = torch.stack([p.convert_to_binarymask() for p in self.polygons])
        else:
            size = self.size
            masks = torch.empty([0, size[1], size[0]], dtype=torch.uint8)
        return BinaryMaskList(masks, size=self.size)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, item):
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            selected_polygons = []
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return PolygonList(selected_polygons, size=self.size)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.polygons))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple) and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]
    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)


class BinaryMaskList:
    """
    This class handles binary masks for all objects in the image
    """

    def __init__(self, masks, size):
        """
            Arguments:
                masks: Either torch.tensor of [num_instances, H, W]
                    or list of torch.tensors of [H, W] with num_instances elems,
                    or RLE (Run Length Encoding) - interpreted as list of dicts,
                    or BinaryMaskList.
                size: absolute image size, width first

            After initialization, a hard copy will be made, to leave the
            initializing source data intact.
        """
        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        if isinstance(masks, torch.Tensor):
            masks = masks.clone()
        elif isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                masks = torch.empty([0, size[1], size[0]])
            elif isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).clone()
            elif isinstance(masks[0], dict) and 'counts' in masks[0]:
                rle_sizes = [tuple(inst['size']) for inst in masks]
                masks = mask_utils.decode(masks)
                masks = torch.tensor(masks).permute(2, 0, 1)
                assert rle_sizes.count(rle_sizes[0]) == len(rle_sizes), 'All the sizes must be the same size: %s' % rle_sizes
                rle_height, rle_width = rle_sizes[0]
                assert masks.shape[1] == rle_height
                assert masks.shape[2] == rle_width
                width, height = size
                if width != rle_width or height != rle_height:
                    masks = interpolate(input=masks[None].float(), size=(height, width), mode='bilinear', align_corners=False)[0].type_as(masks)
            else:
                RuntimeError('Type of `masks[0]` could not be interpreted: %s' % type(masks))
        elif isinstance(masks, BinaryMaskList):
            masks = masks.masks.clone()
        else:
            RuntimeError('Type of `masks` argument could not be interpreted:%s' % type(masks))
        if len(masks.shape) == 2:
            masks = masks[None]
        assert len(masks.shape) == 3
        assert masks.shape[1] == size[1], '%s != %s' % (masks.shape[1], size[1])
        assert masks.shape[2] == size[0], '%s != %s' % (masks.shape[2], size[0])
        self.masks = masks
        self.size = tuple(size)

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = self.masks.flip(dim)
        return BinaryMaskList(flipped_masks, self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]
        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)
        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)
        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)
        width, height = xmax - xmin, ymax - ymin
        cropped_masks = self.masks[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return BinaryMaskList(cropped_masks, cropped_size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)
        assert width > 0
        assert height > 0
        resized_masks = interpolate(input=self.masks[None].float(), size=(height, width), mode='bilinear', align_corners=False)[0].type_as(self.masks)
        resized_size = width, height
        return BinaryMaskList(resized_masks, resized_size)

    def convert_to_polygon(self):
        if self.masks.numel() == 0:
            return PolygonList([], self.size)
        contours = self._findContours()
        return PolygonList(contours, self.size)

    def to(self, *args, **kwargs):
        return self

    def _findContours(self):
        contours = []
        masks = self.masks.detach().numpy()
        for mask in masks:
            mask = cv2.UMat(mask)
            contour, hierarchy = cv2_util.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            reshaped_contour = []
            for entity in contour:
                assert len(entity.shape) == 3
                assert entity.shape[1] == 1, 'Hierarchical contours are not allowed'
                reshaped_contour.append(entity.reshape(-1).tolist())
            contours.append(reshaped_contour)
        return contours

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        if self.masks.numel() == 0:
            raise RuntimeError('Indexing empty BinaryMaskList')
        return BinaryMaskList(self.masks[index], self.size)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.masks))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


class SegmentationMask:
    """
    This class stores the segmentations for all objects in the image.
    It wraps BinaryMaskList and PolygonList conveniently.
    """

    def __init__(self, instances, size, mode='poly'):
        """
        Arguments:
            instances: two types
                (1) polygon
                (2) binary mask
            size: (width, height)
            mode: 'poly', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """
        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        if isinstance(size[0], torch.Tensor):
            assert isinstance(size[1], torch.Tensor)
            size = size[0].item(), size[1].item()
        assert isinstance(size[0], (int, float))
        assert isinstance(size[1], (int, float))
        if mode == 'poly':
            self.instances = PolygonList(instances, size)
        elif mode == 'mask':
            self.instances = BinaryMaskList(instances, size)
        else:
            raise NotImplementedError('Unknown mode: %s' % str(mode))
        self.mode = mode
        self.size = tuple(size)

    def transpose(self, method):
        flipped_instances = self.instances.transpose(method)
        return SegmentationMask(flipped_instances, self.size, self.mode)

    def crop(self, box):
        cropped_instances = self.instances.crop(box)
        cropped_size = cropped_instances.size
        return SegmentationMask(cropped_instances, cropped_size, self.mode)

    def resize(self, size, *args, **kwargs):
        resized_instances = self.instances.resize(size)
        resized_size = size
        return SegmentationMask(resized_instances, resized_size, self.mode)

    def to(self, *args, **kwargs):
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self
        if mode == 'poly':
            converted_instances = self.instances.convert_to_polygon()
        elif mode == 'mask':
            converted_instances = self.instances.convert_to_binarymask()
        else:
            raise NotImplementedError('Unknown mode: %s' % str(mode))
        return SegmentationMask(converted_instances, self.size, mode)

    def get_mask_tensor(self):
        instances = self.instances
        if self.mode == 'poly':
            instances = instances.convert_to_binarymask()
        return instances.masks.squeeze(0)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        selected_instances = self.instances.__getitem__(item)
        return SegmentationMask(selected_instances, self.size, self.mode)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_segmentation = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_segmentation
        raise StopIteration()
    next = __next__

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.instances))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


class GeneralizedRCNN(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        output_channel = in_channels[2]
        self.rpn = rpn.build_rpn(output_channel)
        self.roi_heads = None
        if config.MODEL.INSTANCE2D.ROI_HEADS.USE:
            self.roi_heads = roi_heads.build_roi_heads(output_channel)
        self.matching_overlap_threshold = 0.5

    def forward(self, features: Dict, targets: List[FieldList]=None, is_validate: bool=False) ->ModuleResult:
        if is_validate:
            self.train()
        losses = {}
        bounding_boxes_gt = [target.get_field('instance2d') for target in targets]
        features = features['blocks'][2:3]
        results_detection, losses_rpn = self.rpn(features, bounding_boxes_gt)
        losses.update(losses_rpn)
        if config.MODEL.INSTANCE2D.GT_PROPOSAL:
            results_detection = [target.copy_with_fields([]) for target in bounding_boxes_gt]
            for result_item in results_detection:
                result_item.add_field('objectness', torch.ones(len(result_item.bbox)))
        if self.roi_heads:
            results_detection, losses_roi = self.roi_heads(features, results_detection, bounding_boxes_gt)
            losses.update(losses_roi)
        if is_validate or self.training:
            score_key = 'objectness'
        else:
            score_key = 'scores2d'
        boxes, masks, raws, locations = self.match_process(results_detection, bounding_boxes_gt)
        results = {'boxes': [box.bbox for box in boxes], 'masks': [mask.get_mask_tensor() for mask in masks], 'raw': [raw for raw in raws], 'locations': [(location - 1) for location in locations], 'label': [box.get_field('label') for box in boxes]}
        if boxes[0].has_field(score_key):
            results[f'{score_key}'] = [box.get_field(score_key) for box in boxes]
        for name, loss in losses.items():
            losses[name] = config.MODEL.INSTANCE2D.LOSS_WEIGHT * loss
        return losses, results

    def match_process(self, predictions, targets):
        boxes_matched = []
        instance_matched = []
        raw_matched = []
        instance_locations_matched = []
        for proposals_per_image, targets_per_image in zip(predictions, targets):
            if len(proposals_per_image) > 0:
                matched_proposals = self.match_proposals_to_targets(proposals_per_image, targets_per_image)
                boxes = matched_proposals
                boxes_matched.append(boxes)
                segmentation_masks = SegmentationMask(matched_proposals.get_field('mask2d'), targets_per_image.size, mode='mask')
                instance_matched.append(segmentation_masks)
                raw_matched.append(matched_proposals.get_field('mask2draw'))
                instance_locations = matched_proposals.get_field('instance_locations')
                instance_locations_matched.append(instance_locations)
            else:
                boxes = proposals_per_image[[]]
                boxes_matched.append(boxes)
                segmentation_masks = targets_per_image.get_field('mask2d')[[]]
                instance_matched.append(segmentation_masks)
                raw_matched.append(segmentation_masks.get_mask_tensor())
                locations = targets_per_image.get_field('mask2dInstance')[[]]
                instance_locations_matched.append(locations)
        return boxes_matched, instance_matched, raw_matched, instance_locations_matched

    def match_proposals_to_targets(self, proposals, targets):
        locations = []
        matched_proposal_indices = []
        for target_mask, target_location in zip(targets.get_field('mask2d'), targets.get_field('mask2d_instance')):
            for proposal_index, proposal_mask in enumerate(proposals.get_field('mask2d')):
                if proposal_index not in matched_proposal_indices:
                    overlap = intersection_over_union.compute_iou(proposal_mask, target_mask.get_mask_tensor())
                    if overlap > self.matching_overlap_threshold:
                        locations.append(target_location)
                        matched_proposal_indices.append(proposal_index)
                        break
        matched_proposals = proposals[matched_proposal_indices]
        matched_proposals.add_field('instance_locations', torch.tensor(locations, dtype=torch.long))
        return matched_proposals

    def inference(self, features):
        features = features['blocks'][2:3]
        self.eval()
        rpn_results, _ = self.rpn.inference(features)
        detection_results = self.roi_heads.inference(features, rpn_results)
        boxes, masks, raws, locations = self.filter_detections(detection_results)
        results = {'boxes': [box.bbox for box in boxes], 'masks': [mask.get_mask_tensor() for mask in masks], 'raw': [raw for raw in raws], 'locations': [(location - 1) for location in locations], 'label': [box.get_field('label') for box in boxes]}
        return results

    def filter_detections(self, detections):
        boxes = []
        masks = []
        raws = []
        locations = []
        score_threshold = config.MODEL.INSTANCE2D.RPN.SCORE_THRESH
        for proposals in detections:
            scores = proposals.get_field('scores2d')
            mask = torch.zeros_like(scores).bool()
            for instance_id, score in enumerate(scores):
                if score > score_threshold:
                    mask[instance_id] = True
            filtered_proposals = proposals[mask]
            boxes.append(filtered_proposals)
            masks.append(SegmentationMask(filtered_proposals.get_field('mask2d'), proposals.size, mode='mask'))
            raws.append(filtered_proposals.get_field('mask2draw'))
            locations.append(torch.arange(len(scores)) + 1)
        return boxes, masks, raws, locations


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=3, eps=1e-06):
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
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super().__init__()
        poolers = []
        for scale in scales:
            poolers.append(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio))
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat([torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes)], dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)
        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, output_size, output_size), dtype=dtype, device=device)
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level, as_tuple=False).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)
        return result


class ResNet50Conv5ROIFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        resolution = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        if config.MODEL.BACKBONE.CONV_BODY == 'R-50':
            stage = resnet.StageSpec(index=4, block_count=6, return_features=False)
        elif config.MODEL.BACKBONE.CONV_BODY == 'R-18':
            stage = resnet.StageSpec(index=2, block_count=3, return_features=False)
        if config.MODEL.FIXNORM:
            block_module = 'BottleneckWithFixedBatchNorm'
        else:
            block_module = 'Bottleneck'
        head = resnet.ResNetHead(block_module=block_module, stages=(stage,), num_groups=1, width_per_group=64, stride_in_1x1=True, stride_init=None, res2_out_channels=256, dilation=1)
        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


def make_roi_box_feature_extractor():
    return ResNet50Conv5ROIFeatureExtractor()


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
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
            positive = torch.nonzero(matched_idxs_per_image >= 1, as_tuple=False).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0, as_tuple=False).squeeze(1)
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


class BoxCoder:
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
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes
        TO_REMOVE = 1
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        return pred_boxes

    def encode_z(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, 1] - proposals[:, 0] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        gt_widths = reference_boxes[:, 1] - reference_boxes[:, 0] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets = torch.stack((targets_dx, targets_dw), dim=1)
        return targets

    def decode_z(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes
        TO_REMOVE = 1
        widths = boxes[:, 1] - boxes[:, 0] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::2] / wx
        dw = rel_codes[:, 1::2] / ww
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::2] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::2] = pred_ctr_x + 0.5 * pred_w - 1
        return pred_boxes

    def encode_xyz(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, 3] - proposals[:, 0] + TO_REMOVE
        ex_lengths = proposals[:, 1] - proposals[:, 4] + TO_REMOVE
        ex_heights = proposals[:, 5] - proposals[:, 2] + TO_REMOVE
        ex_widths = ex_widths.clamp(min=0.01)
        ex_lengths = ex_lengths.clamp(min=0.01)
        ex_heights = ex_heights.clamp(min=0.01)
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 4] + 0.5 * ex_lengths
        ex_ctr_z = proposals[:, 2] + 0.5 * ex_heights
        gt_widths = reference_boxes[:, 3] - reference_boxes[:, 0] + TO_REMOVE
        gt_lengths = reference_boxes[:, 1] - reference_boxes[:, 4] + TO_REMOVE
        gt_heights = reference_boxes[:, 5] - reference_boxes[:, 2] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 4] + 0.5 * gt_lengths
        gt_ctr_z = reference_boxes[:, 2] + 0.5 * gt_heights
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wx * (gt_ctr_y - ex_ctr_y) / ex_lengths
        targets_dz = wx * (gt_ctr_z - ex_ctr_z) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dl = ww * torch.log(gt_lengths / ex_lengths)
        targets_dh = ww * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dw, targets_dl, targets_dh), dim=1)
        return targets

    def decode_xyz(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes
        TO_REMOVE = 1
        widths = boxes[:, 3] - boxes[:, 0] + TO_REMOVE
        lengths = boxes[:, 1] - boxes[:, 4] + TO_REMOVE
        heights = boxes[:, 5] - boxes[:, 2] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 4] + 0.5 * lengths
        ctr_z = boxes[:, 2] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wx
        dz = rel_codes[:, 2::6] / wx
        dw = rel_codes[:, 3::6] / ww
        dl = rel_codes[:, 4::6] / ww
        dh = rel_codes[:, 5::6] / ww
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dl = torch.clamp(dl, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * lengths[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * heights[:, None] + ctr_z[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_l = torch.exp(dl) * lengths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 4::6] = pred_ctr_y - 0.5 * pred_l
        pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_h
        pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 1::6] = pred_ctr_y + 0.5 * pred_l - 1
        pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_h - 1
        return pred_boxes

    def encode_xyz3d(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        TO_REMOVE = 1
        ex_widths = proposals[:, 3] - proposals[:, 0] + TO_REMOVE
        ex_lengths = proposals[:, 4] - proposals[:, 1] + TO_REMOVE
        ex_heights = proposals[:, 5] - proposals[:, 2] + TO_REMOVE
        ex_widths = ex_widths.clamp(min=1)
        ex_lengths = ex_lengths.clamp(min=1)
        ex_heights = ex_heights.clamp(min=1)
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_lengths
        ex_ctr_z = proposals[:, 2] + 0.5 * ex_heights
        gt_widths = reference_boxes[:, 3] - reference_boxes[:, 0] + TO_REMOVE
        gt_lengths = reference_boxes[:, 4] - reference_boxes[:, 1] + TO_REMOVE
        gt_heights = reference_boxes[:, 5] - reference_boxes[:, 2] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_lengths
        gt_ctr_z = reference_boxes[:, 2] + 0.5 * gt_heights
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wx * (gt_ctr_y - ex_ctr_y) / ex_lengths
        targets_dz = wx * (gt_ctr_z - ex_ctr_z) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dl = ww * torch.log(gt_lengths / ex_lengths)
        targets_dh = ww * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dw, targets_dl, targets_dh), dim=1)
        return targets

    def decode_xyz3d(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes
        TO_REMOVE = 1
        widths = boxes[:, 3] - boxes[:, 0] + TO_REMOVE
        lengths = boxes[:, 4] - boxes[:, 1] + TO_REMOVE
        heights = boxes[:, 5] - boxes[:, 2] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * lengths
        ctr_z = boxes[:, 2] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::6] / wx
        dy = rel_codes[:, 1::6] / wx
        dz = rel_codes[:, 2::6] / wx
        dw = rel_codes[:, 3::6] / ww
        dl = rel_codes[:, 4::6] / ww
        dh = rel_codes[:, 5::6] / ww
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dl = torch.clamp(dl, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * lengths[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * heights[:, None] + ctr_z[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_l = torch.exp(dl) * lengths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_l
        pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_h
        pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 4::6] = pred_ctr_y + 0.5 * pred_l - 1
        pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_h - 1
        return pred_boxes


class Matcher:
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

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
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
                raise ValueError('No ground-truth boxes available for one of the images during training')
            else:
                raise ValueError('No proposal boxes available for one of the images during training')
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None], as_tuple=False)
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
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
        raise RuntimeError('boxlists should have same image size, got {}, {}'.format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert('xyxy')
    boxlist2 = boxlist2.convert('xyxy')
    N = len(boxlist1)
    M = len(boxlist2)
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
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


class FastRCNNLossComputation:
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, cls_agnostic_bbox_reg=False):
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
        self.weights = torch.ones(config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES)
        for loc in range(config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES):
            self.weights[loc] = config.MODEL.FRUSTUM3D.CLASS_WEIGHTS[loc]

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['label'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('label')
            labels_per_image = labels_per_image
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, proposals_per_image.bbox)
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
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(labels, regression_targets, proposals):
            proposals_per_image.add_field('label', labels_per_image)
            proposals_per_image.add_field('regression_targets', regression_targets_per_image)
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img, as_tuple=False).squeeze(1)
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
        labels = cat([proposal.get_field('label') for proposal in proposals], dim=0)
        regression_targets = cat([proposal.get_field('regression_targets') for proposal in proposals], dim=0)
        classification_loss = F.cross_entropy(class_logits, labels, self.weights)
        sampled_pos_inds_subset = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset[:, None], map_inds], regression_targets[sampled_pos_inds_subset], size_average=False, beta=1)
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss


def make_roi_box_loss_evaluator():
    matcher = Matcher(config.MODEL.INSTANCE2D.ROI_HEADS.FG_IOU_THRESHOLD, config.MODEL.INSTANCE2D.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    bbox_reg_weights = config.MODEL.INSTANCE2D.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    fg_bg_sampler = BalancedPositiveNegativeSampler(config.MODEL.INSTANCE2D.ROI_HEADS.BATCH_SIZE_PER_IMAGE, config.MODEL.INSTANCE2D.ROI_HEADS.POSITIVE_FRACTION)
    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator


class BoxList:
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format(bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError('last dimension of bbox should have a size of 4, got {}'.format(bbox.size(-1)))
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

    def fields(self):
        return list(self.extra_fields.keys())

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
            bbox = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + (w - TO_REMOVE).clamp(min=0), ymin + (h - TO_REMOVE).clamp(min=0)
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            for k, v in self.extra_fields.items():
                if isinstance(v, SegmentationMask):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox
        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')
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
        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
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
        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        return bbox.convert(self.mode)

    def rotate(self, angle):
        bbox = []
        masks = []
        for mask in self.extra_fields['mask2d']:
            mask = mask.get_mask_tensor()
            mask = Image.fromarray(mask.numpy())
            mask = T.functional.rotate(mask, angle, False, False, None)
            mask = np.array(mask)
            coords = mask.nonzero()
            miny, minx, maxy, maxx = np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1])
            masks.append(mask)
            bbox.append([minx, miny, maxx, maxy])
        self.bbox = torch.FloatTensor(np.array(bbox))
        self.extra_fields['mask2d'] = SegmentationMask(torch.from_numpy(np.array(masks)), (320, 240), mode='mask')

    def jitter(self, min_jitter, max_jitter):
        bboxes = []
        masks = []
        for idx, (bbox, mask) in enumerate(zip(self.bbox, self.extra_fields['mask2d'])):
            minx = int(np.clip(bbox[0] + min_jitter, 0, self.size[0]).item())
            miny = int(np.clip(bbox[1] + min_jitter, 0, self.size[1]).item())
            maxx = int(np.clip(bbox[2] + max_jitter, 0, self.size[0]).item())
            maxy = int(np.clip(bbox[3] + max_jitter, 0, self.size[1]).item())
            bboxes.append([minx, miny, maxx, maxy])
            new_mask = np.zeros_like(mask.get_mask_tensor())
            new_mask[miny:maxy, minx:maxx] = mask.get_mask_tensor()[miny:maxy, minx:maxx]
            masks.append(new_mask)
        self.bbox = torch.FloatTensor(np.array(bboxes))
        self.extra_fields['mask2d'] = SegmentationMask(torch.from_numpy(np.array(masks)), (320, 240), mode='mask')

    def to(self, device, non_blocking=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            if isinstance(v, list):
                bbox.add_field(k, [v[int(idx.item())] for idx in item])
            else:
                bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]
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
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field='scores2d'):
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
    cat_boxes = BoxList(cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    for field in fields:
        data = cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(self, score_thresh=0.05, nms=0.5, detections_per_img=100, box_coder=None):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super().__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_coder = box_coder

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
        proposals = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)
        num_classes = class_prob.shape[1]
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            box_list = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            box_list = box_list.clip_to_image(remove_empty=False)
            box_list = self.filter_results(box_list, num_classes)
            results.append(box_list)
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
        box_list = BoxList(boxes, image_shape, mode='xyxy')
        box_list.add_field('scores2d', scores)
        return box_list

    def filter_results(self, box_list, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        boxes = box_list.bbox.reshape(-1, num_classes * 4)
        scores = box_list.get_field('scores2d').reshape(-1, num_classes)
        device = scores.device
        result = []
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4:(j + 1) * 4]
            box_list_for_class = BoxList(boxes_j, box_list.size, mode='xyxy')
            box_list_for_class.add_field('scores2d', scores_j)
            box_list_for_class = boxlist_nms(box_list_for_class, self.nms)
            num_labels = len(box_list_for_class)
            box_list_for_class.add_field('label', torch.full((num_labels,), j, dtype=torch.int64, device=device))
            result.append(box_list_for_class)
        result = cat_boxlist(result)
        number_of_detections = len(result)
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field('scores2d')
            image_thresh, _ = torch.kthvalue(cls_scores.cpu(), number_of_detections - self.detections_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor():
    bbox_reg_weights = config.MODEL.INSTANCE2D.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    score_thresh = config.MODEL.INSTANCE2D.ROI_HEADS.SCORE_THRESH
    nms_thresh = config.MODEL.INSTANCE2D.ROI_HEADS.NMS
    detections_per_img = config.MODEL.INSTANCE2D.ROI_HEADS.DETECTIONS_PER_IMG
    postprocessor = PostProcessor(score_thresh, nms_thresh, detections_per_img, box_coder)
    return postprocessor


class FastRCNNPredictor(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


def make_roi_box_predictor(in_channels):
    return FastRCNNPredictor(in_channels)


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.feature_extractor = make_roi_box_feature_extractor()
        self.predictor = make_roi_box_predictor(self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor()
        self.loss_evaluator = make_roi_box_loss_evaluator()

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
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}
        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])
        return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)


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
        super().__init__()
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
        labels = [bbox.get_field('label') for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]
        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        if self.masker:
            mask_prob, masks_raw = self.masker(mask_prob, boxes)
        results = []
        for prob, raw, box in zip(mask_prob, masks_raw, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field('mask2d', prob[:, 0, :, :])
            bbox.add_field('mask2draw', raw[:, 0, :, :])
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
    assert boxes[0].has_field('label')
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field('label')
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, 'GroupNorm: can only specify G or C/G.'
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, 'dim: {}, dim_per_gp: {}'.format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, 'dim: {}, num_groups: {}'.format(dim, num_groups)
        group_gn = num_groups
    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = config.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = config.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = config.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine)


def make_conv3x3(in_channels, out_channels, dilation=1, stride=1, use_gn=False, use_relu=False, kaiming_init=True):
    conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False if use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
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


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, in_channels):
        super().__init__()
        resolution = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels
        self.pooler = pooler
        layers = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.CONV_LAYERS
        dilation = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.DILATION
        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = 'mask_fcn{}'.format(layer_idx)
            module = make_conv3x3(next_feature, layer_features, dilation=dilation, stride=1, use_gn=False)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


def make_roi_mask_feature_extractor(in_channels):
    if config.MODEL.INSTANCE2D.FPN:
        return MaskRCNNFPNFeatureExtractor(in_channels)
    else:
        return ResNet50Conv5ROIFeatureExtractor()


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
    assert segmentation_masks.size == proposals.size, '{}, {}'.format(segmentation_masks, proposals)
    proposals = proposals.bbox
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0)


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
        target = target.copy_with_fields(['label', 'mask2d'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('label')
            labels_per_image = labels_per_image
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            positive_inds = torch.nonzero(labels_per_image > 0, as_tuple=False).squeeze(1)
            segmentation_masks = matched_targets.get_field('mask2d')
            segmentation_masks = segmentation_masks[positive_inds]
            positive_proposals = proposals_per_image[positive_inds]
            masks_per_image = project_masks_on_boxes(segmentation_masks, positive_proposals, self.discretization_size)
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
        positive_inds = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
        labels_pos = labels[positive_inds]
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits[positive_inds, labels_pos], mask_targets)
        return mask_loss


def make_roi_mask_loss_evaluator():
    matcher = Matcher(config.MODEL.INSTANCE2D.ROI_HEADS.FG_IOU_THRESHOLD, config.MODEL.INSTANCE2D.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    loss_evaluator = MaskRCNNLossComputation(matcher, config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.RESOLUTION)
    return loss_evaluator


def copy_mask_pixels(box, im_h, im_w, mask):
    im_mask = torch.zeros((im_h, im_w), dtype=torch.float)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0]:x_1 - box[0]]
    return im_mask


def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
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
    box = box
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = mask
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    if thresh >= 0:
        mask_thresholded = mask > thresh
    else:
        mask_thresholded = mask * 255
    im_mask_raw = copy_mask_pixels(box, im_h, im_w, mask)
    im_mask = copy_mask_pixels(box, im_h, im_w, mask_thresholded)
    return im_mask, im_mask_raw


class Masker:
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
        res = [paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding) for mask, box in zip(masks, boxes.bbox)]
        if len(res) > 0:
            res = list(zip(*res))
            res_thresholded = res[0]
            res_raw = res[1]
            if len(res_thresholded) > 0:
                res_thresholded = torch.stack(res_thresholded, dim=0)[:, None]
                res_raw = torch.stack(res_raw, dim=0)[:, None]
            return res_thresholded, res_raw
        res_thresholded = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        res_raw = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res_thresholded, res_raw

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == len(masks), 'Masks and boxes should have the same length.'
        results = []
        results_raw = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), 'Number of objects should be the same.'
            result, result_raw = self.forward_single_image(mask, box)
            results.append(result)
            results_raw.append(result_raw)
        return results, results_raw


def make_roi_mask_post_processor():
    mask_threshold = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
    masker = Masker(threshold=mask_threshold, padding=1)
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor


class MaskRCNNC4Predictor(nn.Module):

    def __init__(self, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.CONV_LAYERS[-1]
        num_inputs = in_channels
        self.conv5_mask = nn.ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(in_channels):
    return MaskRCNNC4Predictor(in_channels)


class ROIMaskHead(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.feature_extractor = make_roi_mask_feature_extractor(in_channels)
        self.predictor = make_roi_mask_predictor(self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor()
        self.loss_evaluator = make_roi_mask_loss_evaluator()

    def forward(self, features, proposals, targets=None) ->ModuleResult:
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
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)
        result = self.post_processor(mask_logits, proposals)
        if not self.training:
            return result, {}
        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
        return result, dict(loss_mask=loss_mask)


class CombinedROIHeads(nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, heads):
        super().__init__(heads)
        if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE:
            mask_features = features
            if self.training and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        return detections, losses

    def inference(self, features, proposals):
        x, detections, _ = self.box(features, proposals)
        if config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE:
            mask_features = features
            if self.training and config.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            detections, _ = self.mask(mask_features, detections, None)
        return detections


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super().__init__()
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
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
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
    anchors = np.vstack([_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])])
    return torch.from_numpy(anchors)


def generate_anchors(stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(stride, np.array(sizes, dtype=np.float) / stride, np.array(aspect_ratios, dtype=np.float))


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0), anchor_strides=(8, 16, 32), straddle_thresh=0):
        super().__init__()
        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [generate_anchors(anchor_stride, sizes, aspect_ratios).float()]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError('FPN should have #anchor_strides == #sizes')
            cell_anchors = [generate_anchors(anchor_stride, size if isinstance(size, (tuple, list)) else (size,), aspect_ratios).float() for anchor_stride, size in zip(anchor_strides, sizes)]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (anchors[..., 0] >= -self.straddle_thresh) & (anchors[..., 1] >= -self.straddle_thresh) & (anchors[..., 2] < image_width + self.straddle_thresh) & (anchors[..., 3] < image_height + self.straddle_thresh)
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field('visibility', inds_inside)

    def forward(self, image_sizes, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_width, image_height) in enumerate(image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(anchors_per_feature_map, (image_width, image_height), mode='xyxy')
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    xywh_boxes = boxlist.convert('xywh').bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero(as_tuple=False).squeeze(1)
    return boxlist[keep]


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, box_coder=None):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
        """
        super().__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        device = proposals[0].bbox.device
        gt_boxes = [target.copy_with_fields([]) for target in targets]
        for gt_box, target in zip(gt_boxes, targets):
            gt_box.add_field('objectness', torch.ones(len(gt_box), device=device))
        proposals = [cat_boxlist((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]
        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
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
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]
        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        proposals = self.box_coder.decode(box_regression.view(-1, 4), concat_anchors.view(-1, 4))
        proposals = proposals.view(N, -1, 4)
        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode='xyxy')
            boxlist.add_field('objectness', score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = boxlist_nms(boxlist, self.nms_thresh, max_proposals=self.post_nms_top_n, score_field='objectness')
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
        if self.training:
            objectness = torch.cat([boxlist.get_field('objectness') for boxlist in boxlists], dim=0)
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i].bool()]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field('objectness')
                post_nms_top_n = min(self.post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def make_anchor_generator():
    anchor_sizes = config.MODEL.INSTANCE2D.RPN.ANCHOR_SIZES
    aspect_ratios = config.MODEL.INSTANCE2D.RPN.ASPECT_RATIOS
    anchor_stride = config.MODEL.INSTANCE2D.RPN.ANCHOR_STRIDE
    straddle_thresh = config.MODEL.INSTANCE2D.RPN.STRADDLE_THRESH
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh)
    return anchor_generator


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RPNLossComputation:
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder, generate_labels_func):
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
            matched_targets = self.match_targets_to_anchors(anchors_per_image, targets_per_image, self.copied_fields)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            if 'not_visibility' in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field('visibility')] = -1
            if 'between_thresholds' in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1
            regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)
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
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0), as_tuple=False).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0), as_tuple=False).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = smooth_l1_loss(box_regression[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1.0 / 9, size_average=False) / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss


def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field('matched_idxs')
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(box_coder):
    matcher = Matcher(config.MODEL.INSTANCE2D.RPN.FG_IOU_THRESHOLD, config.MODEL.INSTANCE2D.RPN.BG_IOU_THRESHOLD, allow_low_quality_matches=True)
    fg_bg_sampler = BalancedPositiveNegativeSampler(config.MODEL.INSTANCE2D.RPN.BATCH_SIZE_PER_IMAGE, config.MODEL.INSTANCE2D.RPN.POSITIVE_FRACTION)
    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder, generate_rpn_labels)
    return loss_evaluator


def make_rpn_postprocessor(rpn_box_coder, is_train):
    pre_nms_top_n = config.MODEL.INSTANCE2D.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.INSTANCE2D.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.INSTANCE2D.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.INSTANCE2D.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.INSTANCE2D.RPN.NMS_THRESH
    min_size = config.MODEL.INSTANCE2D.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(pre_nms_top_n=pre_nms_top_n, post_nms_top_n=post_nms_top_n, nms_thresh=nms_thresh, min_size=min_size, box_coder=rpn_box_coder)
    return box_selector


class RPNModule(nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, in_channels):
        super().__init__()
        anchor_generator = make_anchor_generator()
        head = RPNHead(in_channels, anchor_generator.num_anchors_per_location()[0])
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        box_selector_train = make_rpn_postprocessor(rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(rpn_box_coder, is_train=False)
        loss_evaluator = make_rpn_loss_evaluator(rpn_box_coder)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, features, targets=None):
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator([(320, 240) for _ in range(objectness[0].shape[0])], features)
        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        with torch.no_grad():
            boxes = self.box_selector_train(anchors, objectness, rpn_box_regression, targets)
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(anchors, objectness, rpn_box_regression, targets)
        losses = {'loss_objectness': loss_objectness, 'loss_rpn_box_reg': loss_rpn_box_reg}
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        return boxes, {}

    def inference(self, features):
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator([(320, 240) for _ in range(objectness[0].shape[0])], features)
        return self._forward_test(anchors, objectness, rpn_box_regression)


def conditional_to(x: torch.Tensor, device: str='cuda') ->torch.Tensor:
    if device is None:
        return x
    else:
        return x


def collect(data: List[FieldList], field: str, device: str='cuda', access_fn=None) ->torch.Tensor:
    if access_fn is None:
        return torch.stack([conditional_to(t.get_field(field), device) for t in data], dim=0)
    else:
        return torch.stack([conditional_to(access_fn(t.get_field(field)), device) for t in data], dim=0)


def filter_instances(instances2d, instances3d):
    instances_filtered = torch.zeros_like(instances3d)
    instance_ids_2d = instances2d['locations'][0] + 1
    for instance_id in instance_ids_2d:
        if instance_id != 0:
            instance_mask = instances3d == instance_id
            instances_filtered[instance_mask] = instance_id
    return instances_filtered


def nn_search(grid, point, radius=3):
    start = -radius
    end = radius
    label = torch.zeros([len(point)], device=point.device, dtype=grid.dtype)
    mask = torch.zeros_like(label).bool()
    for x in range(start, end):
        for y in range(start, end):
            for z in range(start, end):
                offset = torch.tensor([x, y, z], device=point.device)
                point_offset = point + offset
                label_bi = grid[point_offset[:, 0], point_offset[:, 1], point_offset[:, 2]]
                if label_bi.sum() != 0:
                    new_mask = (label_bi > 0) * ~mask
                    label[new_mask] = label_bi[new_mask]
                    mask = mask + new_mask
    return label


class PanopticReconstruction(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        backbone2d = backbone.build_backbone()
        self.encoder2d: nn.Module = backbone.ResNetEncoder(backbone2d)
        self.depth2d: nn.Module = depth.DepthPrediction()
        self.instance2d: nn.Module = detector.GeneralizedRCNN(self.encoder2d.out_channels[:3])
        self.projection: nn.Module = projection.SparseProjection()
        self.frustum3d: nn.Module = frustum.FrustumCompletion()
        self.postprocess: nn.Module = frustum.PostProcess()
        self.training_stages = OrderedDict([('LEVEL-64', config.MODEL.FRUSTUM3D.IS_LEVEL_64), ('LEVEL-128', config.MODEL.FRUSTUM3D.IS_LEVEL_128), ('LEVEL-256', config.MODEL.FRUSTUM3D.IS_LEVEL_256), ('FULL', True)])

    def forward(self, images: torch.Tensor, targets: List[FieldList], is_validate=False) ->ModuleResult:
        losses = {}
        results = {}
        _, image_features = self.encoder2d(images)
        depth_targets = [target.get_field('depth') for target in targets]
        depth_losses, depth_results = self.depth2d(image_features['blocks'], depth_targets)
        losses.update({'depth': depth_losses})
        results.update({'depth': depth_results})
        instance_losses, instance_results = self.instance2d(image_features, targets, is_validate)
        losses.update({'instance': instance_losses})
        results.update({'instance': instance_results})
        feature2d = results['depth']['features']
        projection_results = self.projection(results['depth']['prediction'], feature2d, instance_results, targets)
        results.update({'projection': projection_results})
        frustum_losses, frustum_results = self.frustum3d(projection_results, targets)
        losses.update({'frustum': frustum_losses})
        results.update({'frustum': frustum_results})
        if self.get_current_training_stage() == 'FULL':
            _, panoptic_results = self.postprocess(instance_results, frustum_results)
            results.update({'panoptic': panoptic_results})
        return losses, results

    def inference(self, image: torch.Tensor, intrinsic, frustum_mask):
        results = {'input': image, 'intrinsic': intrinsic}
        _, image_features = self.encoder2d(image)
        depth_result, depth_features = self.depth2d.inference(image_features['blocks'])
        results['depth'] = DepthMap(depth_result.squeeze(), intrinsic)
        instance_result = self.instance2d.inference(image_features)
        results['instance'] = instance_result
        projection_result: Me.SparseTensor = self.projection.inference(depth_result, depth_features, instance_result, intrinsic)
        results['projection'] = projection_result
        frustum_result = self.frustum3d.inference(projection_result, frustum_mask)
        results['frustum'] = frustum_result
        _, panoptic_result = self.postprocess(instance_result, frustum_result)
        results['panoptic'] = panoptic_result
        return results

    def log_model_info(self) ->None:
        if config.MODEL.INSTANCE2D.USE:
            logger.info(f'number of weights in detection network: {utils.count_parameters(self.instance2d):,}')
        if config.MODEL.DEPTH2D.USE or not config.MODEL.PROJECTION.OCC_IN:
            logger.info(f'number of weights in depth network: {utils.count_parameters(self.depth2d):,}')
        if config.MODEL.FRUSTUM3D.USE:
            logger.info(f'number of weights in 3D network: {utils.count_parameters(self.frustum3d):,}')

    def fix_weights(self) ->None:
        if config.MODEL.FIX2D:
            modeling.fix_weights(self, '2d')
        if config.MODEL.INSTANCE2D.FIX:
            modeling.fix_weights(self, 'instance2d')
        if config.MODEL.DEPTH2D.FIX:
            modeling.fix_weights(self, 'depth2d')
        if config.MODEL.FRUSTUM3D.FIX:
            modeling.fix_weights(self, 'frustum3d')

    def switch_training(self) ->None:
        self.train()
        if config.MODEL.FIX2D:
            self.encoder2d.eval()
            if config.MODEL.DEPTH2D.USE:
                self.depth2d.eval()
            if config.MODEL.INSTANCE2D.USE:
                self.instance2d.eval()
        if config.MODEL.FRUSTUM3D.FIX:
            self.frustum3d.eval()
        if config.MODEL.DEPTH2D.FIX:
            self.depth2d.eval()
        if config.MODEL.INSTANCE2D.FIX:
            self.instance2d.eval()

    def switch_test(self) ->None:
        self.eval()

    def get_current_training_stage(self) ->str:
        for level, status in self.training_stages.items():
            if status:
                return level

    def set_current_training_stage(self, iteration: int) ->str:
        num_iterations = config.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_64
        last_training_stage = None
        if iteration >= num_iterations and self.training_stages['LEVEL-64']:
            self.training_stages['LEVEL-64'] = False
            config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
            last_training_stage = 'level_64'
        num_iterations += config.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_128
        if iteration >= num_iterations and config.MODEL.FRUSTUM3D.IS_LEVEL_128:
            self.training_stages['LEVEL-128'] = False
            config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
            last_training_stage = 'level_128'
        num_iterations += config.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_256
        if iteration >= num_iterations and config.MODEL.FRUSTUM3D.IS_LEVEL_256:
            self.training_stages['LEVEL-256'] = False
            config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
            last_training_stage = 'level_256'
        return last_training_stage


def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max, transform=None):
    x = image_size[0]
    y = image_size[1]
    eight_points = np.array([[0 * depth_min, 0 * depth_min, depth_min, 1.0], [0 * depth_min, y * depth_min, depth_min, 1.0], [x * depth_min, y * depth_min, depth_min, 1.0], [x * depth_min, 0 * depth_min, depth_min, 1.0], [0 * depth_max, 0 * depth_max, depth_max, 1.0], [0 * depth_max, y * depth_max, depth_max, 1.0], [x * depth_max, y * depth_max, depth_max, 1.0], [x * depth_max, 0 * depth_max, depth_max, 1.0]]).transpose()
    frustum = np.dot(intrinsic_inv, eight_points)
    if transform is not None:
        frustum = np.dot(transform, frustum)
    frustum = frustum.transpose()
    return frustum[:, :3]


def generate_frustum_volume(frustum, voxel_size):
    max_x = np.max(frustum[:, 0]) / voxel_size
    max_y = np.max(frustum[:, 1]) / voxel_size
    max_z = np.max(frustum[:, 2]) / voxel_size
    min_x = np.min(frustum[:, 0]) / voxel_size
    min_y = np.min(frustum[:, 1]) / voxel_size
    min_z = np.min(frustum[:, 2]) / voxel_size
    dim_x = math.ceil(max_x - min_x)
    dim_y = math.ceil(max_y - min_y)
    dim_z = math.ceil(max_z - min_z)
    camera2frustum = np.array([[1.0 / voxel_size, 0, 0, -min_x], [0, 1.0 / voxel_size, 0, -min_y], [0, 0, 1.0 / voxel_size, -min_z], [0, 0, 0, 1.0]])
    return (dim_x, dim_y, dim_z), camera2frustum


def compute_camera2frustum_transform(intrinsic: torch.Tensor, image_size: Tuple, depth_min: float, depth_max: float, voxel_size: float) ->torch.Tensor:
    frustum = generate_frustum(image_size, torch.inverse(intrinsic), depth_min, depth_max)
    _, camera2frustum = generate_frustum_volume(frustum, voxel_size)
    camera2frustum = torch.from_numpy(camera2frustum).float()
    return camera2frustum


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvTranspose3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FrozenBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InstanceNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (InstanceNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (InstanceNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RPNHead,
     lambda: ([], {'in_channels': 4, 'num_anchors': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBlock3d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetDownsample,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sobel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_xheon_panoptic_reconstruction(_paritybench_base):
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

