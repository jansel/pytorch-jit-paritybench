import sys
_module = sys.modules[__name__]
del sys
coco_detection = _module
coco_instance = _module
default_runtime = _module
db_cascade_mask_rcnn_r50_fpn = _module
schedule_1x = _module
db_cascade_mask_rcnn_r50_fpn_1x_coco = _module
db_cascade_mask_rcnn_x101_32x4d_fpn_1x_coco = _module
convert_dual_backbone = _module
image_demo = _module
conf = _module
mmdet = _module
apis = _module
inference = _module
test = _module
train = _module
core = _module
anchor = _module
anchor_generator = _module
builder = _module
point_generator = _module
utils = _module
bbox = _module
assigners = _module
approx_max_iou_assigner = _module
assign_result = _module
atss_assigner = _module
base_assigner = _module
center_region_assigner = _module
max_iou_assigner = _module
point_assigner = _module
coder = _module
base_bbox_coder = _module
delta_xywh_bbox_coder = _module
legacy_delta_xywh_bbox_coder = _module
pseudo_bbox_coder = _module
tblr_bbox_coder = _module
demodata = _module
iou_calculators = _module
iou2d_calculator = _module
samplers = _module
base_sampler = _module
combined_sampler = _module
instance_balanced_pos_sampler = _module
iou_balanced_neg_sampler = _module
ohem_sampler = _module
pseudo_sampler = _module
random_sampler = _module
sampling_result = _module
score_hlr_sampler = _module
transforms = _module
evaluation = _module
bbox_overlaps = _module
class_names = _module
eval_hooks = _module
mean_ap = _module
recall = _module
fp16 = _module
decorators = _module
hooks = _module
utils = _module
mask = _module
mask_target = _module
structures = _module
optimizer = _module
builder = _module
copy_of_sgd = _module
default_constructor = _module
post_processing = _module
bbox_nms = _module
merge_augs = _module
dist_utils = _module
misc = _module
datasets = _module
builder = _module
cityscapes = _module
coco = _module
custom = _module
dataset_wrappers = _module
pipelines = _module
compose = _module
formating = _module
instaboost = _module
loading = _module
test_aug = _module
distributed_sampler = _module
group_sampler = _module
voc = _module
wider_face = _module
xml_style = _module
models = _module
backbones = _module
db_resnet = _module
db_resnext = _module
hrnet = _module
regnet = _module
res2net = _module
resnet = _module
resnext = _module
ssd_vgg = _module
builder = _module
dense_heads = _module
anchor_head = _module
atss_head = _module
fcos_head = _module
fovea_head = _module
free_anchor_retina_head = _module
fsaf_head = _module
ga_retina_head = _module
ga_rpn_head = _module
guided_anchor_head = _module
nasfcos_head = _module
pisa_retinanet_head = _module
pisa_ssd_head = _module
reppoints_head = _module
retina_head = _module
retina_sepbn_head = _module
rpn_head = _module
ssd_head = _module
detectors = _module
atss = _module
base = _module
cascade_rcnn = _module
fast_rcnn = _module
faster_rcnn = _module
fcos = _module
fovea = _module
fsaf = _module
grid_rcnn = _module
htc = _module
mask_rcnn = _module
mask_scoring_rcnn = _module
nasfcos = _module
reppoints_detector = _module
retinanet = _module
rpn = _module
single_stage = _module
test_mixins = _module
two_stage = _module
losses = _module
accuracy = _module
balanced_l1_loss = _module
cross_entropy_loss = _module
focal_loss = _module
ghm_loss = _module
iou_loss = _module
mse_loss = _module
pisa_loss = _module
smooth_l1_loss = _module
utils = _module
necks = _module
bfp = _module
fpn = _module
fpn_carafe = _module
hrfpn = _module
nas_fpn = _module
nasfcos_fpn = _module
pafpn = _module
roi_heads = _module
base_roi_head = _module
bbox_heads = _module
bbox_head = _module
convfc_bbox_head = _module
double_bbox_head = _module
cascade_roi_head = _module
double_roi_head = _module
grid_roi_head = _module
htc_roi_head = _module
mask_heads = _module
fcn_mask_head = _module
fused_semantic_head = _module
grid_head = _module
htc_mask_head = _module
maskiou_head = _module
mask_scoring_roi_head = _module
pisa_roi_head = _module
roi_extractors = _module
single_level = _module
shared_heads = _module
res_layer = _module
standard_roi_head = _module
test_mixins = _module
res_layer = _module
ops = _module
carafe = _module
carafe = _module
grad_check = _module
setup = _module
context_block = _module
conv_ws = _module
dcn = _module
deform_conv = _module
deform_pool = _module
generalized_attention = _module
masked_conv = _module
masked_conv = _module
merge_cells = _module
nms = _module
nms_wrapper = _module
non_local = _module
plugin = _module
roi_align = _module
gradcheck = _module
roi_align = _module
roi_pool = _module
gradcheck = _module
roi_pool = _module
sigmoid_focal_loss = _module
sigmoid_focal_loss = _module
wrappers = _module
collect_env = _module
contextmanagers = _module
flops_counter = _module
logger = _module
profiling = _module
util_mixins = _module
version = _module
setup = _module
async_benchmark = _module
test_anchor = _module
test_assigner = _module
test_async = _module
test_backbone = _module
test_config = _module
test_dataset = _module
test_forward = _module
test_fp16 = _module
test_heads = _module
test_losses = _module
test_masks = _module
test_merge_cells = _module
test_necks = _module
test_nms = _module
test_optimizer = _module
test_loading = _module
test_transform = _module
test_pisa_heads = _module
test_sampler = _module
test_soft_nms = _module
test_utils = _module
test_wrappers = _module
analyze_logs = _module
benchmark = _module
browse_dataset = _module
coco_error_analysis = _module
pascal_voc = _module
detectron2pytorch = _module
fuse_conv_bn = _module
get_flops = _module
print_config = _module
publish_model = _module
pytorch2onnx = _module
regnet2mmdet = _module
robustness_eval = _module
test = _module
test_robustness = _module
train = _module
upgrade_model_version = _module

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


import warnings


import matplotlib.pyplot as plt


import time


import torch.distributed as dist


import random


from collections import OrderedDict


import numpy as np


from torch.nn.modules.utils import _pair


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import DataLoader


import functools


from inspect import getfullargspec


import copy


import torch.nn as nn


from collections import abc


import inspect


from torch.optim import SGD


from torch.nn import GroupNorm


from torch.nn import LayerNorm


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.modules.instancenorm import _InstanceNorm


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from functools import partial


from torch.utils.data import Dataset


import math


from collections import defaultdict


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from collections.abc import Sequence


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import Sampler


import torch.utils.checkpoint as cp


import torch.nn.functional as F


from torch import nn


from torch.utils.checkpoint import checkpoint


import logging


from torch import nn as nn


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.autograd import gradcheck


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _single


import torchvision


from typing import List


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


from torch.utils.cpp_extension import CppExtension


from torch.nn.modules import AvgPool2d


from torch.nn.modules import GroupNorm


from itertools import product


from torch.onnx import OperatorExportTypes


import re


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
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


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ContextBlock(nn.Module):
    """ContextBlock module in GCNet.

    See 'GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond'
    (https://arxiv.org/abs/1904.11492) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling
        fusion_types (list[str]|tuple[str]): Fusion method for feature fusion,
            options: 'channels_add', 'channel_mul'
    """

    def __init__(self, in_channels, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
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


class GeneralizedAttention(nn.Module):
    """GeneralizedAttention module.

    See 'An Empirical Study of Spatial Attention Mechanisms in Deep Networks'
    (https://arxiv.org/abs/1711.07971) for details.

    Args:
        in_channels (int): Channels of the input feature map.
        spatial_range (int): The spatial range.
            -1 indicates no spatial range constraint.
        num_heads (int): The head number of empirical_attention module.
        position_embedding_dim (int): The position embedding dimension.
        position_magnitude (int): A multiplier acting on coord difference.
        kv_stride (int): The feature stride acting on key/value feature map.
        q_stride (int): The feature stride acting on query feature map.
        attention_type (str): A binary indicator string for indicating which
            items in generalized empirical_attention module are used.
            '1000' indicates 'query and key content' (appr - appr) item,
            '0100' indicates 'query content and relative position'
              (appr - position) item,
            '0010' indicates 'key content only' (bias - appr) item,
            '0001' indicates 'relative position only' (bias - position) item.
    """

    def __init__(self, in_channels, spatial_range=-1, num_heads=9, position_embedding_dim=-1, position_magnitude=1, kv_stride=2, q_stride=1, attention_type='1111'):
        super(GeneralizedAttention, self).__init__()
        self.position_embedding_dim = position_embedding_dim if position_embedding_dim > 0 else in_channels
        self.position_magnitude = position_magnitude
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.spatial_range = spatial_range
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.attention_type = [bool(int(_)) for _ in attention_type]
        self.qk_embed_dim = in_channels // num_heads
        out_c = self.qk_embed_dim * num_heads
        if self.attention_type[0] or self.attention_type[1]:
            self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=1, bias=False)
            self.query_conv.kaiming_init = True
        if self.attention_type[0] or self.attention_type[2]:
            self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_c, kernel_size=1, bias=False)
            self.key_conv.kaiming_init = True
        self.v_dim = in_channels // num_heads
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.v_dim * num_heads, kernel_size=1, bias=False)
        self.value_conv.kaiming_init = True
        if self.attention_type[1] or self.attention_type[3]:
            self.appr_geom_fc_x = nn.Linear(self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_x.kaiming_init = True
            self.appr_geom_fc_y = nn.Linear(self.position_embedding_dim // 2, out_c, bias=False)
            self.appr_geom_fc_y.kaiming_init = True
        if self.attention_type[2]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.appr_bias = nn.Parameter(appr_bias_value)
        if self.attention_type[3]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            geom_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.geom_bias = nn.Parameter(geom_bias_value)
        self.proj_conv = nn.Conv2d(in_channels=self.v_dim * num_heads, out_channels=in_channels, kernel_size=1, bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.spatial_range >= 0:
            if in_channels == 256:
                max_len = 84
            elif in_channels == 512:
                max_len = 42
            max_len_kv = int((max_len - 1.0) / self.kv_stride + 1)
            local_constraint_map = np.ones((max_len, max_len, max_len_kv, max_len_kv), dtype=np.int)
            for iy in range(max_len):
                for ix in range(max_len):
                    local_constraint_map[iy, ix, max((iy - self.spatial_range) // self.kv_stride, 0):min((iy + self.spatial_range + 1) // self.kv_stride + 1, max_len), max((ix - self.spatial_range) // self.kv_stride, 0):min((ix + self.spatial_range + 1) // self.kv_stride + 1, max_len)] = 0
            self.local_constraint_map = nn.Parameter(torch.from_numpy(local_constraint_map).byte(), requires_grad=False)
        if self.q_stride > 1:
            self.q_downsample = nn.AvgPool2d(kernel_size=1, stride=self.q_stride)
        else:
            self.q_downsample = None
        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None
        self.init_weights()

    def get_position_embedding(self, h, w, h_kv, w_kv, q_stride, kv_stride, device, feat_dim, wave_length=1000):
        h_idxs = torch.linspace(0, h - 1, h)
        h_idxs = h_idxs.view((h, 1)) * q_stride
        w_idxs = torch.linspace(0, w - 1, w)
        w_idxs = w_idxs.view((w, 1)) * q_stride
        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv)
        h_kv_idxs = h_kv_idxs.view((h_kv, 1)) * kv_stride
        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv)
        w_kv_idxs = w_kv_idxs.view((w_kv, 1)) * kv_stride
        h_diff = h_idxs.unsqueeze(1) - h_kv_idxs.unsqueeze(0)
        h_diff *= self.position_magnitude
        w_diff = w_idxs.unsqueeze(1) - w_kv_idxs.unsqueeze(0)
        w_diff *= self.position_magnitude
        feat_range = torch.arange(0, feat_dim / 4)
        dim_mat = torch.Tensor([wave_length])
        dim_mat = dim_mat ** (4.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view((1, 1, -1))
        embedding_x = torch.cat(((w_diff / dim_mat).sin(), (w_diff / dim_mat).cos()), dim=2)
        embedding_y = torch.cat(((h_diff / dim_mat).sin(), (h_diff / dim_mat).cos()), dim=2)
        return embedding_x, embedding_y

    def forward(self, x_input):
        num_heads = self.num_heads
        if self.q_downsample is not None:
            x_q = self.q_downsample(x_input)
        else:
            x_q = x_input
        n, _, h, w = x_q.shape
        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape
        if self.attention_type[0] or self.attention_type[1]:
            proj_query = self.query_conv(x_q).view((n, num_heads, self.qk_embed_dim, h * w))
            proj_query = proj_query.permute(0, 1, 3, 2)
        if self.attention_type[0] or self.attention_type[2]:
            proj_key = self.key_conv(x_kv).view((n, num_heads, self.qk_embed_dim, h_kv * w_kv))
        if self.attention_type[1] or self.attention_type[3]:
            position_embed_x, position_embed_y = self.get_position_embedding(h, w, h_kv, w_kv, self.q_stride, self.kv_stride, x_input.device, self.position_embedding_dim)
            position_feat_x = self.appr_geom_fc_x(position_embed_x).view(1, w, w_kv, num_heads, self.qk_embed_dim).permute(0, 3, 1, 2, 4).repeat(n, 1, 1, 1, 1)
            position_feat_y = self.appr_geom_fc_y(position_embed_y).view(1, h, h_kv, num_heads, self.qk_embed_dim).permute(0, 3, 1, 2, 4).repeat(n, 1, 1, 1, 1)
            position_feat_x /= math.sqrt(2)
            position_feat_y /= math.sqrt(2)
        if np.sum(self.attention_type) == 1 and self.attention_type[2]:
            appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim).repeat(n, 1, 1, 1)
            energy = torch.matmul(appr_bias, proj_key).view(n, num_heads, 1, h_kv * w_kv)
            h = 1
            w = 1
        else:
            if not self.attention_type[0]:
                energy = torch.zeros(n, num_heads, h, w, h_kv, w_kv, dtype=x_input.dtype, device=x_input.device)
            if self.attention_type[0] or self.attention_type[2]:
                if self.attention_type[0] and self.attention_type[2]:
                    appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim)
                    energy = torch.matmul(proj_query + appr_bias, proj_key).view(n, num_heads, h, w, h_kv, w_kv)
                elif self.attention_type[0]:
                    energy = torch.matmul(proj_query, proj_key).view(n, num_heads, h, w, h_kv, w_kv)
                elif self.attention_type[2]:
                    appr_bias = self.appr_bias.view(1, num_heads, 1, self.qk_embed_dim).repeat(n, 1, 1, 1)
                    energy += torch.matmul(appr_bias, proj_key).view(n, num_heads, 1, 1, h_kv, w_kv)
            if self.attention_type[1] or self.attention_type[3]:
                if self.attention_type[1] and self.attention_type[3]:
                    geom_bias = self.geom_bias.view(1, num_heads, 1, self.qk_embed_dim)
                    proj_query_reshape = (proj_query + geom_bias).view(n, num_heads, h, w, self.qk_embed_dim)
                    energy_x = torch.matmul(proj_query_reshape.permute(0, 1, 3, 2, 4), position_feat_x.permute(0, 1, 2, 4, 3))
                    energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)
                    energy_y = torch.matmul(proj_query_reshape, position_feat_y.permute(0, 1, 2, 4, 3))
                    energy_y = energy_y.unsqueeze(5)
                    energy += energy_x + energy_y
                elif self.attention_type[1]:
                    proj_query_reshape = proj_query.view(n, num_heads, h, w, self.qk_embed_dim)
                    proj_query_reshape = proj_query_reshape.permute(0, 1, 3, 2, 4)
                    position_feat_x_reshape = position_feat_x.permute(0, 1, 2, 4, 3)
                    position_feat_y_reshape = position_feat_y.permute(0, 1, 2, 4, 3)
                    energy_x = torch.matmul(proj_query_reshape, position_feat_x_reshape)
                    energy_x = energy_x.permute(0, 1, 3, 2, 4).unsqueeze(4)
                    energy_y = torch.matmul(proj_query_reshape, position_feat_y_reshape)
                    energy_y = energy_y.unsqueeze(5)
                    energy += energy_x + energy_y
                elif self.attention_type[3]:
                    geom_bias = self.geom_bias.view(1, num_heads, self.qk_embed_dim, 1).repeat(n, 1, 1, 1)
                    position_feat_x_reshape = position_feat_x.view(n, num_heads, w * w_kv, self.qk_embed_dim)
                    position_feat_y_reshape = position_feat_y.view(n, num_heads, h * h_kv, self.qk_embed_dim)
                    energy_x = torch.matmul(position_feat_x_reshape, geom_bias)
                    energy_x = energy_x.view(n, num_heads, 1, w, 1, w_kv)
                    energy_y = torch.matmul(position_feat_y_reshape, geom_bias)
                    energy_y = energy_y.view(n, num_heads, h, 1, h_kv, 1)
                    energy += energy_x + energy_y
            energy = energy.view(n, num_heads, h * w, h_kv * w_kv)
        if self.spatial_range >= 0:
            cur_local_constraint_map = self.local_constraint_map[:h, :w, :h_kv, :w_kv].contiguous().view(1, 1, h * w, h_kv * w_kv)
            energy = energy.masked_fill_(cur_local_constraint_map, float('-inf'))
        attention = F.softmax(energy, 3)
        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.view((n, num_heads, self.v_dim, h_kv * w_kv)).permute(0, 1, 3, 2)
        out = torch.matmul(attention, proj_value_reshape).permute(0, 1, 3, 2).contiguous().view(n, self.v_dim * self.num_heads, h, w)
        out = self.proj_conv(out)
        out = self.gamma * out + x_input
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(m, mode='fan_in', nonlinearity='leaky_relu', bias=0, distribution='uniform', a=1)


class NonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self, in_channels, reduction=2, use_scale=True, conv_cfg=None, norm_cfg=None, mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']
        self.g = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, act_cfg=None)
        self.theta = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, act_cfg=None)
        self.phi = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, act_cfg=None)
        self.conv_out = ConvModule(self.inter_channels, self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1] ** 0.5
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


plugin_cfg = {'ContextBlock': ('context_block', ContextBlock), 'GeneralizedAttention': ('gen_attention_block', GeneralizedAttention), 'NonLocal2D': ('nonlocal_block', NonLocal2D)}


def build_plugin_layer(cfg, postfix='', **kwargs):
    """ Build plugin layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify plugin layer type.
            layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created plugin layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in plugin_cfg:
        raise KeyError(f'Unrecognized plugin type {layer_type}')
    else:
        abbr, plugin_layer = plugin_cfg[layer_type]
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    layer = plugin_layer(**kwargs, **cfg_)
    return name, layer


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None
        if self.with_plugins:
            self.after_conv1_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv1']
            self.after_conv2_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv2']
            self.after_conv3_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv3']
        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(conv_cfg, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(dcn, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """ make plugins for block

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.

        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)
        return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), **kwargs):
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1]])
            downsample = nn.Sequential(*downsample)
        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs))
        super(ResLayer, self).__init__(*layers)


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)
    return logger


class DB_ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, in_channels=3, base_channels=64, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), style='pytorch', deep_stem=False, avg_down=False, frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, dcn=None, stage_with_dcn=(False, False, False, False), plugins=None, with_cp=False, zero_init_residual=True):
        super(DB_ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = base_channels
        self._make_stem_layer(in_channels, base_channels)
        self.res_layers = []
        self.second_res_layers = []
        self.eb_second_conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.eb_second_conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.eb_second_conv3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.eb_second_conv4 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        normalize = self.norm_cfg
        self.eb_second_bn1_name, eb_second_bn1 = build_norm_layer(normalize, 64, postfix='eb_second_bn1')
        self.add_module(self.eb_second_bn1_name, eb_second_bn1)
        self.eb_second_bn2_name, eb_second_bn2 = build_norm_layer(normalize, 256, postfix='eb_second_bn2')
        self.add_module(self.eb_second_bn2_name, eb_second_bn2)
        self.eb_second_bn3_name, eb_second_bn3 = build_norm_layer(normalize, 512, postfix='eb_second_bn3')
        self.add_module(self.eb_second_bn3_name, eb_second_bn3)
        self.eb_second_bn4_name, eb_second_bn4 = build_norm_layer(normalize, 1024, postfix='eb_second_bn4')
        self.add_module(self.eb_second_bn4_name, eb_second_bn4)
        self.eb_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(block=self.block, inplanes=self.inplanes, planes=planes, num_blocks=num_blocks, stride=stride, dilation=dilation, style=self.style, avg_down=self.avg_down, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, plugins=stage_plugins)
            second_res_layer = self.make_res_layer(block=self.block, inplanes=self.inplanes, planes=planes, num_blocks=num_blocks, stride=stride, dilation=dilation, style=self.style, avg_down=self.avg_down, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            second_layer_name = 'second_layer{}'.format(i + 1)
            self.add_module(second_layer_name, second_res_layer)
            self.second_res_layers.append(second_layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """ make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.
        An example of plugins format could be:

        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)
        return stage_plugins

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, base_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(build_conv_layer(self.conv_cfg, in_channels, base_channels // 2, kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, base_channels // 2)[1], nn.ReLU(inplace=True), build_conv_layer(self.conv_cfg, base_channels // 2, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, base_channels // 2)[1], nn.ReLU(inplace=True), build_conv_layer(self.conv_cfg, base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, base_channels)[1], nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(self.conv_cfg, in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, base_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_cur_norm(self, key, i):
        if key == 'second' and i == 0:
            return getattr(self, self.eb_second_bn1_name)
        if key == 'second' and i == 1:
            return getattr(self, self.eb_second_bn2_name)
        if key == 'second' and i == 2:
            return getattr(self, self.eb_second_bn3_name)
        if key == 'second' and i == 3:
            return getattr(self, self.eb_second_bn4_name)
        if key == 'third' and i == 0:
            return getattr(self, self.eb_third_bn1_name)
        if key == 'third' and i == 1:
            return getattr(self, self.eb_third_bn2_name)
        if key == 'third' and i == 2:
            return getattr(self, self.eb_third_bn3_name)
        if key == 'third' and i == 3:
            return getattr(self, self.eb_third_bn4_name)
        if key == 'four' and i == 0:
            return getattr(self, self.eb_four_bn1_name)
        if key == 'four' and i == 1:
            return getattr(self, self.eb_four_bn2_name)
        if key == 'four' and i == 2:
            return getattr(self, self.eb_four_bn3_name)
        if key == 'four' and i == 3:
            return getattr(self, self.eb_four_bn4_name)
        assert 1 > 5

    def get_cur_conv(self, key, i):
        if key == 'second' and i == 0:
            return self.eb_second_conv1
        if key == 'second' and i == 1:
            return self.eb_second_conv2
        if key == 'second' and i == 2:
            return self.eb_second_conv3
        if key == 'second' and i == 3:
            return self.eb_second_conv4
        if key == 'third' and i == 0:
            return self.eb_third_conv1
        if key == 'third' and i == 1:
            return self.eb_third_conv2
        if key == 'third' and i == 2:
            return self.eb_third_conv3
        if key == 'third' and i == 3:
            return self.eb_third_conv4
        if key == 'four' and i == 0:
            return self.eb_four_conv1
        if key == 'four' and i == 1:
            return self.eb_four_conv2
        if key == 'four' and i == 2:
            return self.eb_four_conv3
        if key == 'four' and i == 3:
            return self.eb_four_conv4
        assert 1 > 5

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        second_x = x
        third_x = x
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            second_layer_name = self.second_res_layers[i]
            second_res_layer = getattr(self, second_layer_name)
            second_x = second_res_layer(second_x)
            cur_second_norm = self.get_cur_norm('second', i)
            cur_second_conv = self.get_cur_conv('second', i)
            tmp = cur_second_conv(second_x)
            tmp = cur_second_norm(tmp)
            if i != 0:
                tmp = self.eb_upsample(tmp)
            x = x + tmp
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(DB_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class DB_ResNetV1d(DB_ResNet):
    """ResNetV1d variant described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 2x2 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    """

    def __init__(self, **kwargs):
        super(DB_ResNetV1d, self).__init__(deep_stem=True, avg_down=True, **kwargs)


class HRModule(nn.Module):
    """ High-Resolution Module for HRNet. In this module, every branch
    has 4 BasicBlocks/Bottlenecks. Fusion/Exchange is in this module.
    """

    def __init__(self, num_branches, blocks, num_blocks, in_channels, num_channels, multiscale_output=True, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN')):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(build_conv_layer(self.conv_cfg, self.in_channels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(self.norm_cfg, num_channels[branch_index] * block.expansion)[1])
        layers = []
        layers.append(block(self.in_channels[branch_index], num_channels[branch_index], stride, downsample=downsample, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index], num_channels[branch_index], with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=1, stride=1, padding=0, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1], nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[i], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels[j], in_channels[j], kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, in_channels[j])[1], nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """HRNet backbone.

    High-Resolution Representations for Labeling Pixels and Regions
    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Normally 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self, extra, in_channels=3, conv_cfg=None, norm_cfg=dict(type='BN'), norm_eval=True, with_cp=False, zero_init_residual=False):
        super(HRNet, self).__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)
        self.conv1 = build_conv_layer(self.conv_cfg, in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(self.conv_cfg, 64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']
        block = self.blocks_dict[block_type]
        num_channels = [(channel * block.expansion) for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(build_conv_layer(self.conv_cfg, num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1], nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(nn.Sequential(build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, out_channels)[1], nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(build_conv_layer(self.conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(self.norm_cfg, planes * block.expansion)[1])
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        hr_modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            hr_modules.append(HRModule(num_branches, block, num_blocks, in_channels, num_channels, reset_multiscale_output, with_cp=self.with_cp, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg))
        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def train(self, mode=True):
        super(HRNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class Res2Layer(nn.Sequential):
    """Res2Layer to build Res2Net style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scales (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
    """

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, avg_down=True, conv_cfg=None, norm_cfg=dict(type='BN'), scales=4, base_width=26, **kwargs):
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False), build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1])
        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, scales=scales, base_width=base_width, stage_type='stage', **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, scales=scales, base_width=base_width, **kwargs))
        super(Res2Layer, self).__init__(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, in_channels=3, base_channels=64, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), style='pytorch', deep_stem=False, avg_down=False, frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, dcn=None, stage_with_dcn=(False, False, False, False), plugins=None, with_cp=False, zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = base_channels
        self._make_stem_layer(in_channels, base_channels)
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            res_layer = self.make_res_layer(block=self.block, inplanes=self.inplanes, planes=planes, num_blocks=num_blocks, stride=stride, dilation=dilation, style=self.style, avg_down=self.avg_down, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """ make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.
        An example of plugins format could be:

        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)
        return stage_plugins

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, base_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(build_conv_layer(self.conv_cfg, in_channels, base_channels // 2, kernel_size=3, stride=2, padding=1, bias=False), build_norm_layer(self.norm_cfg, base_channels // 2)[1], nn.ReLU(inplace=True), build_conv_layer(self.conv_cfg, base_channels // 2, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, base_channels // 2)[1], nn.ReLU(inplace=True), build_conv_layer(self.conv_cfg, base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1, bias=False), build_norm_layer(self.norm_cfg, base_channels)[1], nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(self.conv_cfg, in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, base_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class ResNetV1d(ResNet):
    """ResNetV1d variant described in
    `Bag of Tricks <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv
    in the input stem with three 3x3 convs. And in the downsampling block,
    a 2x2 avg_pool with stride 2 is added before conv, whose stride is
    changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(deep_stem=True, avg_down=True, **kwargs)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) * x_float / norm).type_as(x)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & (flat_anchors[:, 0] >= -allowed_border) & (flat_anchors[:, 1] >= -allowed_border) & (flat_anchors[:, 2] < img_w + allowed_border) & (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)


def build_assigner(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_bbox_coder(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_CODERS, default_args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_sampler(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            output = old_func(*new_args, **new_kwargs)
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output
        return new_func
    return force_fp32_wrapper


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def batched_nms(bboxes, scores, inds, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        bboxes (torch.Tensor): bboxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        nms_cfg (dict): specify nms type and class_agnostic as well as other
            parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all bboxes,
            regardless of the predicted class

    Returns:
        tuple: kept bboxes and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        max_coordinate = bboxes.max()
        offsets = inds * (max_coordinate + 1)
        bboxes_for_nms = bboxes + offsets[:, None]
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    dets, keep = nms_op(torch.cat([bboxes_for_nms, scores[:, None]], -1), **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    return torch.cat([bboxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]
    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    return dets, labels[keep]


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, anchor_generator=dict(type='AnchorGenerator', scales=[8, 16, 32], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]), bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)), reg_decoded_bbox=False, background_label=None, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0), train_cfg=None, test_cfg=None):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox
        self.background_label = num_classes if background_label is None else background_label
        assert self.background_label == 0 or self.background_label == num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image
                valid_flag_list (list[Tensor]): Valid flags of each image
        """
        num_imgs = len(img_metas)
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 6
        anchors = flat_anchors[inside_flags, :]
        assign_result = self.assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.background_label, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, sampling_result

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True, return_sampling_results=False):
        """Compute regression and classification targets for anchors in
            multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end

        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(self._get_targets_single, concat_anchor_list, concat_valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list, sampling_results_list = results[:7]
        rest_results = list(results[7:])
        if any([(labels is None) for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        res = labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg
        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):
            rest_results[i] = images_to_levels(r, num_level_anchors)
        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, rescale=False):
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Size / scale info for each image
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, mlvl_anchors):
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
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class ATSSHead(AnchorHead):
    """
    Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ATSSHead, self).__init__(num_classes, in_channels, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.atss_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.atss_centerness = nn.Conv2d(self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_generator.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.atss_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels, label_weights, bbox_targets, num_total_samples):
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]
            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(pos_anchors, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=centerness_targets, avg_factor=1.0)
            loss_centerness = self.loss_centerness(pos_centerness, centerness_targets, avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = torch.tensor(0)
        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = reduce_mean(torch.tensor(num_total_pos)).item()
        num_total_samples = max(num_total_samples, 1.0)
        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(self.loss_single, anchor_list, cls_scores, bbox_preds, centernesses, labels_list, label_weights_list, bbox_targets_list, num_total_samples=num_total_samples)
        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

    def centerness_target(self, anchors, bbox_targets):
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy
        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses, img_metas, cfg=None, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            centerness_pred_list = [centernesses[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, centernesses, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, label_channels=1, unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._get_target_single, anchor_list, valid_flag_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, img_metas, label_channels=label_channels, unmap_outputs=unmap_outputs)
        if any([(labels is None) for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return anchors_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg

    def _get_target_single(self, flat_anchors, valid_flags, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, label_channels=1, unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 6
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.background_label, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside


INF = 100000000.0


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


class FCOSHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4, strides=(4, 8, 16, 32, 64), regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)), center_sampling=False, center_sample_radius=1.5, background_label=None, loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox=dict(type='IoULoss', loss_weight=1.0), loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), train_cfg=None, test_cfg=None):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.background_label = num_classes if background_label is None else background_label
        assert self.background_label == 0 or self.background_label == num_classes
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets, avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self, cls_scores, bbox_preds, centernesses, img_metas, cfg=None, rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            centerness_pred_list = [centernesses[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points, img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, centernesses, mlvl_points, img_shape, scale_factor, cfg, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(self._get_points_single(featmap_sizes[i], self.strides[i], dtype, device))
        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        labels_list, bbox_targets_list = multi_apply(self._get_target_single, gt_bboxes_list, gt_labels_list, points=concat_points, regress_ranges=concat_regress_ranges, num_points_per_lvl=num_points)
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), gt_bboxes.new_zeros((num_points, 4))
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1])
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.background_label
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')
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
            deform_conv_ext.deform_conv_forward(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
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
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_ext.deform_conv_backward_input(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_ext.deform_conv_backward_parameters(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
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
            raise ValueError(f"convolution input is too small (output would be {'x'.join(map(str, output_size))})")
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, f'in_channels {in_channels} is not divisible by groups {groups}'
        assert out_channels % groups == 0, f'out_channels {out_channels} is not divisible by groups {groups}'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        input_pad = x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1]
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) - pad_w].contiguous()
        return out


class FeatureAlign(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x


class FoveaHead(nn.Module):
    """FoveaBox: Beyond Anchor-based Object Detector
    https://arxiv.org/abs/1904.03797
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=4, strides=(4, 8, 16, 32, 64), base_edge_list=(16, 32, 64, 128, 256), scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)), sigma=0.4, with_deform=False, deformable_groups=4, background_label=None, loss_cls=None, loss_bbox=None, conv_cfg=None, norm_cfg=None, train_cfg=None, test_cfg=None):
        super(FoveaHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deformable_groups = deformable_groups
        self.background_label = num_classes if background_label is None else background_label
        assert self.background_label == 0 or self.background_label == num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
        self.fovea_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        if not self.with_deform:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.fovea_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.cls_convs.append(ConvModule(self.feat_channels, self.feat_channels * 4, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.cls_convs.append(ConvModule(self.feat_channels * 4, self.feat_channels * 4, 1, stride=1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None))
            self.feature_adaption = FeatureAlign(self.feat_channels, self.feat_channels, kernel_size=3, deformable_groups=self.deformable_groups)
            self.fovea_cls = nn.Conv2d(int(self.feat_channels * 4), self.cls_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fovea_cls, std=0.01, bias=bias_cls)
        normal_init(self.fovea_reg, std=0.01)
        if self.with_deform:
            self.feature_adaption.init_weights()

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.fovea_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fovea_cls(cls_feat)
        return cls_score, bbox_pred

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        points = []
        for featmap_size in featmap_sizes:
            x_range = torch.arange(featmap_size[1], dtype=dtype, device=device) + 0.5
            y_range = torch.arange(featmap_size[0], dtype=dtype, device=device) + 0.5
            y, x = torch.meshgrid(y_range, x_range)
            if flatten:
                points.append((y.flatten(), x.flatten()))
            else:
                points.append((y, x))
        return points

    def loss(self, cls_scores, bbox_preds, gt_bbox_list, gt_label_list, img_metas, gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels, flatten_bbox_targets = self.get_targets(gt_bbox_list, gt_label_list, featmap_sizes, points)
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.background_label)).nonzero().view(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_zeros(pos_bbox_targets.size()) + 1.0
            loss_bbox = self.loss_bbox(pos_bbox_preds, pos_bbox_targets, pos_weights, avg_factor=num_pos)
        else:
            loss_bbox = torch.tensor(0, dtype=flatten_bbox_preds.dtype, device=flatten_bbox_preds.device)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self, gt_bbox_list, gt_label_list, featmap_sizes, points):
        label_list, bbox_target_list = multi_apply(self._get_target_single, gt_bbox_list, gt_label_list, featmap_size_list=featmap_sizes, point_list=points)
        flatten_labels = [torch.cat([labels_level_img.flatten() for labels_level_img in labels_level]) for labels_level in zip(*label_list)]
        flatten_bbox_targets = [torch.cat([bbox_targets_level_img.reshape(-1, 4) for bbox_targets_level_img in bbox_targets_level]) for bbox_targets_level in zip(*bbox_target_list)]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        return flatten_labels, flatten_bbox_targets

    def _get_target_single(self, gt_bboxes_raw, gt_labels_raw, featmap_size_list=None, point_list=None):
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        for base_len, (lower_bound, upper_bound), stride, featmap_size, (y, x) in zip(self.base_edge_list, self.scale_ranges, self.strides, featmap_size_list, point_list):
            labels = gt_labels_raw.new_zeros(featmap_size) + self.num_classes
            bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1], 4) + 1
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue
            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            pos_left = torch.ceil(gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long().clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long().clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long().clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long().clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in zip(pos_left, pos_top, pos_right, pos_down, gt_labels, gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = (stride * x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = (stride * y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = (gt_x2 - stride * x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = (gt_y2 - stride * y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1.0 / 16, max=16.0)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
        return label_list, bbox_target_list

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg=None, rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device, flatten=True)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list, bbox_pred_list, featmap_sizes, points, img_shape, scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, featmap_sizes, point_list, img_shape, scale_factor, cfg, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        for cls_score, bbox_pred, featmap_size, stride, base_len, (y, x) in zip(cls_scores, bbox_preds, featmap_sizes, self.strides, self.base_edge_list, point_list):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                y = y[topk_inds]
                x = x[topk_inds]
            x1 = (stride * x - base_len * bbox_pred[:, 0]).clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:, 1]).clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, 2]).clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, 3]).clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)
        det_bboxes = torch.cat(det_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_scores = torch.cat(det_scores)
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        det_scores = torch.cat([det_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(det_bboxes, det_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(2, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x


class MaskedConv2dFunction(Function):

    @staticmethod
    def forward(ctx, features, mask, weight, bias, padding=0, stride=1):
        assert mask.dim() == 3 and mask.size(0) == 1
        assert features.dim() == 4 and features.size(0) == 1
        assert features.size()[2:] == mask.size()[1:]
        pad_h, pad_w = _pair(padding)
        stride_h, stride_w = _pair(stride)
        if stride_h != 1 or stride_w != 1:
            raise ValueError('Stride could not only be 1 in masked_conv2d currently.')
        if not features.is_cuda:
            raise NotImplementedError
        out_channel, in_channel, kernel_h, kernel_w = weight.size()
        batch_size = features.size(0)
        out_h = int(math.floor((features.size(2) + 2 * pad_h - (kernel_h - 1) - 1) / stride_h + 1))
        out_w = int(math.floor((features.size(3) + 2 * pad_w - (kernel_h - 1) - 1) / stride_w + 1))
        mask_inds = torch.nonzero(mask[0] > 0, as_tuple=False)
        output = features.new_zeros(batch_size, out_channel, out_h, out_w)
        if mask_inds.numel() > 0:
            mask_h_idx = mask_inds[:, 0].contiguous()
            mask_w_idx = mask_inds[:, 1].contiguous()
            data_col = features.new_zeros(in_channel * kernel_h * kernel_w, mask_inds.size(0))
            masked_conv2d_ext.masked_im2col_forward(features, mask_h_idx, mask_w_idx, kernel_h, kernel_w, pad_h, pad_w, data_col)
            masked_output = torch.addmm(1, bias[:, None], 1, weight.view(out_channel, -1), data_col)
            masked_conv2d_ext.masked_col2im_forward(masked_output, mask_h_idx, mask_w_idx, out_h, out_w, out_channel, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return (None,) * 5


masked_conv2d = MaskedConv2dFunction.apply


class MaskedConv2d(nn.Conv2d):
    """A MaskedConv2d which inherits the official Conv2d.

    The masked forward doesn't implement the backward function and only
    supports the stride parameter to be 1 currently.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input, mask=None):
        if mask is None:
            return super(MaskedConv2d, self).forward(input)
        else:
            return masked_conv2d(input, mask, self.weight, self.bias, self.padding)


def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return x1, y1, x2, y2


class GuidedAnchorHead(AnchorHead):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    This GuidedAnchorHead will predict high-quality feature guided
    anchors and locations where anchors will be kept in inference.
    There are mainly 3 categories of bounding-boxes.

    - Sampled 9 pairs for target assignment. (approxes)
    - The square boxes where the predicted anchors are based on. (squares)
    - Guided anchors.

    Please refer to https://arxiv.org/abs/1901.03278 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels.
        approx_anchor_generator (dict): Config dict for approx generator
        square_anchor_generator (dict): Config dict for square generator
        anchor_coder (dict): Config dict for anchor coder
        bbox_coder (dict): Config dict for bbox coder
        deformable_groups: (int): Group number of DCN in
            FeatureAdaption module.
        loc_filter_thr (float): Threshold to filter out unconcerned regions.
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_loc (dict): Config of location loss.
        loss_shape (dict): Config of anchor shape loss.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of bbox regression loss.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, approx_anchor_generator=dict(type='AnchorGenerator', octave_base_scale=8, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]), square_anchor_generator=dict(type='AnchorGenerator', ratios=[1.0], scales=[8], strides=[4, 8, 16, 32, 64]), anchor_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]), bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]), reg_decoded_bbox=False, deformable_groups=4, loc_filter_thr=0.01, background_label=None, train_cfg=None, test_cfg=None, loss_loc=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.deformable_groups = deformable_groups
        self.loc_filter_thr = loc_filter_thr
        assert approx_anchor_generator['octave_base_scale'] == square_anchor_generator['scales'][0]
        assert approx_anchor_generator['strides'] == square_anchor_generator['strides']
        self.approx_anchor_generator = build_anchor_generator(approx_anchor_generator)
        self.square_anchor_generator = build_anchor_generator(square_anchor_generator)
        self.approxs_per_octave = self.approx_anchor_generator.num_base_anchors[0]
        self.reg_decoded_bbox = reg_decoded_bbox
        self.background_label = num_classes if background_label is None else background_label
        assert self.background_label == 0 or self.background_label == num_classes
        self.num_anchors = 1
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.ga_sampling = train_cfg is not None and hasattr(train_cfg, 'ga_sampler')
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.anchor_coder = build_bbox_coder(anchor_coder)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_loc = build_loss(loss_loc)
        self.loss_shape = build_loss(loss_shape)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.ga_assigner = build_assigner(self.train_cfg.ga_assigner)
            if self.ga_sampling:
                ga_sampler_cfg = self.train_cfg.ga_sampler
            else:
                ga_sampler_cfg = dict(type='PseudoSampler')
            self.ga_sampler = build_sampler(ga_sampler_cfg, context=self)
        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.in_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.in_channels, self.num_anchors * 2, 1)
        self.feature_adaption = FeatureAdaption(self.in_channels, self.feat_channels, kernel_size=3, deformable_groups=self.deformable_groups)
        self.conv_cls = MaskedConv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = MaskedConv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        self.feature_adaption.init_weights()

    def forward_single(self, x):
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_sampled_approxs(self, featmap_sizes, img_metas, device='cuda'):
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(img_metas)
        multi_level_approxs = self.approx_anchor_generator.grid_anchors(featmap_sizes, device=device)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]
        inside_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]
            multi_level_approx_flags = self.approx_anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device=device)
            for i, flags in enumerate(multi_level_approx_flags):
                approxs = multi_level_approxs[i]
                inside_flags_list = []
                for i in range(self.approxs_per_octave):
                    split_valid_flags = flags[i::self.approxs_per_octave]
                    split_approxs = approxs[i::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(split_approxs, split_valid_flags, img_meta['img_shape'][:2], self.train_cfg.allowed_border)
                    inside_flags_list.append(inside_flags)
                inside_flags = torch.stack(inside_flags_list, 0).sum(dim=0) > 0
                multi_level_flags.append(inside_flags)
            inside_flag_list.append(multi_level_flags)
        return approxs_list, inside_flag_list

    def get_anchors(self, featmap_sizes, shape_preds, loc_preds, img_metas, use_loc_filter=False, device='cuda'):
        """Get squares according to feature map sizes and guided
        anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
                loc masks of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_squares = self.square_anchor_generator.grid_anchors(featmap_sizes, device=device)
        squares_list = [multi_level_squares for _ in range(num_imgs)]
        guided_anchors_list = []
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self._get_guided_anchors_single(squares, shape_pred, loc_pred, use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return squares_list, guided_anchors_list, loc_mask_list

    def _get_guided_anchors_single(self, squares, shape_pred, loc_pred, use_loc_filter=False):
        """Get guided anchors and loc masks for a single level.

        Args:
            square (tensor): Squares of a single level.
            shape_pred (tensor): Shape predections of a single level.
            loc_pred (tensor): Loc predections of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.

        Returns:
            tuple: guided anchors, location masks
        """
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
        mask = mask.contiguous().view(-1)
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(-1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        guided_anchors = self.anchor_coder.decode(squares, bbox_deltas, wh_ratio_clip=1e-06)
        return guided_anchors, mask

    def ga_loc_targets(self, gt_bboxes_list, featmap_sizes):
        """Compute location targets for guided anchoring.

        Each feature map is divided into positive, negative and ignore regions.
        - positive regions: target 1, weight 1
        - ignore regions: target 0, weight 0
        - negative regions: target 0, weight 0.1

        Args:
            gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
            featmap_sizes (list[tuple]): Multi level sizes of each feature
                maps.

        Returns:
            tuple
        """
        anchor_scale = self.approx_anchor_generator.octave_base_scale
        anchor_strides = self.approx_anchor_generator.strides
        for stride in anchor_strides:
            assert stride[0] == stride[1]
        anchor_strides = [stride[0] for stride in anchor_strides]
        center_ratio = self.train_cfg.center_ratio
        ignore_ratio = self.train_cfg.ignore_ratio
        img_per_gpu = len(gt_bboxes_list)
        num_lvls = len(featmap_sizes)
        r1 = (1 - center_ratio) / 2
        r2 = (1 - ignore_ratio) / 2
        all_loc_targets = []
        all_loc_weights = []
        all_ignore_map = []
        for lvl_id in range(num_lvls):
            h, w = featmap_sizes[lvl_id]
            loc_targets = torch.zeros(img_per_gpu, 1, h, w, device=gt_bboxes_list[0].device, dtype=torch.float32)
            loc_weights = torch.full_like(loc_targets, -1)
            ignore_map = torch.zeros_like(loc_targets)
            all_loc_targets.append(loc_targets)
            all_loc_weights.append(loc_weights)
            all_ignore_map.append(ignore_map)
        for img_id in range(img_per_gpu):
            gt_bboxes = gt_bboxes_list[img_id]
            scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
            min_anchor_size = scale.new_full((1,), float(anchor_scale * anchor_strides[0]))
            target_lvls = torch.floor(torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
            target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
            for gt_id in range(gt_bboxes.size(0)):
                lvl = target_lvls[gt_id].item()
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[lvl])
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(gt_, r1, featmap_sizes[lvl])
                all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 0
                all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
                if lvl > 0:
                    d_lvl = lvl - 1
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[d_lvl])
                    all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 1
                if lvl < num_lvls - 1:
                    u_lvl = lvl + 1
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(gt_, r2, featmap_sizes[u_lvl])
                    all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 + 1, ignore_x1:ignore_x2 + 1] = 1
        for lvl_id in range(num_lvls):
            all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0) & (all_ignore_map[lvl_id] > 0)] = 0
            all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
        loc_avg_factor = sum([(t.size(0) * t.size(-1) * t.size(-2)) for t in all_loc_targets]) / 200
        return all_loc_targets, all_loc_weights, loc_avg_factor

    def _ga_shape_target_single(self, flat_approxs, inside_flags, flat_squares, gt_bboxes, gt_bboxes_ignore, img_meta, unmap_outputs=True):
        """Compute guided anchoring targets.

        This function returns sampled anchors and gt bboxes directly
        rather than calculates regression targets.

        Args:
            flat_approxs (Tensor): flat approxs of a single image,
                shape (n, 4)
            inside_flags (Tensor): inside flags of a single image,
                shape (n, ).
            flat_squares (Tensor): flat squares of a single image,
                shape (approxs_per_octave * n, 4)
            gt_bboxes (Tensor): Ground truth bboxes of a single image.
            img_meta (dict): Meta info of a single image.
            approxs_per_octave (int): number of approxs per octave
            cfg (dict): RPN train configs.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple
        """
        if not inside_flags.any():
            return (None,) * 5
        expand_inside_flags = inside_flags[:, None].expand(-1, self.approxs_per_octave).reshape(-1)
        approxs = flat_approxs[expand_inside_flags, :]
        squares = flat_squares[inside_flags, :]
        assign_result = self.ga_assigner.assign(approxs, squares, self.approxs_per_octave, gt_bboxes, gt_bboxes_ignore)
        sampling_result = self.ga_sampler.sample(assign_result, squares, gt_bboxes)
        bbox_anchors = torch.zeros_like(squares)
        bbox_gts = torch.zeros_like(squares)
        bbox_weights = torch.zeros_like(squares)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_anchors[pos_inds, :] = sampling_result.pos_bboxes
            bbox_gts[pos_inds, :] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
        if unmap_outputs:
            num_total_anchors = flat_squares.size(0)
            bbox_anchors = unmap(bbox_anchors, num_total_anchors, inside_flags)
            bbox_gts = unmap(bbox_gts, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        return bbox_anchors, bbox_gts, bbox_weights, pos_inds, neg_inds

    def ga_shape_targets(self, approx_list, inside_flag_list, square_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, unmap_outputs=True):
        """Compute guided anchoring targets.

        Args:
            approx_list (list[list]): Multi level approxs of each image.
            inside_flag_list (list[list]): Multi level inside flags of each
                image.
            square_list (list[list]): Multi level squares of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple
        """
        num_imgs = len(img_metas)
        assert len(approx_list) == len(inside_flag_list) == len(square_list) == num_imgs
        num_level_squares = [squares.size(0) for squares in square_list[0]]
        inside_flag_flat_list = []
        approx_flat_list = []
        square_flat_list = []
        for i in range(num_imgs):
            assert len(square_list[i]) == len(inside_flag_list[i])
            inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
            approx_flat_list.append(torch.cat(approx_list[i]))
            square_flat_list.append(torch.cat(square_list[i]))
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        all_bbox_anchors, all_bbox_gts, all_bbox_weights, pos_inds_list, neg_inds_list = multi_apply(self._ga_shape_target_single, approx_flat_list, inside_flag_flat_list, square_flat_list, gt_bboxes_list, gt_bboxes_ignore_list, img_metas, unmap_outputs=unmap_outputs)
        if any([(bbox_anchors is None) for bbox_anchors in all_bbox_anchors]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        bbox_anchors_list = images_to_levels(all_bbox_anchors, num_level_squares)
        bbox_gts_list = images_to_levels(all_bbox_gts, num_level_squares)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_squares)
        return bbox_anchors_list, bbox_gts_list, bbox_weights_list, num_total_pos, num_total_neg

    def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts, anchor_weights, anchor_total_num):
        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        inds = torch.nonzero(anchor_weights[:, 0] > 0, as_tuple=False).squeeze(1)
        bbox_deltas_ = bbox_deltas[inds]
        bbox_anchors_ = bbox_anchors[inds]
        bbox_gts_ = bbox_gts[inds]
        anchor_weights_ = anchor_weights[inds]
        pred_anchors_ = self.anchor_coder.decode(bbox_anchors_, bbox_deltas_, wh_ratio_clip=1e-06)
        loss_shape = self.loss_shape(pred_anchors_, bbox_gts_, anchor_weights_, avg_factor=anchor_total_num)
        return loss_shape

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor):
        loss_loc = self.loss_loc(loc_pred.reshape(-1, 1), loc_target.reshape(-1, 1).long(), loc_weight.reshape(-1, 1), avg_factor=loc_avg_factor)
        return loss_loc

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def loss(self, cls_scores, bbox_preds, shape_preds, loc_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.approx_anchor_generator.num_levels
        device = cls_scores[0].device
        loc_targets, loc_weights, loc_avg_factor = self.ga_loc_targets(gt_bboxes, featmap_sizes)
        approxs_list, inside_flag_list = self.get_sampled_approxs(featmap_sizes, img_metas, device=device)
        squares_list, guided_anchors_list, _ = self.get_anchors(featmap_sizes, shape_preds, loc_preds, img_metas, device=device)
        shape_targets = self.ga_shape_targets(approxs_list, inside_flag_list, squares_list, gt_bboxes, img_metas)
        if shape_targets is None:
            return None
        bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num, anchor_bg_num = shape_targets
        anchor_total_num = anchor_fg_num if not self.ga_sampling else anchor_fg_num + anchor_bg_num
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(guided_anchors_list, inside_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_level_anchors = [anchors.size(0) for anchors in guided_anchors_list[0]]
        concat_anchor_list = []
        for i in range(len(guided_anchors_list)):
            concat_anchor_list.append(torch.cat(guided_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)
        losses_cls, losses_bbox = multi_apply(self.loss_single, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples=num_total_samples)
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(loc_preds[i], loc_targets[i], loc_weights[i], loc_avg_factor=loc_avg_factor)
            losses_loc.append(loss_loc)
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(shape_preds[i], bbox_anchors_list[i], bbox_gts_list[i], anchor_weights_list[i], anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_shape=losses_shape, loss_loc=losses_loc)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, shape_preds, loc_preds, img_metas, cfg=None, rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        _, guided_anchors, loc_masks = self.get_anchors(featmap_sizes, shape_preds, loc_preds, img_metas, use_loc_filter=not self.training, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            guided_anchor_list = [guided_anchors[img_id][i].detach() for i in range(num_levels)]
            loc_mask_list = [loc_masks[img_id][i].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, guided_anchor_list, loc_mask_list, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, mlvl_masks, img_shape, scale_factor, cfg, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds, mlvl_anchors, mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            if mask.sum() == 0:
                continue
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels


class NASFCOSHead(FCOSHead):
    """Anchor-free head used in `NASFCOS <https://arxiv.org/abs/1906.04423>`_.

    It is quite similar with FCOS head, except for the searched structure
    of classification branch and bbox regression branch, where a structure
    of "dconv3x3, conv3x3, dconv3x3, conv1x1" is utilized instead.

    """

    def _init_layers(self):
        dconv3x3_config = dict(type='DCNv2', kernel_size=3, use_bias=True, deformable_groups=2, padding=1)
        conv3x3_config = dict(type='Conv', kernel_size=3, padding=1)
        conv1x1_config = dict(type='Conv', kernel_size=1)
        self.arch_config = [dconv3x3_config, conv3x3_config, dconv3x3_config, conv1x1_config]
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i, op_ in enumerate(self.arch_config):
            op = copy.deepcopy(op_)
            chn = self.in_channels if i == 0 else self.feat_channels
            assert isinstance(op, dict)
            use_bias = op.pop('use_bias', False)
            padding = op.pop('padding', 0)
            kernel_size = op.pop('kernel_size')
            module = ConvModule(chn, self.feat_channels, kernel_size, stride=1, padding=padding, norm_cfg=self.norm_cfg, bias=use_bias, conv_cfg=op)
            self.cls_convs.append(copy.deepcopy(module))
            self.reg_convs.append(copy.deepcopy(module))
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        for branch in [self.cls_convs, self.reg_convs]:
            for module in branch.modules():
                if isinstance(module, ConvModule) and isinstance(module.conv, nn.Conv2d):
                    caffe2_xavier_init(module.conv)


class PointGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0.0, feat_w, device=device) * stride
        shift_y = torch.arange(0.0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0],), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid


class RepPointsHead(nn.Module):
    """RepPoint head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """

    def __init__(self, num_classes, in_channels, feat_channels=256, point_feat_channels=256, stacked_convs=3, num_points=9, gradient_mul=0.1, point_strides=[8, 16, 32, 64, 128], point_base_scale=4, conv_cfg=None, norm_cfg=None, background_label=None, loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5), loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0), use_grid_points=False, center_init=True, transform_method='moment', moment_mul=0.01, train_cfg=None, test_cfg=None):
        super(RepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.background_label = num_classes if background_label is None else background_label
        assert self.background_label == 0 or self.background_label == num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, 'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, 'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(-1)
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = self.moment_transfer * self.moment_mul + self.moment_transfer.detach() * (1 - self.moment_mul)
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([pts_x_mean - half_width, pts_y_mean - half_height, pts_x_mean + half_width, pts_y_mean + half_height], dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.0
        bwh = (previous_boxes[:, 2:, ...] - previous_boxes[:, :2, ...]).clamp(min=1e-06)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0.0, 1.0, self.dcn_kernel).view(1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([grid_left, grid_top, grid_left + grid_width, grid_top + grid_height], 1)
        return grid_yx, regressed_bbox

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale, scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        return cls_out, pts_out_init, pts_out_refine

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale, scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate.
        """
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self, flat_proposals, valid_flags, gt_bboxes, gt_bboxes_ignore, gt_labels, label_channels=1, stage='init', unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None,) * 7
        proposals = flat_proposals[inside_flags, :]
        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore, None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals, gt_bboxes)
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals,), self.background_label, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals, inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals, inside_flags)
        return labels, label_weights, bbox_gt, pos_proposals, proposals_weights, pos_inds, neg_inds

    def get_targets(self, proposals_list, valid_flag_list, gt_bboxes_list, img_metas, gt_bboxes_ignore_list=None, gt_labels_list=None, stage='init', label_channels=1, unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_labels, all_label_weights, all_bbox_gt, all_proposals, all_proposal_weights, pos_inds_list, neg_inds_list = multi_apply(self._point_target_single, proposals_list, valid_flag_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, stage=stage, label_channels=label_channels, unmap_outputs=unmap_outputs)
        if any([(labels is None) for labels in all_labels]):
            return None
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights, num_level_proposals)
        return labels_list, label_weights_list, bbox_gt_list, proposals_list, proposal_weights_list, num_total_pos, num_total_neg

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels, label_weights, bbox_gt_init, bbox_weights_init, bbox_gt_refine, bbox_weights_refine, stride, num_total_samples_init, num_total_samples_refine):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples_refine)
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(bbox_pred_init / normalize_term, bbox_gt_init / normalize_term, bbox_weights_init, avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(bbox_pred_refine / normalize_term, bbox_gt_refine / normalize_term, bbox_weights_refine, avg_factor=num_total_samples_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self, cls_scores, pts_preds_init, pts_preds_refine, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list, pts_preds_init)
        if self.train_cfg.init.assigner['type'] == 'PointAssigner':
            candidate_list = center_list
        else:
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = self.get_targets(candidate_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, stage='init', label_channels=label_channels)
        *_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init, num_total_pos_init, num_total_neg_init = cls_reg_targets_init
        num_total_samples_init = num_total_pos_init + num_total_neg_init if self.sampling else num_total_pos_init
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(bbox_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, stage='refine', label_channels=label_channels)
        labels_list, label_weights_list, bbox_gt_list_refine, candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine = cls_reg_targets_refine
        num_total_samples_refine = num_total_pos_refine + num_total_neg_refine if self.sampling else num_total_pos_refine
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(self.loss_single, cls_scores, pts_coordinate_preds_init, pts_coordinate_preds_refine, labels_list, label_weights_list, bbox_gt_list_init, bbox_weights_list_init, bbox_gt_list_refine, bbox_weights_list_refine, self.point_strides, num_total_samples_init=num_total_samples_init, num_total_samples_refine=num_total_samples_refine)
        loss_dict_all = {'loss_cls': losses_cls, 'loss_pts_init': losses_pts_init, 'loss_pts_refine': losses_pts_refine}
        return loss_dict_all

    def get_bboxes(self, cls_scores, pts_preds_init, pts_preds_refine, img_metas, cfg=None, rescale=False, nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        bbox_preds_refine = [self.points2bbox(pts_pred_refine) for pts_pred_refine in pts_preds_refine]
        num_levels = len(cls_scores)
        mlvl_points = [self.point_generators[i].grid_points(cls_scores[i].size()[-2:], self.point_strides[i]) for i in range(num_levels)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds_refine[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_points, img_shape, scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, mlvl_points, img_shape, scale_factor, cfg, rescale=False, nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(zip(cls_scores, bbox_preds, mlvl_points)):
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
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores


class RetinaHead(AnchorHead):
    """An anchor-based head used in
    `RetinaNet <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self, num_classes, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=None, anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]), **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(num_classes, in_channels, anchor_generator=anchor_generator, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred


class RetinaSepBNHead(AnchorHead):
    """"RetinaHead with separate BN.

    In RetinaHead, conv/norm layers are shared across different FPN levels,
    while in RetinaSepBNHead, conv layers are shared across different FPN
    levels, but BN layers are separated.
    """

    def __init__(self, num_classes, num_ins, in_channels, stacked_convs=4, conv_cfg=None, norm_cfg=None, **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_ins = num_ins
        super(RetinaSepBNHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_ins):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
                reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        for i in range(self.stacked_convs):
            for j in range(1, self.num_ins):
                self.cls_convs[j][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[j][i].conv = self.reg_convs[0][i].conv
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs[0]:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs[0]:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for i, x in enumerate(feats):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds


class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(1, in_channels, background_label=0, **kwargs)

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

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(cls_scores, bbox_preds, gt_bboxes, None, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, :-1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), idx, dtype=torch.long))
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)
        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero((w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size), as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]
        nms_cfg = dict(type='nms', iou_thr=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


class SSDHead(AnchorHead):

    def __init__(self, num_classes=80, in_channels=(512, 1024, 512, 256, 256, 256), anchor_generator=dict(type='SSDAnchorGenerator', scale_major=False, input_size=300, strides=[8, 16, 32, 64, 100, 300], ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]), basesize_ratio_range=(0.1, 0.9)), background_label=None, bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]), reg_decoded_bbox=False, train_cfg=None, test_cfg=None):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1
        self.anchor_generator = build_anchor_generator(anchor_generator)
        num_anchors = self.anchor_generator.num_base_anchors
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(nn.Conv2d(in_channels[i], num_anchors[i] * 4, kernel_size=3, padding=1))
            cls_convs.append(nn.Conv2d(in_channels[i], num_anchors[i] * (num_classes + 1), kernel_size=3, padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
        self.background_label = num_classes if background_label is None else background_label
        assert self.background_label == 0 or self.background_label == num_classes
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

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

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
        pos_inds = ((labels >= 0) & (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)
        loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_weights, beta=self.train_cfg.smoothl1_beta, avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas, gt_bboxes_ignore_list=gt_bboxes_ignore, gt_labels_list=gt_labels, label_channels=1, unmap_outputs=False)
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
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))
        assert torch.isfinite(all_cls_scores).all().item(), 'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), 'bbox predications become infinite or NaN!'
        losses_cls, losses_bbox = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            args_to_cast = args_info.args if apply_to is None else apply_to
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            output = old_func(*new_args, **new_kwargs)
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output
        return new_func
    return auto_fp16_wrapper


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self.roi_head, 'shared_head') and self.roi_head.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self.roi_head, 'bbox_head') and self.roi_head.bbox_head is not None or hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self.roi_head, 'mask_head') and self.roi_head.mask_head is not None or hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log(f'load model from: {pretrained}', logger='root')

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) != num of image metas ({len(img_metas)})')
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1
        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != num of image meta ({len(img_metas)})')
        samples_per_gpu = imgs[0].size(0)
        assert samples_per_gpu == 1
        if num_augs == 1:
            """
            proposals (List[List[Tensor]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. The Tensor should have a shape Px4, where
                P is the number of proposals.
            """
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def show_result(self, img, result, score_thr=0.3, bbox_color='green', text_color='green', thickness=1, font_scale=0.5, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        if segm_result is not None and len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(max(labels) + 1)]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        if out_file is not None:
            show = False
        mmcv.imshow_det_bboxes(img, bboxes, labels, class_names=self.CLASSES, score_thr=score_thr, bbox_color=bbox_color, text_color=text_color, thickness=thickness, font_scale=font_scale, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only result image will be returned')
            return img


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
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_head(cfg):
    return build(cfg, HEADS)


def build_neck(cfg):
    return build(cfg, NECKS)


class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
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
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in bbox_list]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


logger = logging.getLogger(__name__)


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4]
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4]
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.6
        >>> suppressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(suppressed) == 3
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets)
    else:
        raise TypeError(f'dets must be either a Tensor or numpy array, but got {type(dets)}')
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    elif dets_th.is_cuda:
        inds = nms_ext.nms(dets_th, iou_thr)
    else:
        inds = nms_ext.nms(dets_th, iou_thr)
    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def merge_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and my also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

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
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = topk,
        return_single = True
    else:
        return_single = False
    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1,)):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)


@weighted_loss
def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff, gamma * diff + gamma / b - alpha * beta)
    return loss


class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, reduction='mean', loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * balanced_l1_loss(pred, target, weight, alpha=self.alpha, gamma=self.gamma, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight=class_weight, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None):
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None, class_weight=None):
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, weight=class_weight, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        loss = sigmoid_focal_loss_ext.forward(input, target, num_classes, gamma, alpha)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_ext.backward(input, target, d_loss, num_classes, gamma, alpha)
        return d_input, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply


class FocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero((labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-06
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


class GHMR(nn.Module):
    """GHM Regression Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector"
    https://arxiv.org/abs/1811.05181

    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """

    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] = 1000.0
        self.momentum = momentum
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """Calculate the GHM-R loss.

        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)
        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n
        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof']
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)
    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / area1[:, None]
    return ious


@weighted_loss
def iou_loss(pred, target, eps=1e-06):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


class IoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=0.001):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry
    loss_dx = 1 - torch.max((target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max((target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(loss_dx.size(0), -1)
    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta)
    return loss


class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=0.001, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * bounded_iou_loss(pred, target, weight, beta=self.beta, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-07):
    """
    Generalized Intersection over Union: A Metric and A Loss for
    Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    code refer to:
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
    gious = ious - (enclose_area - union) / enclose_area
    loss = 1 - gious
    return loss


class GIoULoss(nn.Module):

    def __init__(self, eps=1e-06, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction, avg_factor=avg_factor)
        return loss


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * smooth_l1_loss(pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_bbox


@weighted_loss
def l1_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


class L1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self, in_channels, num_levels, refine_level=2, refine_type=None, conv_cfg=None, norm_cfg=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']
        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels
        if self.refine_type == 'conv':
            self.refine = ConvModule(self.in_channels, self.in_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(self.in_channels, reduction=1, use_scale=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(inputs[i], size=gather_size, mode='nearest')
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


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False, no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, act_cfg=None, upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
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
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            if extra_convs_on_inputs:
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg if not self.no_norm_on_lateral else None, act_cfg=act_cfg, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class CARAFEFunction(Function):

    @staticmethod
    def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()
        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        routput = features.new_zeros(output.size(), requires_grad=False)
        rfeatures = features.new_zeros(features.size(), requires_grad=False)
        rmasks = masks.new_zeros(masks.size(), requires_grad=False)
        if features.is_cuda:
            carafe_ext.forward(features, rfeatures, masks, rmasks, kernel_size, group_size, scale_factor, routput, output)
        else:
            raise NotImplementedError
        if features.requires_grad or masks.requires_grad:
            ctx.save_for_backward(features, masks, rfeatures)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, masks, rfeatures = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor
        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input_hs = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_masks = torch.zeros_like(masks, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_masks = torch.zeros_like(masks, requires_grad=False)
        carafe_ext.backward(grad_output.contiguous(), rfeatures, masks, kernel_size, group_size, scale_factor, rgrad_output, rgrad_input_hs, rgrad_input, rgrad_masks, grad_input, grad_masks)
        return grad_input, grad_masks, None, None, None, None


carafe = CARAFEFunction.apply


class CARAFEPack(nn.Module):
    """ A unified package of CARAFE upsampler that contains:
    1) channel compressor 2) content encoder 3) CARAFE op

    Official implementation of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self, channels, scale_factor, up_kernel=5, up_group=1, encoder_kernel=3, encoder_dilation=1, compressed_channels=64):
        super(CARAFEPack, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(self.compressed_channels, self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor, self.encoder_kernel, padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2), dilation=self.encoder_dilation, groups=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / (self.up_kernel * self.up_kernel))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2)
        mask = mask.view(n, mask_c, h, w).contiguous()
        return mask

    def feature_reassemble(self, x, mask):
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)
        x = self.feature_reassemble(x, mask)
        return x


class FPN_CARAFE(nn.Module):
    """FPN_CARAFE is a more flexible implementation of FPN.
    It allows more choice for upsample methods during the top-down pathway.

    It can reproduce the preformance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        upsample (str): Type of upsample layer.
        upsample_cfg (dict): Dictionary to construct and config upsample layer.
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, norm_cfg=None, act_cfg=None, order=('conv', 'norm', 'act'), upsample_cfg=dict(type='carafe', up_kernel=5, up_group=1, encoder_kernel=3, encoder_dilation=1)):
        super(FPN_CARAFE, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample = self.upsample_cfg.get('type')
        self.relu = nn.ReLU(inplace=False)
        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]
        assert self.upsample in ['nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(self.upsample_cfg, 'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.pop('upsample_kernel')
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, norm_cfg=norm_cfg, bias=self.with_bias, act_cfg=act_cfg, inplace=False, order=self.order)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=self.norm_cfg, bias=self.with_bias, act_cfg=act_cfg, inplace=False, order=self.order)
            if i != self.backbone_end_level - 1:
                upsample_cfg_ = self.upsample_cfg.copy()
                if self.upsample == 'deconv':
                    upsample_cfg_.update(in_channels=out_channels, out_channels=out_channels, kernel_size=self.upsample_kernel, stride=2, padding=(self.upsample_kernel - 1) // 2, output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_cfg_.update(in_channels=out_channels, out_channels=out_channels, scale_factor=2, upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    upsample_cfg_.update(channels=out_channels, scale_factor=2)
                else:
                    align_corners = None if self.upsample == 'nearest' else False
                    upsample_cfg_.update(scale_factor=2, mode=self.upsample, align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_out_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = self.in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                extra_l_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, bias=self.with_bias, act_cfg=act_cfg, inplace=False, order=self.order)
                if self.upsample == 'deconv':
                    upsampler_cfg_ = dict(in_channels=out_channels, out_channels=out_channels, kernel_size=self.upsample_kernel, stride=2, padding=(self.upsample_kernel - 1) // 2, output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsampler_cfg_ = dict(in_channels=out_channels, out_channels=out_channels, scale_factor=2, upsample_kernel=self.upsample_kernel)
                elif self.upsample == 'carafe':
                    upsampler_cfg_ = dict(channels=out_channels, scale_factor=2, **self.upsample_cfg)
                else:
                    align_corners = None if self.upsample == 'nearest' else False
                    upsampler_cfg_ = dict(scale_factor=2, mode=self.upsample, align_corners=align_corners)
                upsampler_cfg_['type'] = self.upsample
                upsample_module = build_upsample_layer(upsampler_cfg_)
                extra_fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=self.norm_cfg, bias=self.with_bias, act_cfg=act_cfg, inplace=False, order=self.order)
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def slice_as(self, src, dst):
        assert src.size(2) >= dst.size(2) and src.size(3) >= dst.size(3)
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)
        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                upsample_feat = self.upsample_modules[i - 1](laterals[i])
            else:
                upsample_feat = laterals[i]
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)
        return tuple(outs)


class HRFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """

    def __init__(self, in_channels, out_channels, num_outs=5, pooling_type='AVG', conv_cfg=None, norm_cfg=None, with_cp=False, stride=1):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reduction_conv = ConvModule(sum(in_channels), out_channels, kernel_size=1, conv_cfg=self.conv_cfg, act_cfg=None)
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, conv_cfg=self.conv_cfg, act_cfg=None))
        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2 ** i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2 ** i, stride=2 ** i))
        outputs = []
        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)


class BaseMergeCell(nn.Module):
    """The basic class for cells used in NAS-FPN and NAS-FCOS.

    BaseMergeCell takes 2 inputs. After applying concolution
    on them, they are resized to the target size. Then,
    they go through binary_op, which depends on the type of cell.
    If with_out_conv is True, the result of output will go through
    another convolution layer.

    Args:
        in_channels (int): number of input channels in out_conv layer.
        out_channels (int): number of output channels in out_conv layer.
        with_out_conv (bool): Whether to use out_conv layer
        out_conv_cfg (dict): Config dict for convolution layer, which should
            contain "groups", "kernel_size", "padding", "bias" to build
            out_conv layer.
        out_norm_cfg (dict): Config dict for normalization layer in out_conv.
        out_conv_order (tuple): The order of conv/norm/activation layers in
            out_conv.
        with_input1_conv (bool): Whether to use convolution on input1.
        with_input2_conv (bool): Whether to use convolution on input2.
        input_conv_cfg (dict): Config dict for building input1_conv layer and
            input2_conv layer, which is expected to contain the type of
            convolution.
            Default: None, which means using conv2d.
        input_norm_cfg (dict): Config dict for normalization layer in
            input1_conv and input2_conv layer. Default: None.
        upsample_mode (str): Interpolation method used to resize the output
            of input1_conv and input2_conv to target size. Currently, we
            support ['nearest', 'bilinear']. Default: 'nearest'.

    """

    def __init__(self, fused_channels=256, out_channels=256, with_out_conv=True, out_conv_cfg=dict(groups=1, kernel_size=3, padding=1, bias=True), out_norm_cfg=None, out_conv_order=('act', 'conv', 'norm'), with_input1_conv=False, with_input2_conv=False, input_conv_cfg=None, input_norm_cfg=None, upsample_mode='nearest'):
        super(BaseMergeCell, self).__init__()
        assert upsample_mode in ['nearest', 'bilinear']
        self.with_out_conv = with_out_conv
        self.with_input1_conv = with_input1_conv
        self.with_input2_conv = with_input2_conv
        self.upsample_mode = upsample_mode
        if self.with_out_conv:
            self.out_conv = ConvModule(fused_channels, out_channels, **out_conv_cfg, norm_cfg=out_norm_cfg, order=out_conv_order)
        self.input1_conv = self._build_input_conv(out_channels, input_conv_cfg, input_norm_cfg) if with_input1_conv else nn.Sequential()
        self.input2_conv = self._build_input_conv(out_channels, input_conv_cfg, input_norm_cfg) if with_input2_conv else nn.Sequential()

    def _build_input_conv(self, channel, conv_cfg, norm_cfg):
        return ConvModule(channel, channel, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=True)

    @abstractmethod
    def _binary_op(self, x1, x2):
        pass

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            assert x.shape[-2] % size[-2] == 0 and x.shape[-1] % size[-1] == 0
            kernel_size = x.shape[-1] // size[-1]
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
            return x

    def forward(self, x1, x2, out_size=None):
        assert x1.shape[:2] == x2.shape[:2]
        assert out_size is None or len(out_size) == 2
        if out_size is None:
            out_size = max(x1.size()[2:], x2.size()[2:])
        x1 = self.input1_conv(x1)
        x2 = self.input2_conv(x2)
        x1 = self._resize(x1, out_size)
        x2 = self._resize(x2, out_size)
        x = self._binary_op(x1, x2)
        if self.with_out_conv:
            x = self.out_conv(x)
        return x


class GlobalPoolingCell(BaseMergeCell):

    def __init__(self, in_channels=None, out_channels=None, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _binary_op(self, x1, x2):
        x2_att = self.global_pool(x2).sigmoid()
        return x2 + x2_att * x1


class SumCell(BaseMergeCell):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(SumCell, self).__init__(in_channels, out_channels, **kwargs)

    def _binary_op(self, x1, x2):
        return x1 + x2


class NASFPN(nn.Module):
    """NAS-FPN.

    NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object
    Detection. (https://arxiv.org/abs/1904.07392)
    """

    def __init__(self, in_channels, out_channels, num_outs, stack_times, start_level=0, end_level=-1, add_extra_convs=False, norm_cfg=None):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.stack_times = stack_times
        self.norm_cfg = norm_cfg
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
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            self.lateral_convs.append(l_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_conv = ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            self.extra_downsamples.append(nn.Sequential(extra_conv, nn.MaxPool2d(2, 2)))
        self.fpn_stages = nn.ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            stage['gp_64_4'] = GlobalPoolingCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            stage['sum_44_4'] = SumCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            stage['sum_43_3'] = SumCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            stage['sum_34_4'] = SumCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            stage['gp_43_5'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_55_5'] = SumCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            stage['gp_54_7'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_77_7'] = SumCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            stage['gp_75_6'] = GlobalPoolingCell(in_channels=out_channels, out_channels=out_channels, out_norm_cfg=norm_cfg)
            self.fpn_stages.append(stage)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        feats = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))
        p3, p4, p5, p6, p7 = feats
        for stage in self.fpn_stages:
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])
        return p3, p4, p5, p6, p7


class ConcatCell(BaseMergeCell):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConcatCell, self).__init__(in_channels * 2, out_channels, **kwargs)

    def _binary_op(self, x1, x2):
        ret = torch.cat([x1, x2], dim=1)
        return ret


class NASFCOS_FPN(nn.Module):
    """FPN structure in NASFPN

    NAS-FCOS: Fast Neural Architecture Search for Object Detection
    <https://arxiv.org/abs/1906.04423>
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=1, end_level=-1, add_extra_convs=False, conv_cfg=None, norm_cfg=None):
        super(NASFCOS_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
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
        self.adapt_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            adapt_conv = ConvModule(in_channels[i], out_channels, 1, stride=1, padding=0, bias=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU', inplace=False))
            self.adapt_convs.append(adapt_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level

        def build_concat_cell(with_input1_conv, with_input2_conv):
            cell_conv_cfg = dict(kernel_size=1, padding=0, bias=False, groups=out_channels)
            return ConcatCell(in_channels=out_channels, out_channels=out_channels, with_out_conv=True, out_conv_cfg=cell_conv_cfg, out_norm_cfg=dict(type='BN'), out_conv_order=('norm', 'act', 'conv'), with_input1_conv=with_input1_conv, with_input2_conv=with_input2_conv, input_conv_cfg=conv_cfg, input_norm_cfg=norm_cfg, upsample_mode='nearest')
        self.fpn = nn.ModuleDict()
        self.fpn['c22_1'] = build_concat_cell(True, True)
        self.fpn['c22_2'] = build_concat_cell(True, True)
        self.fpn['c32'] = build_concat_cell(True, False)
        self.fpn['c02'] = build_concat_cell(True, False)
        self.fpn['c42'] = build_concat_cell(True, True)
        self.fpn['c36'] = build_concat_cell(True, True)
        self.fpn['c61'] = build_concat_cell(True, True)
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_act_cfg = None if i == 0 else dict(type='ReLU', inplace=False)
            self.extra_downsamples.append(ConvModule(out_channels, out_channels, 3, stride=2, padding=1, act_cfg=extra_act_cfg, order=('act', 'norm', 'conv')))

    def forward(self, inputs):
        feats = [adapt_conv(inputs[i + self.start_level]) for i, adapt_conv in enumerate(self.adapt_convs)]
        for i, module_name in enumerate(self.fpn):
            idx_1, idx_2 = int(module_name[1]), int(module_name[2])
            res = self.fpn[module_name](feats[idx_1], feats[idx_2])
            feats.append(res)
        ret = []
        for idx, input_idx in zip([9, 8, 7], [1, 2, 3]):
            feats1, feats2 = feats[idx], feats[5]
            feats2_resize = F.interpolate(feats2, size=feats1.size()[2:], mode='bilinear', align_corners=False)
            feats_sum = feats1 + feats2_resize
            ret.append(F.interpolate(feats_sum, size=inputs[input_idx].size()[2:], mode='bilinear', align_corners=False))
        for submodule in self.extra_downsamples:
            ret.append(submodule(ret[-1]))
        return tuple(ret)

    def init_weights(self):
        for module in self.fpn.values():
            if hasattr(module, 'conv_out'):
                caffe2_xavier_init(module.out_conv.conv)
        for modules in [self.adapt_convs.modules(), self.extra_downsamples.modules()]:
            for module in modules:
                if isinstance(module, nn.Conv2d):
                    caffe2_xavier_init(module)


class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the PAFPN in Path Aggregation Network
    (https://arxiv.org/abs/1803.01534).

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False, no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(PAFPN, self).__init__(in_channels, out_channels, num_outs, start_level, end_level, add_extra_convs, extra_convs_on_inputs, relu_before_extra_convs, no_norm_on_lateral, conv_cfg, norm_cfg, act_cfg)
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            pafpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')
        inter_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])
        outs = []
        outs.append(inter_outs[0])
        outs.extend([self.pafpn_convs[i - 1](inter_outs[i]) for i in range(1, used_backbone_levels)])
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


class BaseRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for RoIHeads"""

    def __init__(self, bbox_roi_extractor=None, bbox_head=None, mask_roi_extractor=None, mask_head=None, shared_head=None, train_cfg=None, test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)
        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        pass

    @abstractmethod
    def init_bbox_head(self):
        pass

    @abstractmethod
    def init_mask_head(self):
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        pass

    @abstractmethod
    def forward_train(self, x, img_meta, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, **kwargs):
        """Forward function during training"""
        pass

    async def async_simple_test(self, x, img_meta, **kwargs):
        raise NotImplementedError

    def simple_test(self, x, proposal_list, img_meta, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self, with_avg_pool=False, with_cls=True, with_reg=True, roi_feat_size=7, in_channels=256, num_classes=80, bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]), reg_class_agnostic=False, reg_decoded_bbox=False, loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes + 1)
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

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(self._get_target_single, pos_bboxes_list, neg_bboxes_list, pos_gt_bboxes_list, pos_gt_labels_list, cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor, reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred, bbox_targets[pos_inds.type(torch.bool)], bbox_weights[pos_inds.type(torch.bool)], avg_factor=bbox_targets.size(0), reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)
        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])
        return bboxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)
        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4
        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
        return new_rois


class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \\-> reg convs -> reg fcs -> reg
    """

    def __init__(self, num_shared_convs=0, num_shared_fcs=0, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0, num_reg_fcs=0, conv_out_channels=256, fc_out_channels=1024, conv_cfg=None, norm_cfg=None, *args, **kwargs):
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
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area
        self.relu = nn.ReLU(inplace=True)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
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
                branch_convs.append(ConvModule(conv_in_channels, self.conv_out_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
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
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        x_cls = x
        x_reg = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(*args, num_shared_convs=0, num_shared_fcs=2, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0, num_reg_fcs=0, fc_out_channels=fc_out_channels, **kwargs)


class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(*args, num_shared_convs=4, num_shared_fcs=1, num_cls_convs=0, num_cls_fcs=0, num_reg_convs=0, num_reg_fcs=0, fc_out_channels=fc_out_channels, **kwargs)


class BasicResBlock(nn.Module):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()
        self.conv1 = ConvModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(in_channels, out_channels, kernel_size=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.conv_identity = ConvModule(in_channels, out_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        identity = self.conv_identity(identity)
        out = x + identity
        out = self.relu(out)
        return out


class DoubleConvFCBBoxHead(BBoxHead):
    """Bbox head used in Double-Head R-CNN

    .. code-block:: none

                                          /-> cls
                      /-> shared convs ->
                                          \\-> reg
        roi features
                                          /-> cls
                      \\-> shared fc    ->
                                          \\-> reg
    """

    def __init__(self, num_convs=0, num_fcs=0, conv_out_channels=1024, fc_out_channels=1024, conv_cfg=None, norm_cfg=dict(type='BN'), **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.res_block = BasicResBlock(self.in_channels, self.conv_out_channels)
        self.conv_branch = self._add_conv_branch()
        self.fc_branch = self._add_fc_branch()
        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(Bottleneck(inplanes=self.conv_out_channels, planes=self.conv_out_channels // 4, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = self.in_channels * self.roi_feat_area if i == 0 else self.fc_out_channels
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)
        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        x_conv = self.res_block(x_reg)
        for conv in self.conv_branch:
            x_conv = conv(x_conv)
        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))
        cls_score = self.fc_cls(x_fc)
        return cls_score, bbox_pred


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


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
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


def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = [(mask if not img_info[0]['flip'] else mask[..., ::-1]) for mask, img_info in zip(aug_masks, img_metas)]
    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


BYTES_PER_FLOAT = 4


class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None


class Conv2d(nn.Conv2d):

    def forward(self, x):
        if x.numel() == 0 and torch.__version__ <= '1.4':
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-2:], self.kernel_size, self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty
        return super().forward(x)


GPU_MEM_LIMIT = 1024 ** 3


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)
    N = masks.shape[0]
    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)
    img_masks = F.grid_sample(masks, grid, align_corners=False)
    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        mask_targets = gt_masks.crop_and_resize(proposals_np, mask_size, device=device, inds=pos_assigned_gt_inds).to_ndarray()
        mask_targets = torch.from_numpy(mask_targets).float()
    else:
        mask_targets = pos_proposals.new_zeros((0,) + mask_size)
    return mask_targets


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


class FusedSemanticHead(nn.Module):
    """Multi-level fused semantic segmentation head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |                  \\-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """

    def __init__(self, num_ins, fusion_level, num_convs=4, in_channels=256, conv_out_channels=256, num_classes=183, ignore_label=255, loss_weight=0.2, conv_cfg=None, norm_cfg=None):
        super(FusedSemanticHead, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(ConvModule(self.in_channels, self.in_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, inplace=False))
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(ConvModule(in_channels, conv_out_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.conv_embedding = ConvModule(conv_out_channels, conv_out_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def init_weights(self):
        kaiming_init(self.conv_logits)

    @auto_fp16()
    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)
        for i in range(self.num_convs):
            x = self.convs[i](x)
        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        return mask_pred, x

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg


class GridHead(nn.Module):

    def __init__(self, grid_points=9, num_convs=8, roi_feat_size=14, in_channels=256, conv_kernel_size=3, point_feat_channels=64, deconv_kernel_size=4, class_agnostic=False, loss_grid=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=15), conv_cfg=None, norm_cfg=dict(type='GN', num_groups=36)):
        super(GridHead, self).__init__()
        self.grid_points = grid_points
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.point_feat_channels = point_feat_channels
        self.conv_out_channels = self.point_feat_channels * self.grid_points
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0
        assert self.grid_points >= 4
        self.grid_size = int(np.sqrt(self.grid_points))
        if self.grid_size * self.grid_size != self.grid_points:
            raise ValueError('grid_points must be a square number')
        if not isinstance(self.roi_feat_size, int):
            raise ValueError('Only square RoIs are supporeted in Grid R-CNN')
        self.whole_map_size = self.roi_feat_size * 4
        self.sub_regions = self.calc_sub_regions()
        self.convs = []
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.conv_out_channels
            stride = 2 if i == 0 else 1
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(ConvModule(in_channels, self.conv_out_channels, self.conv_kernel_size, stride=stride, padding=padding, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=True))
        self.convs = nn.Sequential(*self.convs)
        self.deconv1 = nn.ConvTranspose2d(self.conv_out_channels, self.conv_out_channels, kernel_size=deconv_kernel_size, stride=2, padding=(deconv_kernel_size - 2) // 2, groups=grid_points)
        self.norm1 = nn.GroupNorm(grid_points, self.conv_out_channels)
        self.deconv2 = nn.ConvTranspose2d(self.conv_out_channels, grid_points, kernel_size=deconv_kernel_size, stride=2, padding=(deconv_kernel_size - 2) // 2, groups=grid_points)
        self.neighbor_points = []
        grid_size = self.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                neighbors = []
                if i > 0:
                    neighbors.append((i - 1) * grid_size + j)
                if j > 0:
                    neighbors.append(i * grid_size + j - 1)
                if j < grid_size - 1:
                    neighbors.append(i * grid_size + j + 1)
                if i < grid_size - 1:
                    neighbors.append((i + 1) * grid_size + j)
                self.neighbor_points.append(tuple(neighbors))
        self.num_edges = sum([len(p) for p in self.neighbor_points])
        self.forder_trans = nn.ModuleList()
        self.sorder_trans = nn.ModuleList()
        for neighbors in self.neighbor_points:
            fo_trans = nn.ModuleList()
            so_trans = nn.ModuleList()
            for _ in range(len(neighbors)):
                fo_trans.append(nn.Sequential(nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 5, stride=1, padding=2, groups=self.point_feat_channels), nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 1)))
                so_trans.append(nn.Sequential(nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 5, 1, 2, groups=self.point_feat_channels), nn.Conv2d(self.point_feat_channels, self.point_feat_channels, 1)))
            self.forder_trans.append(fo_trans)
            self.sorder_trans.append(so_trans)
        self.loss_grid = build_loss(loss_grid)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_init(m)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
        nn.init.constant_(self.deconv2.bias, -np.log(0.99 / 0.01))

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        x = self.convs(x)
        c = self.point_feat_channels
        x_fo = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_fo[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_fo[i] = x_fo[i] + self.forder_trans[i][j](x[:, point_idx * c:(point_idx + 1) * c])
        x_so = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_so[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])
        x2 = torch.cat(x_so, dim=1)
        x2 = self.deconv1(x2)
        x2 = F.relu(self.norm1(x2), inplace=True)
        heatmap = self.deconv2(x2)
        if self.training:
            x1 = x
            x1 = self.deconv1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            heatmap_unfused = self.deconv2(x1)
        else:
            heatmap_unfused = heatmap
        return dict(fused=heatmap, unfused=heatmap_unfused)

    def calc_sub_regions(self):
        """Compute point specific representation regions.

        See Grid R-CNN Plus (https://arxiv.org/abs/1906.05688) for details.
        """
        half_size = self.whole_map_size // 4 * 2
        sub_regions = []
        for i in range(self.grid_points):
            x_idx = i // self.grid_size
            y_idx = i % self.grid_size
            if x_idx == 0:
                sub_x1 = 0
            elif x_idx == self.grid_size - 1:
                sub_x1 = half_size
            else:
                ratio = x_idx / (self.grid_size - 1) - 0.25
                sub_x1 = max(int(ratio * self.whole_map_size), 0)
            if y_idx == 0:
                sub_y1 = 0
            elif y_idx == self.grid_size - 1:
                sub_y1 = half_size
            else:
                ratio = y_idx / (self.grid_size - 1) - 0.25
                sub_y1 = max(int(ratio * self.whole_map_size), 0)
            sub_regions.append((sub_x1, sub_y1, sub_x1 + half_size, sub_y1 + half_size))
        return sub_regions

    def get_targets(self, sampling_results, rcnn_train_cfg):
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results], dim=0).cpu()
        pos_gt_bboxes = torch.cat([res.pos_gt_bboxes for res in sampling_results], dim=0).cpu()
        assert pos_bboxes.shape == pos_gt_bboxes.shape
        x1 = pos_bboxes[:, 0] - (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y1 = pos_bboxes[:, 1] - (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        x2 = pos_bboxes[:, 2] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y2 = pos_bboxes[:, 3] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        pos_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        pos_bbox_ws = (pos_bboxes[:, 2] - pos_bboxes[:, 0]).unsqueeze(-1)
        pos_bbox_hs = (pos_bboxes[:, 3] - pos_bboxes[:, 1]).unsqueeze(-1)
        num_rois = pos_bboxes.shape[0]
        map_size = self.whole_map_size
        targets = torch.zeros((num_rois, self.grid_points, map_size, map_size), dtype=torch.float)
        factors = []
        for j in range(self.grid_points):
            x_idx = j // self.grid_size
            y_idx = j % self.grid_size
            factors.append((1 - x_idx / (self.grid_size - 1), 1 - y_idx / (self.grid_size - 1)))
        radius = rcnn_train_cfg.pos_radius
        radius2 = radius ** 2
        for i in range(num_rois):
            if pos_bbox_ws[i] <= self.grid_size or pos_bbox_hs[i] <= self.grid_size:
                continue
            for j in range(self.grid_points):
                factor_x, factor_y = factors[j]
                gridpoint_x = factor_x * pos_gt_bboxes[i, 0] + (1 - factor_x) * pos_gt_bboxes[i, 2]
                gridpoint_y = factor_y * pos_gt_bboxes[i, 1] + (1 - factor_y) * pos_gt_bboxes[i, 3]
                cx = int((gridpoint_x - pos_bboxes[i, 0]) / pos_bbox_ws[i] * map_size)
                cy = int((gridpoint_y - pos_bboxes[i, 1]) / pos_bbox_hs[i] * map_size)
                for x in range(cx - radius, cx + radius + 1):
                    for y in range(cy - radius, cy + radius + 1):
                        if x >= 0 and x < map_size and y >= 0 and y < map_size:
                            if (x - cx) ** 2 + (y - cy) ** 2 <= radius2:
                                targets[i, j, y, x] = 1
        sub_targets = []
        for i in range(self.grid_points):
            sub_x1, sub_y1, sub_x2, sub_y2 = self.sub_regions[i]
            sub_targets.append(targets[:, [i], sub_y1:sub_y2, sub_x1:sub_x2])
        sub_targets = torch.cat(sub_targets, dim=1)
        sub_targets = sub_targets
        return sub_targets

    def loss(self, grid_pred, grid_targets):
        loss_fused = self.loss_grid(grid_pred['fused'], grid_targets)
        loss_unfused = self.loss_grid(grid_pred['unfused'], grid_targets)
        loss_grid = loss_fused + loss_unfused
        return dict(loss_grid=loss_grid)

    def get_bboxes(self, det_bboxes, grid_pred, img_metas):
        assert det_bboxes.shape[0] == grid_pred.shape[0]
        det_bboxes = det_bboxes.cpu()
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]
        grid_pred = grid_pred.sigmoid().cpu()
        R, c, h, w = grid_pred.shape
        half_size = self.whole_map_size // 4 * 2
        assert h == w == half_size
        assert c == self.grid_points
        grid_pred = grid_pred.view(R * c, h * w)
        pred_scores, pred_position = grid_pred.max(dim=1)
        xs = pred_position % w
        ys = pred_position // w
        for i in range(self.grid_points):
            xs[i::self.grid_points] += self.sub_regions[i][0]
            ys[i::self.grid_points] += self.sub_regions[i][1]
        pred_scores, xs, ys = tuple(map(lambda x: x.view(R, c), [pred_scores, xs, ys]))
        widths = (det_bboxes[:, 2] - det_bboxes[:, 0]).unsqueeze(-1)
        heights = (det_bboxes[:, 3] - det_bboxes[:, 1]).unsqueeze(-1)
        x1 = det_bboxes[:, 0, None] - widths / 2
        y1 = det_bboxes[:, 1, None] - heights / 2
        abs_xs = (xs.float() + 0.5) / w * widths + x1
        abs_ys = (ys.float() + 0.5) / h * heights + y1
        x1_inds = [i for i in range(self.grid_size)]
        y1_inds = [(i * self.grid_size) for i in range(self.grid_size)]
        x2_inds = [(self.grid_points - self.grid_size + i) for i in range(self.grid_size)]
        y2_inds = [((i + 1) * self.grid_size - 1) for i in range(self.grid_size)]
        bboxes_x1 = (abs_xs[:, x1_inds] * pred_scores[:, x1_inds]).sum(dim=1, keepdim=True) / pred_scores[:, x1_inds].sum(dim=1, keepdim=True)
        bboxes_y1 = (abs_ys[:, y1_inds] * pred_scores[:, y1_inds]).sum(dim=1, keepdim=True) / pred_scores[:, y1_inds].sum(dim=1, keepdim=True)
        bboxes_x2 = (abs_xs[:, x2_inds] * pred_scores[:, x2_inds]).sum(dim=1, keepdim=True) / pred_scores[:, x2_inds].sum(dim=1, keepdim=True)
        bboxes_y2 = (abs_ys[:, y2_inds] * pred_scores[:, y2_inds]).sum(dim=1, keepdim=True) / pred_scores[:, y2_inds].sum(dim=1, keepdim=True)
        bbox_res = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2, cls_scores], dim=1)
        bbox_res[:, [0, 2]].clamp_(min=0, max=img_metas[0]['img_shape'][1])
        bbox_res[:, [1, 3]].clamp_(min=0, max=img_metas[0]['img_shape'][0])
        return bbox_res


class Linear(torch.nn.Linear):

    def forward(self, x):
        if x.numel() == 0:
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty
        return super().forward(x)


class MaxPool2d(nn.MaxPool2d):

    def forward(self, x):
        if x.numel() == 0 and torch.__version__ <= '1.4':
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-2:], _pair(self.kernel_size), _pair(self.padding), _pair(self.stride), _pair(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty
        return super().forward(x)


class MaskIoUHead(nn.Module):
    """Mask IoU Head.

    This head predicts the IoU of predicted masks and corresponding gt masks.
    """

    def __init__(self, num_convs=4, num_fcs=2, roi_feat_size=14, in_channels=256, conv_out_channels=256, fc_out_channels=1024, num_classes=80, loss_iou=dict(type='MSELoss', loss_weight=0.5)):
        super(MaskIoUHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.fp16_enabled = False
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                in_channels = self.in_channels + 1
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(Conv2d(in_channels, self.conv_out_channels, 3, stride=stride, padding=1))
        roi_feat_size = _pair(roi_feat_size)
        pooled_area = roi_feat_size[0] // 2 * (roi_feat_size[1] // 2)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = self.conv_out_channels * pooled_area if i == 0 else self.fc_out_channels
            self.fcs.append(Linear(in_channels, self.fc_out_channels))
        self.fc_mask_iou = Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = MaxPool2d(2, 2)
        self.loss_iou = build_loss(loss_iou)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(fc, a=1, mode='fan_in', nonlinearity='leaky_relu', distribution='uniform')
        normal_init(self.fc_mask_iou, std=0.01)

    def forward(self, mask_feat, mask_pred):
        mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))
        x = torch.cat((mask_feat, mask_pred_pooled), 1)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    @force_fp32(apply_to=('mask_iou_pred',))
    def loss(self, mask_iou_pred, mask_iou_targets):
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(mask_iou_pred[pos_inds], mask_iou_targets[pos_inds])
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)

    @force_fp32(apply_to=('mask_pred',))
    def get_targets(self, sampling_results, gt_masks, mask_pred, mask_targets, rcnn_train_cfg):
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (BitmapMask | PolygonMask): Gt masks (the whole instance)
                of each image, with the same shape of the input image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        area_ratios = map(self._get_area_ratio, pos_proposals, pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)
        mask_pred = (mask_pred > rcnn_train_cfg.mask_thr_binary).float()
        mask_pred_areas = mask_pred.sum((-1, -2))
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-07)
        mask_iou_targets = overlap_areas / (mask_pred_areas + gt_full_areas - overlap_areas)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance"""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            gt_instance_mask_area = gt_masks.areas
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]
                bbox = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask.crop(bbox)
                ratio = gt_mask_in_proposal.areas[0] / (gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-07)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float()
        else:
            area_ratios = pos_proposals.new_zeros((0,))
        return area_ratios

    @force_fp32(apply_to=('mask_iou_pred',))
    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = bbox_score * mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = mask_iou_pred[inds, det_labels] * det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [mask_scores[det_labels == i] for i in range(self.num_classes)]


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
        self.fp16_enabled = False

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

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-06))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)
        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return roi_feats


class CARAFENaiveFunction(Function):

    @staticmethod
    def forward(ctx, features, masks, kernel_size, group_size, scale_factor):
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()
        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        if features.is_cuda:
            carafe_naive_ext.forward(features, masks, kernel_size, group_size, scale_factor, output)
        else:
            raise NotImplementedError
        if features.requires_grad or masks.requires_grad:
            ctx.save_for_backward(features, masks)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, masks = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor
        grad_input = torch.zeros_like(features)
        grad_masks = torch.zeros_like(masks)
        carafe_naive_ext.backward(grad_output.contiguous(), features, masks, kernel_size, group_size, scale_factor, grad_input, grad_masks)
        return grad_input, grad_masks, None, None, None


class CARAFENaive(Module):

    def __init__(self, kernel_size, group_size, scale_factor):
        super(CARAFENaive, self).__init__()
        assert isinstance(kernel_size, int) and isinstance(group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features, masks):
        return CARAFENaiveFunction.apply(features, masks, self.kernel_size, self.group_size, self.scale_factor)


class CARAFE(Module):
    """ CARAFE: Content-Aware ReAssembly of FEatures

    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        kernel_size (int): reassemble kernel size
        group_size (int): reassemble group size
        scale_factor (int): upsample ratio

    Returns:
        upsampled feature map
    """

    def __init__(self, kernel_size, group_size, scale_factor):
        super(CARAFE, self).__init__()
        assert isinstance(kernel_size, int) and isinstance(group_size, int) and isinstance(scale_factor, int)
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.scale_factor = scale_factor

    def forward(self, features, masks):
        return CARAFEFunction.apply(features, masks, self.kernel_size, self.group_size, self.scale_factor)


def conv_ws_2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-05):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-05):
        super(ConvWS2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.eps)


class DeformConvPack(DeformConv):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'conv_offset.weight' not in state_dict and prefix[:-1] + '_offset.weight' in state_dict:
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] + '_offset.weight')
            if prefix + 'conv_offset.bias' not in state_dict and prefix[:-1] + '_offset.bias' in state_dict:
                state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] + '_offset.bias')
        if version is not None and version > 1:
            print_log(f"DeformConvPack {prefix.rstrip('.')} is upgraded to version 2.", logger='root')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


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
        deform_conv_ext.modulated_deform_conv_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
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
        deform_conv_ext.modulated_deform_conv_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
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
        self.transposed = False
        self.output_padding = _single(0)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
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
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """
    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(self.in_channels, self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation), bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'conv_offset.weight' not in state_dict and prefix[:-1] + '_offset.weight' in state_dict:
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(prefix[:-1] + '_offset.weight')
            if prefix + 'conv_offset.bias' not in state_dict and prefix[:-1] + '_offset.bias' in state_dict:
                state_dict[prefix + 'conv_offset.bias'] = state_dict.pop(prefix[:-1] + '_offset.bias')
        if version is not None and version > 1:
            print_log(f"ModulatedDeformConvPack {prefix.rstrip('.')} is upgraded to version 2.", logger='root')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        assert out_h == out_w
        out_size = out_h
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
        deform_pool_ext.deform_psroi_pooling_forward(data, rois, offset, output, output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
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
        deform_pool_ext.deform_psroi_pooling_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = _pair(out_size)
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
            ic = self.out_size[0] * self.out_size[1] * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1] * 2
                seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        n = rois.shape[0]
        if n == 0:
            return data.new_empty(n, self.out_channels, self.out_size[0], self.out_size[1])
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size[0], self.out_size[1])
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0, num_offset_fcs=3, num_mask_fcs=2, deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(spatial_scale, out_size, out_channels, no_trans, group_size, part_size, sample_per_part, trans_std)
        self.num_offset_fcs = num_offset_fcs
        self.num_mask_fcs = num_mask_fcs
        self.deform_fc_channels = deform_fc_channels
        if not no_trans:
            offset_fc_seq = []
            ic = self.out_size[0] * self.out_size[1] * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1] * 2
                offset_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    offset_fc_seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*offset_fc_seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            mask_fc_seq = []
            ic = self.out_size[0] * self.out_size[1] * self.out_channels
            for i in range(self.num_mask_fcs):
                if i < self.num_mask_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1]
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
        n = rois.shape[0]
        if n == 0:
            return data.new_empty(n, self.out_channels, self.out_size[0], self.out_size[1])
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
        else:
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, True, self.group_size, self.part_size, self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size[0], self.out_size[1])
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size[0], self.out_size[1])
            return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std) * mask


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0, aligned=True):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        ctx.aligned = aligned
        if aligned:
            output = roi_align_ext.forward_v2(features, rois, spatial_scale, out_h, out_w, sample_num, aligned)
        elif features.is_cuda:
            batch_size, num_channels, data_height, data_width = features.size()
            num_rois = rois.size(0)
            output = features.new_zeros(num_rois, num_channels, out_h, out_w)
            roi_align_ext.forward_v1(features, rois, out_h, out_w, spatial_scale, sample_num, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        aligned = ctx.aligned
        assert feature_size is not None
        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)
        grad_input = grad_rois = None
        if not aligned:
            if ctx.needs_input_grad[0]:
                grad_input = rois.new_zeros(batch_size, num_channels, data_height, data_width)
                roi_align_ext.backward_v1(grad_output.contiguous(), rois, out_h, out_w, spatial_scale, sample_num, grad_input)
        else:
            grad_input = roi_align_ext.backward_v2(grad_output, rois, spatial_scale, out_h, out_w, batch_size, num_channels, data_height, data_width, sample_num, aligned)
        return grad_input, grad_rois, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):

    def __init__(self, out_size, spatial_scale, sample_num=0, use_torchvision=False, aligned=True):
        """
        Args:
            out_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sample_num (int): number of inputs samples to take for each
                output sample. 2 to take samples densely for current models.
            use_torchvision (bool): whether to use roi_align from torchvision
            aligned (bool): if False, use the legacy implementation in
                MMDetection. If True, align the results more perfectly.

        Note:
            The implementation of RoIAlign when aligned=True is modified from
            https://github.com/facebookresearch/detectron2/

            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel
            indices (in our pixel model) are computed by floor(c - 0.5) and
            ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal
            at continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing
            neighboring pixel indices and therefore it uses pixels with a
            slightly incorrect alignment (relative to our pixel model) when
            performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors;

            The difference does not make a difference to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(RoIAlign, self).__init__()
        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.aligned = aligned
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision
        assert not (use_torchvision and aligned), 'Torchvision does not support aligned RoIAlgin'

    def forward(self, features, rois):
        """
        Args:
            features: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4
            columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            return tv_roi_align(features, rois, self.out_size, self.spatial_scale, self.sample_num)
        else:
            return roi_align(features, rois, self.out_size, self.spatial_scale, self.sample_num, self.aligned)

    def __repr__(self):
        indent_str = '\n    '
        format_str = self.__class__.__name__
        format_str += f'({indent_str}out_size={self.out_size},'
        format_str += f'{indent_str}spatial_scale={self.spatial_scale},'
        format_str += f'{indent_str}sample_num={self.sample_num},'
        format_str += f'{indent_str}use_torchvision={self.use_torchvision},'
        format_str += f'{indent_str}aligned={self.aligned})'
        return format_str


class RoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.save_for_backward(rois)
        num_channels = features.size(1)
        num_rois = rois.size(0)
        out_size = num_rois, num_channels, out_h, out_w
        output = features.new_zeros(out_size)
        argmax = features.new_zeros(out_size, dtype=torch.int)
        roi_pool_ext.forward(features, rois, out_h, out_w, spatial_scale, output, argmax)
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = features.size()
        ctx.argmax = argmax
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        spatial_scale = ctx.spatial_scale
        feature_size = ctx.feature_size
        argmax = ctx.argmax
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            roi_pool_ext.backward(grad_output.contiguous(), rois, argmax, spatial_scale, grad_input)
        return grad_input, grad_rois, None, None


roi_pool = RoIPoolFunction.apply


class RoIPool(nn.Module):

    def __init__(self, out_size, spatial_scale, use_torchvision=False):
        super(RoIPool, self).__init__()
        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_pool as tv_roi_pool
            return tv_roi_pool(features, rois, self.out_size, self.spatial_scale)
        else:
            return roi_pool(features, rois, self.out_size, self.spatial_scale)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(out_size={self.out_size}, '
        format_str += f'spatial_scale={self.spatial_scale}, '
        format_str += f'use_torchvision={self.use_torchvision})'
        return format_str


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__
        tmpstr += f'(gamma={self.gamma}, alpha={self.alpha})'
        return tmpstr


class ConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() == 0 and torch.__version__ <= '1.4.0':
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size, self.padding, self.stride, self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty
        return super(ConvTranspose2d, self).forward(x)


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()

    def forward(self, x):
        return x


class ExampleDuplicateModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1))
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.conv3 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv3[0] = self.conv1[0]

    def forward(self, x):
        return x


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ExampleModel()

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BalancedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BoundedIoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvWS2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ExampleDuplicateModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExampleModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GHMC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GHMR,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GIoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (IoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (L1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (L2Norm,
     lambda: ([], {'n_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaskedConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PseudoDataParallel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SubModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mdv3101_CDeCNet(_paritybench_base):
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

