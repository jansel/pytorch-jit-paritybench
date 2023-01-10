import sys
_module = sys.modules[__name__]
del sys
mmdet3d = _module
apis = _module
test = _module
train = _module
core = _module
anchor = _module
anchor_3d_generator = _module
bbox = _module
assigners = _module
hungarian_assigner = _module
hungarian_assigner_3d = _module
box_np_ops = _module
coders = _module
anchor_free_bbox_coder = _module
centerpoint_bbox_coders = _module
delta_xyzwhlr_bbox_coder = _module
groupfree3d_bbox_coder = _module
nms_free_coder = _module
partial_bin_based_bbox_coder = _module
transfusion_bbox_coder = _module
iou_calculators = _module
iou3d_calculator = _module
match_costs = _module
match_cost = _module
samplers = _module
iou_neg_piecewise_sampler = _module
structures = _module
base_box3d = _module
box_3d_mode = _module
cam_box3d = _module
coord_3d_mode = _module
depth_box3d = _module
lidar_box3d = _module
utils = _module
util = _module
points = _module
base_points = _module
cam_points = _module
depth_points = _module
lidar_points = _module
post_processing = _module
box3d_nms = _module
gaussian = _module
visualize = _module
voxel = _module
builder = _module
voxel_generator = _module
datasets = _module
custom_3d = _module
dataset_wrappers = _module
nuscenes_dataset = _module
pipelines = _module
dbsampler = _module
formating = _module
loading = _module
loading_utils = _module
transforms_3d = _module
utils = _module
models = _module
backbones = _module
dla = _module
pillar_encoder = _module
resnet = _module
second = _module
sparse_encoder = _module
vovnet = _module
fusers = _module
add = _module
conv = _module
fusion_models = _module
base = _module
bevfusion = _module
heads = _module
centerpoint = _module
transfusion = _module
segm = _module
vanilla = _module
losses = _module
necks = _module
detectron_fpn = _module
generalized_lss = _module
lss = _module
second = _module
flops_counter = _module
transformer = _module
vtransforms = _module
base = _module
depth_lss = _module
lss = _module
ops = _module
ball_query = _module
ball_query = _module
bev_pool = _module
bev_pool = _module
furthest_point_sample = _module
furthest_point_sample = _module
points_sampler = _module
utils = _module
gather_points = _module
gather_points = _module
group_points = _module
group_points = _module
interpolate = _module
three_interpolate = _module
three_nn = _module
iou3d = _module
iou3d_utils = _module
knn = _module
knn = _module
norm = _module
paconv = _module
assign_score = _module
paconv = _module
utils = _module
pointnet_modules = _module
builder = _module
paconv_sa_module = _module
point_fp_module = _module
point_sa_module = _module
roiaware_pool3d = _module
points_in_boxes = _module
roiaware_pool3d = _module
sparse_block = _module
spconv = _module
conv = _module
functional = _module
modules = _module
ops = _module
pool = _module
structure = _module
test_utils = _module
scatter_points = _module
voxelize = _module
runner = _module
epoch_based_runner = _module
config = _module
logger = _module
syncbn = _module
setup = _module
benchmark = _module
create_data = _module
data_converter = _module
create_gt_database = _module
nuscenes_converter = _module
export = _module
test = _module
train = _module
visualize = _module

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


import numpy as np


from abc import abstractmethod


from enum import IntEnum


from enum import unique


from logging import warning


import warnings


from torch.utils.data import Dataset


from typing import Any


from typing import Dict


import torchvision


from numpy import random


from collections import OrderedDict


import torch.nn.functional as F


from torch import nn


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn import functional as F


from typing import List


from typing import Tuple


from torch import nn as nn


import torch.nn as nn


import random


from abc import ABCMeta


import torch.distributed as dist


import copy


from typing import Optional


from typing import Union


import math


from torch.nn.parameter import Parameter


from torch.nn import Linear


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.autograd import Function


from torch import distributed as dist


from torch.autograd.function import Function


from torch.nn import init


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.nn.modules.utils import _pair


from collections import deque


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import time


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvModule(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=norm_cfg is None, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=norm_cfg is None, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = out + residual
        out = F.relu_(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=None):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = ConvModule(inplanes, bottle_planes, kernel_size=1, bias=norm_cfg is None, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=norm_cfg is None, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(bottle_planes, planes, kernel_size=1, bias=norm_cfg is None, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)
        out = F.relu_(out)
        out = self.conv3(out)
        out = out + residual
        out = F.relu_(out)
        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=None):
        super(Root, self).__init__()
        self.conv = ConvModule(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=norm_cfg is None, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.residual = residual

    def forward(self, *x):
        children = x
        y = self.conv(torch.cat(x, 1))
        if self.residual:
            y = y + children[0]
        y = F.relu_(y)
        return y


class Tree(nn.Module):

    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN2d'), act_cfg=None):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels and not isinstance(self.tree1, Tree):
            self.project = ConvModule(in_channels, out_channels, kernel_size=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            y = self.root(x2, x1, *children)
        else:
            children.append(x1)
            y = self.tree2(x1, children=children)
        return y


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d', eps=0.001, momentum=0.01)
        self.norm_cfg = norm_cfg
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PillarFeatureNet(nn.Module):

    def __init__(self, in_channels=4, feat_channels=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), point_cloud_range=(0, -40, -3, 70.4, 40, 1), norm_cfg=None):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(feat_channels) > 0
        self.in_channels = in_channels
        in_channels += 5
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 1].unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].unsqueeze(1) * self.vy + self.y_offset)
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze()


class PointPillarsScatter(nn.Module):

    def __init__(self, in_channels=64, output_shape=(512, 512), **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        self.nx = output_shape[0]
        self.ny = output_shape[1]

    def extra_repr(self):
        return f'in_channels={self.in_channels}, output_shape={tuple(self.output_shape)}'

    def forward(self, voxel_features, coords, batch_size):
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.in_channels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny + this_coords[:, 2]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.nx, self.ny)
        return batch_canvas


def build_backbone(cfg):
    return BACKBONES.build(cfg)


class PointPillarsEncoder(nn.Module):

    def __init__(self, pts_voxel_encoder: Dict[str, Any], pts_middle_encoder: Dict[str, Any], **kwargs):
        super().__init__()
        self.pts_voxel_encoder = build_backbone(pts_voxel_encoder)
        self.pts_middle_encoder = build_backbone(pts_middle_encoder)

    def forward(self, feats, coords, batch_size, sizes):
        x = self.pts_voxel_encoder(feats, sizes, coords)
        x = self.pts_middle_encoder(x, coords, batch_size)
        return x


class GeneralizedResNet(nn.ModuleList):

    def __init__(self, in_channels: int, blocks: List[Tuple[int, int, int]]) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.blocks = blocks
        for num_blocks, out_channels, stride in self.blocks:
            blocks = make_res_layer(BasicBlock, in_channels, out_channels, num_blocks, stride=stride, dilation=1)
            in_channels = out_channels
            self.append(blocks)

    def forward(self, x: torch.Tensor) ->List[torch.Tensor]:
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs


def make_sparse_convmodule(in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='SubMConv3d', norm_cfg=None, order=('conv', 'norm', 'act')):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}
    conv_cfg = dict(type=conv_type, indice_key=indice_key)
    layers = list()
    for layer in order:
        if layer == 'conv':
            if conv_type not in ['SparseInverseConv3d', 'SparseInverseConv2d', 'SparseInverseConv1d']:
                layers.append(build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False))
            else:
                layers.append(build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, bias=False))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            layers.append(nn.ReLU(inplace=True))
    layers = spconv.SparseSequential(*layers)
    return layers


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), (f'{module_name}_{postfix}/norm', nn.BatchNorm2d(out_channels)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), (f'{module_name}_{postfix}/norm', nn.BatchNorm2d(out_channels)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [('{}_{}/dw_conv3x3'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=out_channels, bias=False)), ('{}_{}/pw_conv1x1'.format(module_name, postfix), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)), ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)), ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True))]


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False):
        super(_OSA_stage, self).__init__()
        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise))


class AddFuser(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout: float=0) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.transforms = nn.ModuleList()
        for k in range(len(in_channels)):
            self.transforms.append(nn.Sequential(nn.Conv2d(in_channels[k], out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)))

    def forward(self, inputs: List[torch.Tensor]) ->torch.Tensor:
        features = []
        for transform, input in zip(self.transforms, inputs):
            features.append(transform(input))
        weights = [1] * len(inputs)
        if self.training and random.random() < self.dropout:
            index = random.randint(0, len(inputs) - 1)
            weights[index] = 0
        return sum(w * f for w, f in zip(weights, features)) / sum(weights)


class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) ->None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, inputs: List[torch.Tensor]) ->torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


class FFN(nn.Module):

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, init_bias=-2.19, conv_cfg=dict(type='Conv1d'), norm_cfg=dict(type='BN1d'), bias='auto', **kwargs):
        super(FFN, self).__init__()
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]
            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(ConvModule(c_in, head_conv, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=bias, conv_cfg=conv_cfg, norm_cfg=norm_cfg))
                c_in = head_conv
            conv_layers.append(build_conv_layer(conv_cfg, head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            conv_layers = nn.Sequential(*conv_layers)
            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.
        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].
        Returns:
            dict[str: torch.Tensor]: contains the following keys:
                -reg （torch.Tensor): 2D regression value with the                     shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the                     shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape                     of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the                     shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the                     shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of                     [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(nn.Conv1d(input_channel, num_pos_feats, kernel_size=1), nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True), nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    if use_separate_proj_weight is not True:
        if qkv_same:
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:embed_dim * 2])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[embed_dim * 2:])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O
        \\text{where} head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented.                     Please re-train your model with the new module', UserWarning)
            return multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == 'relu':
                return F.relu
            if activation == 'gelu':
                return F.gelu
            if activation == 'glu':
                return F.glu
            raise RuntimeError(f'activation should be relu/gelu, not {activation}.')
        self.activation = _get_activation_fn(activation)
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)
        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed), key=self.with_pos_embed(key, key_pos_embed), value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        query = query.permute(1, 2, 0)
        return query


def build_loss(cfg):
    return LOSSES.build(cfg)


def clip_sigmoid(x, eps=0.0001):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [((ss - 1.0) / 2.0) for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(gaussian[radius - top:radius + bottom, radius - left:radius + right])
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out]].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2
    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


class BEVGridTransform(nn.Module):

    def __init__(self, *, input_scope: List[Tuple[float, float, float]], output_scope: List[Tuple[float, float, float]], prescale_factor: float=1) ->None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(x, scale_factor=self.prescale_factor, mode='bilinear', align_corners=False)
        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(self.input_scope, self.output_scope):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v)
        u, v = torch.meshgrid(coords, indexing='ij')
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        return x


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float=-1, gamma: float=2, reduction: str='mean') ->torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def sigmoid_xent_loss(inputs: torch.Tensor, targets: torch.Tensor, reduction: str='mean') ->torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


class BEVSegmentationHead(nn.Module):

    def __init__(self, in_channels: int, grid_transform: Dict[str, Any], classes: List[str], loss: str) ->None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(True), nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(True), nn.Conv2d(in_channels, len(classes), 1))

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor]=None) ->Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.transform(x)
        x = self.classifier(x)
        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == 'xent':
                    loss = sigmoid_xent_loss(x[:, index], target[:, index])
                elif self.loss == 'focal':
                    loss = sigmoid_focal_loss(x[:, index], target[:, index])
                else:
                    raise ValueError(f'unsupported loss: {self.loss}')
                losses[f'{name}/{self.loss}'] = loss
            return losses
        else:
            return torch.sigmoid(x)


class LSSFPN(nn.Module):

    def __init__(self, in_indices: Tuple[int, int], in_channels: Tuple[int, int], out_channels: int, scale_factor: int=1) ->None:
        super().__init__()
        self.in_indices = in_indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.fuse = nn.Sequential(nn.Conv2d(in_channels[0] + in_channels[1], out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        if scale_factor > 1:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

    def forward(self, x: List[torch.Tensor]) ->torch.Tensor:
        x1 = x[self.in_indices[0]]
        assert x1.shape[1] == self.in_channels[0]
        x2 = x[self.in_indices[1]]
        assert x2.shape[1] == self.in_channels[1]
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        if self.scale_factor > 1:
            x = self.upsample(x)
        return x


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([(row[0] + row[2] / 2.0) for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class FurthestPointSampling(Function):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor, num_points: int) ->torch.Tensor:
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_xyz.is_contiguous()
        B, N = points_xyz.size()[:2]
        output = torch.IntTensor(B, num_points)
        temp = torch.FloatTensor(B, N).fill_(10000000000.0)
        furthest_point_sample_ext.furthest_point_sampling_wrapper(B, N, num_points, points_xyz, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class DFPS_Sampler(nn.Module):
    """DFPS_Sampling.

    Using Euclidean distances of points for FPS.
    """

    def __init__(self):
        super(DFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with D-FPS."""
        fps_idx = furthest_point_sample(points.contiguous(), npoint)
        return fps_idx


def calc_square_dist(point_feat_a, point_feat_b, norm=True):
    """Calculating square distance between a and b.

    Args:
        point_feat_a (Tensor): (B, N, C) Feature vector of each point.
        point_feat_b (Tensor): (B, M, C) Feature vector of each point.
        norm (Bool): Whether to normalize the distance.
            Default: True.

    Returns:
        Tensor: (B, N, M) Distance between each pair points.
    """
    length_a = point_feat_a.shape[1]
    length_b = point_feat_b.shape[1]
    num_channel = point_feat_a.shape[-1]
    a_square = torch.sum(point_feat_a.unsqueeze(dim=2).pow(2), dim=-1)
    b_square = torch.sum(point_feat_b.unsqueeze(dim=1).pow(2), dim=-1)
    a_square = a_square.repeat((1, 1, length_b))
    b_square = b_square.repeat((1, length_a, 1))
    coor = torch.matmul(point_feat_a, point_feat_b.transpose(1, 2))
    dist = a_square + b_square - 2 * coor
    if norm:
        dist = torch.sqrt(dist) / num_channel
    return dist


class FurthestPointSamplingWithDist(Function):
    """Furthest Point Sampling With Distance.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    def forward(ctx, points_dist: torch.Tensor, num_points: int) ->torch.Tensor:
        """forward.

        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_dist.is_contiguous()
        B, N, _ = points_dist.size()
        output = points_dist.new_zeros([B, num_points], dtype=torch.int32)
        temp = points_dist.new_zeros([B, N]).fill_(10000000000.0)
        furthest_point_sample_ext.furthest_point_sampling_with_dist_wrapper(B, N, num_points, points_dist, temp, output)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply


class FFPS_Sampler(nn.Module):
    """FFPS_Sampler.

    Using feature distances for FPS.
    """

    def __init__(self):
        super(FFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with F-FPS."""
        assert features is not None, 'feature input to FFPS_Sampler should not be None'
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(features_for_fps, features_for_fps, norm=False)
        fps_idx = furthest_point_sample_with_dist(features_dist, npoint)
        return fps_idx


class FS_Sampler(nn.Module):
    """FS_Sampling.

    Using F-FPS and D-FPS simultaneously.
    """

    def __init__(self):
        super(FS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with FS_Sampling."""
        assert features is not None, 'feature input to FS_Sampler should not be None'
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(features_for_fps, features_for_fps, norm=False)
        fps_idx_ffps = furthest_point_sample_with_dist(features_dist, npoint)
        fps_idx_dfps = furthest_point_sample(points, npoint)
        fps_idx = torch.cat([fps_idx_ffps, fps_idx_dfps], dim=1)
        return fps_idx


def get_sampler_type(sampler_type):
    """Get the type and mode of points sampler.

    Args:
        sampler_type (str): The type of points sampler.
            The valid value are "D-FPS", "F-FPS", or "FS".

    Returns:
        class: Points sampler type.
    """
    if sampler_type == 'D-FPS':
        sampler = DFPS_Sampler
    elif sampler_type == 'F-FPS':
        sampler = FFPS_Sampler
    elif sampler_type == 'FS':
        sampler = FS_Sampler
    else:
        raise ValueError(f'Only "sampler_type" of "D-FPS", "F-FPS", or "FS" are supported, got {sampler_type}')
    return sampler


class BallQuery(Function):
    """Ball Query.

    Find nearby points in spherical space.
    """

    @staticmethod
    def forward(ctx, min_radius: float, max_radius: float, sample_num: int, xyz: torch.Tensor, center_xyz: torch.Tensor) ->torch.Tensor:
        """forward.

        Args:
            min_radius (float): minimum radius of the balls.
            max_radius (float): maximum radius of the balls.
            sample_num (int): maximum number of features in the balls.
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) centers of the ball query.

        Returns:
            Tensor: (B, npoint, nsample) tensor with the indicies of
                the features that form the query balls.
        """
        assert center_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert min_radius < max_radius
        B, N, _ = xyz.size()
        npoint = center_xyz.size(1)
        idx = torch.IntTensor(B, npoint, sample_num).zero_()
        ball_query_ext.ball_query_wrapper(B, N, npoint, min_radius, max_radius, sample_num, center_xyz, xyz, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):
    """Grouping Operation.

    Group feature with given index.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) ->torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) tensor of features to group.
            indices (Tensor): (B, npoint, nsample) the indicies of
                features to group with.

        Returns:
            Tensor: (B, C, npoint, nsample) Grouped features.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        B, nfeatures, nsample = indices.size()
        _, C, N = features.size()
        output = torch.FloatTensor(B, C, nfeatures, nsample)
        group_points_ext.forward(B, C, N, nfeatures, nsample, features, indices, output)
        ctx.for_backwards = indices, N
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """backward.

        Args:
            grad_out (Tensor): (B, C, npoint, nsample) tensor of the gradients
                of the output from forward.

        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        idx, N = ctx.for_backwards
        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        group_points_ext.backward(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class KNN(Function):
    """KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx, k: int, xyz: torch.Tensor, center_xyz: torch.Tensor=None, transposed: bool=False) ->torch.Tensor:
        """Forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) if transposed == False,
                else (B, 3, npoint). centers of the knn query.
            transposed (bool): whether the input tensors are transposed.
                defaults to False. Should not expicitly use this keyword
                when calling knn (=KNN.apply), just add the fourth param.

        Returns:
            Tensor: (B, k, npoint) tensor with the indicies of
                the features that form k-nearest neighbours.
        """
        assert k > 0
        if center_xyz is None:
            center_xyz = xyz
        if transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()
        assert xyz.is_contiguous()
        assert center_xyz.is_contiguous()
        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), 'center_xyz and xyz should be put on the same device'
        if torch.cuda.current_device() != center_xyz_device:
            torch.cuda.set_device(center_xyz_device)
        B, npoint, _ = center_xyz.shape
        N = xyz.shape[1]
        idx = center_xyz.new_zeros((B, npoint, k)).int()
        dist2 = center_xyz.new_zeros((B, npoint, k)).float()
        knn_ext.knn_wrapper(B, N, npoint, k, xyz, center_xyz, idx, dist2)
        idx = idx.transpose(2, 1).contiguous()
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knn = KNN.apply


class QueryAndGroup(nn.Module):
    """Query and Group.

    Groups with a ball query of radius

    Args:
        max_radius (float | None): The maximum radius of the balls.
            If None is given, we will use kNN sampling instead of ball query.
        sample_num (int): Maximum number of features to gather in the ball.
        min_radius (float): The minimum radius of the balls.
        use_xyz (bool): Whether to use xyz.
            Default: True.
        return_grouped_xyz (bool): Whether to return grouped xyz.
            Default: False.
        normalize_xyz (bool): Whether to normalize xyz.
            Default: False.
        uniform_sample (bool): Whether to sample uniformly.
            Default: False
        return_unique_cnt (bool): Whether to return the count of
            unique samples.
            Default: False.
        return_grouped_idx (bool): Whether to return grouped idx.
            Default: False.
    """

    def __init__(self, max_radius, sample_num, min_radius=0, use_xyz=True, return_grouped_xyz=False, normalize_xyz=False, uniform_sample=False, return_unique_cnt=False, return_grouped_idx=False):
        super(QueryAndGroup, self).__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.uniform_sample = uniform_sample
        self.return_unique_cnt = return_unique_cnt
        self.return_grouped_idx = return_grouped_idx
        if self.return_unique_cnt:
            assert self.uniform_sample, 'uniform_sample should be True when returning the count of unique samples'
        if self.max_radius is None:
            assert not self.normalize_xyz, 'can not normalize grouped xyz when max_radius is None'

    def forward(self, points_xyz, center_xyz, features=None):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) Centriods.
            features (Tensor): (B, C, N) Descriptors of the features.

        Return：
            Tensor: (B, 3 + C, npoint, sample_num) Grouped feature.
        """
        if self.max_radius is None:
            idx = knn(self.sample_num, points_xyz, center_xyz, False)
            idx = idx.transpose(1, 2).contiguous()
        else:
            idx = ball_query(self.min_radius, self.max_radius, self.sample_num, points_xyz, center_xyz)
        if self.uniform_sample:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.sample_num - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz_diff = grouped_xyz - center_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz_diff /= self.max_radius
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz_diff, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz_diff
        ret = [new_features]
        if self.return_grouped_xyz:
            ret.append(grouped_xyz)
        if self.return_unique_cnt:
            ret.append(unique_cnt)
        if self.return_grouped_idx:
            ret.append(idx)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """Group All.

    Group xyz with feature.

    Args:
        use_xyz (bool): Whether to use xyz.
    """

    def __init__(self, use_xyz: bool=True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor=None):
        """forward.

        Args:
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            new_xyz (Tensor): Ignored.
            features (Tensor): (B, C, N) features to group.

        Return:
            Tensor: (B, C + 3, 1, N) Grouped feature.
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class ScoreNet(nn.Module):
    """ScoreNet that outputs coefficient scores to assemble kernel weights in
    the weight bank according to the relative position of point pairs.

    Args:
        mlp_channels (List[int]): Hidden unit sizes of SharedMLP layers.
        last_bn (bool, optional): Whether to use BN on the last output of mlps.
            Defaults to False.
        score_norm (str, optional): Normalization function of output scores.
            Can be 'softmax', 'sigmoid' or 'identity'. Defaults to 'softmax'.
        temp_factor (float, optional): Temperature factor to scale the output
            scores before softmax. Defaults to 1.0.
        norm_cfg (dict, optional): Type of normalization method.
            Defaults to dict(type='BN2d').
        bias (bool | str, optional): If specified as `auto`, it will be decided
            by the norm_cfg. Bias will be set as True if `norm_cfg` is None,
            otherwise False. Defaults to 'auto'.

    Note:
        The official code applies xavier_init to all Conv layers in ScoreNet,
            see `PAConv <https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg
            /model/pointnet2/paconv.py#L105>`_. However in our experiments, we
            did not find much difference in applying such xavier initialization
            or not. So we neglect this initialization in our implementation.
    """

    def __init__(self, mlp_channels, last_bn=False, score_norm='softmax', temp_factor=1.0, norm_cfg=dict(type='BN2d'), bias='auto'):
        super(ScoreNet, self).__init__()
        assert score_norm in ['softmax', 'sigmoid', 'identity'], f'unsupported score_norm function {score_norm}'
        self.score_norm = score_norm
        self.temp_factor = temp_factor
        self.mlps = nn.Sequential()
        for i in range(len(mlp_channels) - 2):
            self.mlps.add_module(f'layer{i}', ConvModule(mlp_channels[i], mlp_channels[i + 1], kernel_size=(1, 1), stride=(1, 1), conv_cfg=dict(type='Conv2d'), norm_cfg=norm_cfg, bias=bias))
        i = len(mlp_channels) - 2
        self.mlps.add_module(f'layer{i}', ConvModule(mlp_channels[i], mlp_channels[i + 1], kernel_size=(1, 1), stride=(1, 1), conv_cfg=dict(type='Conv2d'), norm_cfg=norm_cfg if last_bn else None, act_cfg=None, bias=bias))

    def forward(self, xyz_features):
        """Forward.

        Args:
            xyz_features (torch.Tensor): (B, C, N, K), features constructed
                from xyz coordinates of point pairs. May contain relative
                positions, Euclidian distance, etc.

        Returns:
            torch.Tensor: (B, N, K, M), predicted scores for `M` kernels.
        """
        scores = self.mlps(xyz_features)
        if self.score_norm == 'softmax':
            scores = F.softmax(scores / self.temp_factor, dim=1)
        elif self.score_norm == 'sigmoid':
            scores = torch.sigmoid(scores / self.temp_factor)
        else:
            scores = scores
        scores = scores.permute(0, 2, 3, 1)
        return scores


def assign_score(scores, point_features):
    """Perform weighted sum to aggregate output features according to scores.
    This function is used in non-CUDA version of PAConv.

    Compared to the cuda op assigh_score_withk, this pytorch implementation
        pre-computes output features for the neighbors of all centers, and then
        performs aggregation. It consumes more GPU memories.

    Args:
        scores (torch.Tensor): (B, npoint, K, M), predicted scores to
            aggregate weight matrices in the weight bank.
            `npoint` is the number of sampled centers.
            `K` is the number of queried neighbors.
            `M` is the number of weight matrices in the weight bank.
        point_features (torch.Tensor): (B, npoint, K, M, out_dim)
            Pre-computed point features to be aggregated.

    Returns:
        torch.Tensor: (B, npoint, K, out_dim), the aggregated features.
    """
    B, npoint, K, M = scores.size()
    scores = scores.view(B, npoint, K, 1, M)
    output = torch.matmul(scores, point_features).view(B, npoint, K, -1)
    return output


def calc_euclidian_dist(xyz1, xyz2):
    """Calculate the Euclidian distance between two sets of points.

    Args:
        xyz1 (torch.Tensor): (N, 3), the first set of points.
        xyz2 (torch.Tensor): (N, 3), the second set of points.

    Returns:
        torch.Tensor: (N, ), the Euclidian distance between each point pair.
    """
    assert xyz1.shape[0] == xyz2.shape[0], 'number of points are not the same'
    assert xyz1.shape[1] == xyz2.shape[1] == 3, 'points coordinates dimension is not 3'
    return torch.norm(xyz1 - xyz2, dim=-1)


class PAConv(nn.Module):
    """Non-CUDA version of PAConv.

    PAConv stores a trainable weight bank containing several kernel weights.
    Given input points and features, it computes coefficient scores to assemble
    those kernels to form conv kernels, and then runs convolution on the input.

    Args:
        in_channels (int): Input channels of point features.
        out_channels (int): Output channels of point features.
        num_kernels (int): Number of kernel weights in the weight bank.
        norm_cfg (dict, optional): Type of normalization method.
            Defaults to dict(type='BN2d', momentum=0.1).
        act_cfg (dict, optional): Type of activation method.
            Defaults to dict(type='ReLU', inplace=True).
        scorenet_input (str, optional): Type of input to ScoreNet.
            Can be 'identity', 'w_neighbor' or 'w_neighbor_dist'.
            Defaults to 'w_neighbor_dist'.
        weight_bank_init (str, optional): Init method of weight bank kernels.
            Can be 'kaiming' or 'xavier'. Defaults to 'kaiming'.
        kernel_input (str, optional): Input features to be multiplied with
            kernel weights. Can be 'identity' or 'w_neighbor'.
            Defaults to 'w_neighbor'.
        scorenet_cfg (dict, optional): Config of the ScoreNet module, which
            may contain the following keys and values:

            - mlp_channels (List[int]): Hidden units of MLPs.
            - score_norm (str): Normalization function of output scores.
                Can be 'softmax', 'sigmoid' or 'identity'.
            - temp_factor (float): Temperature factor to scale the output
                scores before softmax.
            - last_bn (bool): Whether to use BN on the last output of mlps.
    """

    def __init__(self, in_channels, out_channels, num_kernels, norm_cfg=dict(type='BN2d', momentum=0.1), act_cfg=dict(type='ReLU', inplace=True), scorenet_input='w_neighbor_dist', weight_bank_init='kaiming', kernel_input='w_neighbor', scorenet_cfg=dict(mlp_channels=[16, 16, 16], score_norm='softmax', temp_factor=1.0, last_bn=False)):
        super(PAConv, self).__init__()
        if kernel_input == 'identity':
            kernel_mul = 1
        elif kernel_input == 'w_neighbor':
            kernel_mul = 2
        else:
            raise NotImplementedError(f'unsupported kernel_input {kernel_input}')
        self.kernel_input = kernel_input
        in_channels = kernel_mul * in_channels
        if scorenet_input == 'identity':
            self.scorenet_in_channels = 3
        elif scorenet_input == 'w_neighbor':
            self.scorenet_in_channels = 6
        elif scorenet_input == 'w_neighbor_dist':
            self.scorenet_in_channels = 7
        else:
            raise NotImplementedError(f'unsupported scorenet_input {scorenet_input}')
        self.scorenet_input = scorenet_input
        if weight_bank_init == 'kaiming':
            weight_init = nn.init.kaiming_normal_
        elif weight_bank_init == 'xavier':
            weight_init = nn.init.xavier_normal_
        else:
            raise NotImplementedError(f'unsupported weight bank init method {weight_bank_init}')
        self.num_kernels = num_kernels
        weight_bank = weight_init(torch.empty(self.num_kernels, in_channels, out_channels))
        weight_bank = weight_bank.permute(1, 0, 2).reshape(in_channels, self.num_kernels * out_channels).contiguous()
        self.weight_bank = nn.Parameter(weight_bank, requires_grad=True)
        scorenet_cfg_ = copy.deepcopy(scorenet_cfg)
        scorenet_cfg_['mlp_channels'].insert(0, self.scorenet_in_channels)
        scorenet_cfg_['mlp_channels'].append(self.num_kernels)
        self.scorenet = ScoreNet(**scorenet_cfg_)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg is not None else None
        self.activate = build_activation_layer(act_cfg) if act_cfg is not None else None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_weights()

    def init_weights(self):
        """Initialize weights of shared MLP layers and BN layers."""
        if self.bn is not None:
            constant_init(self.bn, val=1, bias=0)

    def _prepare_scorenet_input(self, points_xyz):
        """Prepare input point pairs features for self.ScoreNet.

        Args:
            points_xyz (torch.Tensor): (B, 3, npoint, K)
                Coordinates of the grouped points.

        Returns:
            torch.Tensor: (B, C, npoint, K)
                The generated features per point pair.
        """
        B, _, npoint, K = points_xyz.size()
        center_xyz = points_xyz[..., :1].repeat(1, 1, 1, K)
        xyz_diff = points_xyz - center_xyz
        if self.scorenet_input == 'identity':
            xyz_features = xyz_diff
        elif self.scorenet_input == 'w_neighbor':
            xyz_features = torch.cat((xyz_diff, points_xyz), dim=1)
        else:
            euclidian_dist = calc_euclidian_dist(center_xyz.permute(0, 2, 3, 1).reshape(B * npoint * K, 3), points_xyz.permute(0, 2, 3, 1).reshape(B * npoint * K, 3)).reshape(B, 1, npoint, K)
            xyz_features = torch.cat((center_xyz, xyz_diff, euclidian_dist), dim=1)
        return xyz_features

    def forward(self, inputs):
        """Forward.

        Args:
            inputs (tuple(torch.Tensor)):

                - features (torch.Tensor): (B, in_c, npoint, K)
                    Features of the queried points.
                - points_xyz (torch.Tensor): (B, 3, npoint, K)
                    Coordinates of the grouped points.

        Returns:
            Tuple[torch.Tensor]:

                - new_features: (B, out_c, npoint, K), features after PAConv.
                - points_xyz: same as input.
        """
        features, points_xyz = inputs
        B, _, npoint, K = features.size()
        if self.kernel_input == 'w_neighbor':
            center_features = features[..., :1].repeat(1, 1, 1, K)
            features_diff = features - center_features
            features = torch.cat((features_diff, features), dim=1)
        xyz_features = self._prepare_scorenet_input(points_xyz)
        scores = self.scorenet(xyz_features)
        new_features = torch.matmul(features.permute(0, 2, 3, 1), self.weight_bank).view(B, npoint, K, self.num_kernels, -1)
        new_features = assign_score(scores, new_features)
        new_features = new_features.permute(0, 3, 1, 2).contiguous()
        if self.bn is not None:
            new_features = self.bn(new_features)
        if self.activate is not None:
            new_features = self.activate(new_features)
        return new_features, points_xyz


def assign_kernel_withoutk(features, kernels, M):
    """Pre-compute features with weight matrices in weight bank. This function
    is used before cuda op assign_score_withk in CUDA version PAConv.

    Args:
        features (torch.Tensor): (B, in_dim, N), input features of all points.
            `N` is the number of points in current point cloud.
        kernels (torch.Tensor): (2 * in_dim, M * out_dim), weight matrices in
            the weight bank, transformed from (M, 2 * in_dim, out_dim).
            `2 * in_dim` is because the input features are concatenation of
            (point_features - center_features, point_features).
        M (int): Number of weight matrices in the weight bank.

    Returns:
        Tuple[torch.Tensor]: both of shape (B, N, M, out_dim):

            - point_features: Pre-computed features for points.
            - center_features: Pre-computed features for centers.
    """
    B, in_dim, N = features.size()
    feat_trans = features.permute(0, 2, 1)
    out_feat_half1 = torch.matmul(feat_trans, kernels[:in_dim]).view(B, N, M, -1)
    out_feat_half2 = torch.matmul(feat_trans, kernels[in_dim:]).view(B, N, M, -1)
    if features.size(1) % 2 != 0:
        out_feat_half_coord = torch.matmul(feat_trans[:, :, :3], kernels[in_dim:in_dim + 3]).view(B, N, M, -1)
    else:
        out_feat_half_coord = torch.zeros_like(out_feat_half2)
    point_features = out_feat_half1 + out_feat_half2
    center_features = out_feat_half1 + out_feat_half_coord
    return point_features, center_features


class PAConvCUDA(PAConv):
    """CUDA version of PAConv that implements a cuda op to efficiently perform
    kernel assembling.

    Different from vanilla PAConv, the input features of this function is not
    grouped by centers. Instead, they will be queried on-the-fly by the
    additional input `points_idx`. This avoids the large intermediate matrix.
    See the `paper <https://arxiv.org/pdf/2103.14635.pdf>`_ appendix Sec. D for
    more detailed descriptions.
    """

    def __init__(self, in_channels, out_channels, num_kernels, norm_cfg=dict(type='BN2d', momentum=0.1), act_cfg=dict(type='ReLU', inplace=True), scorenet_input='w_neighbor_dist', weight_bank_init='kaiming', kernel_input='w_neighbor', scorenet_cfg=dict(mlp_channels=[8, 16, 16], score_norm='softmax', temp_factor=1.0, last_bn=False)):
        super(PAConvCUDA, self).__init__(in_channels=in_channels, out_channels=out_channels, num_kernels=num_kernels, norm_cfg=norm_cfg, act_cfg=act_cfg, scorenet_input=scorenet_input, weight_bank_init=weight_bank_init, kernel_input=kernel_input, scorenet_cfg=scorenet_cfg)
        assert self.kernel_input == 'w_neighbor', 'CUDA implemented PAConv only supports w_neighbor kernel_input'

    def forward(self, inputs):
        """Forward.

        Args:
            inputs (tuple(torch.Tensor)):

                - features (torch.Tensor): (B, in_c, N)
                    Features of all points in the current point cloud.
                    Different from non-CUDA version PAConv, here the features
                        are not grouped by each center to form a K dim.
                - points_xyz (torch.Tensor): (B, 3, npoint, K)
                    Coordinates of the grouped points.
                - points_idx (torch.Tensor): (B, npoint, K)
                    Index of the grouped points.

        Returns:
            Tuple[torch.Tensor]:

                - new_features: (B, out_c, npoint, K), features after PAConv.
                - points_xyz: same as input.
                - points_idx: same as input.
        """
        features, points_xyz, points_idx = inputs
        xyz_features = self._prepare_scorenet_input(points_xyz)
        scores = self.scorenet(xyz_features)
        point_feat, center_feat = assign_kernel_withoutk(features, self.weight_bank, self.num_kernels)
        new_features = assign_score_cuda(scores, point_feat, center_feat, points_idx, 'sum').contiguous()
        if self.bn is not None:
            new_features = self.bn(new_features)
        if self.activate is not None:
            new_features = self.activate(new_features)
        return new_features, points_xyz, points_idx


class GatherPoints(Function):
    """Gather Points.

    Gather points with given index.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) ->torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) features to gather.
            indices (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        B, npoint = indices.size()
        _, C, N = features.size()
        output = torch.FloatTensor(B, C, npoint)
        gather_points_ext.gather_points_wrapper(B, C, N, npoint, features, indices, output)
        ctx.for_backwards = indices, C, N
        ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()
        grad_features = torch.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        gather_points_ext.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_points = GatherPoints.apply


class BasePointSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz in
            `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx in
            `QueryAndGroup`. Defaults to False.
    """

    def __init__(self, num_point, radii, sample_nums, mlp_channels, fps_mod=['D-FPS'], fps_sample_range_list=[-1], dilated_group=False, use_xyz=True, pool_mod='max', normalize_xyz=False, grouper_return_grouped_xyz=False, grouper_return_grouped_idx=False):
        super(BasePointSAModule, self).__init__()
        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)
        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels
        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')
        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list
        self.points_sampler = Points_Sampler(self.num_point, self.fps_mod_list, self.fps_sample_range_list)
        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(radius, sample_num, min_radius=min_radius, use_xyz=use_xyz, normalize_xyz=normalize_xyz, return_grouped_xyz=grouper_return_grouped_xyz, return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _sample_points(self, points_xyz, features, indices, target_xyz):
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, num_point, 3) sampled xyz coordinates of points.
            Tensor: (B, num_point) sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert indices.shape[1] == self.num_point[0]
            new_xyz = gather_points(xyz_flipped, indices).transpose(1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            indices = self.points_sampler(points_xyz, features)
            new_xyz = gather_points(xyz_flipped, indices).transpose(1, 2).contiguous() if self.num_point is not None else None
        return new_xyz, indices

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            new_features = F.max_pool2d(features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            new_features = F.avg_pool2d(features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError
        return new_features.squeeze(-1).contiguous()

    def forward(self, points_xyz, features=None, indices=None, target_xyz=None):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []
        new_xyz, indices = self._sample_points(points_xyz, features, indices, target_xyz)
        for i in range(len(self.groupers)):
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)
            new_features = self.mlps[i](grouped_results)
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), indices


class PointSAModuleMSG(BasePointSAModule):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self, num_point, radii, sample_nums, mlp_channels, fps_mod=['D-FPS'], fps_sample_range_list=[-1], dilated_group=False, norm_cfg=dict(type='BN2d'), use_xyz=True, pool_mod='max', normalize_xyz=False, bias='auto'):
        super(PointSAModuleMSG, self).__init__(num_point=num_point, radii=radii, sample_nums=sample_nums, mlp_channels=mlp_channels, fps_mod=fps_mod, fps_sample_range_list=fps_sample_range_list, dilated_group=dilated_group, use_xyz=use_xyz, pool_mod=pool_mod, normalize_xyz=normalize_xyz)
        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                mlp_channel[0] += 3
            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(f'layer{i}', ConvModule(mlp_channel[i], mlp_channel[i + 1], kernel_size=(1, 1), stride=(1, 1), conv_cfg=dict(type='Conv2d'), norm_cfg=norm_cfg, bias=bias))
            self.mlps.append(mlp)


class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self, mlp_channels, num_point=None, radius=None, num_sample=None, norm_cfg=dict(type='BN2d'), use_xyz=True, pool_mod='max', fps_mod=['D-FPS'], fps_sample_range_list=[-1], normalize_xyz=False):
        super(PointSAModule, self).__init__(mlp_channels=[mlp_channels], num_point=num_point, radii=[radius], sample_nums=[num_sample], norm_cfg=norm_cfg, use_xyz=use_xyz, pool_mod=pool_mod, fps_mod=fps_mod, fps_sample_range_list=fps_sample_range_list, normalize_xyz=normalize_xyz)


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_per_voxel, mode):
        """RoIAwarePool3d function forward.

        Args:
            rois (torch.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (int): 0 (max pool) or 1 (average pool)

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            assert mmcv.is_tuple_of(out_size, int)
            out_x, out_y, out_z = out_size
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_per_voxel), dtype=torch.int)
        roiaware_pool3d_ext.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, mode)
        ctx.roiaware_pool3d_for_backward = pts_idx_of_voxels, argmax, mode, num_pts, num_channels
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """RoIAwarePool3d function forward.

        Args:
            grad_out (torch.Tensor): [N, out_x, out_y, out_z, C]
        Returns:
            grad_in (torch.Tensor): [npoints, C]
        """
        ret = ctx.roiaware_pool3d_for_backward
        pts_idx_of_voxels, argmax, mode, num_pts, num_channels = ret
        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_ext.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, mode)
        return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_per_voxel=128, mode='max'):
        super().__init__()
        """RoIAwarePool3d module

        Args:
            out_size (int or tuple): n or [n1, n2, n3]
            max_pts_per_voxel (int): m
            mode (str): 'max' or 'avg'
        """
        self.out_size = out_size
        self.max_pts_per_voxel = max_pts_per_voxel
        assert mode in ['max', 'avg']
        pool_method_map = {'max': 0, 'avg': 1}
        self.mode = pool_method_map[mode]

    def forward(self, rois, pts, pts_feature):
        """RoIAwarePool3d module forward.

        Args:
            rois (torch.Tensor): [N, 7],in LiDAR coordinate,
                (x, y, z) is the bottom center of rois
            pts (torch.Tensor): [npoints, 3]
            pts_feature (torch.Tensor): [npoints, C]

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_per_voxel, self.mode)


class SparseModule(nn.Module):
    """place holder, All module subclass from this will take sptensor in
    SparseSequential."""
    pass


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class SparseConvTensor:

    def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
        """
        Args:
            grid: pre-allocated grid tensor.
                  should be used when the volume of spatial shape
                  is very large.
        """
        self.features = features
        self.indices = indices
        if self.indices.dtype != torch.int32:
            self.indices.int()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first=True):
        output_shape = [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(self.indices.long(), self.features, output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size


def is_sparse_conv(module):
    return isinstance(module, SparseConvolution)


def is_spconv_module(module):
    spconv_modules = SparseModule,
    return isinstance(module, spconv_modules)


class SparseSequential(SparseModule):
    """A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = SparseSequential(
                  SparseConv2d(1,20,5),
                  nn.ReLU(),
                  SparseConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = SparseSequential(OrderedDict([
                  ('conv1', SparseConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', SparseConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = SparseSequential(
                  conv1=SparseConv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=SparseConv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError('kwargs only supported in py36+')
            if name in self._modules:
                raise ValueError('name exists.')
            self.add_module(name, module)
        self._sparity_dict = {}

    def __getitem__(self, idx):
        if not -len(self) <= idx < len(self):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    @property
    def sparity_dict(self):
        return self._sparity_dict

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError('name exists')
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            if is_spconv_module(module):
                assert isinstance(input, SparseConvTensor)
                self._sparity_dict[k] = input.sparity
                input = module(input)
            elif isinstance(input, SparseConvTensor):
                if input.indices.shape[0] != 0:
                    input.features = module(input.features)
            else:
                input = module(input)
        return input

    def fused(self):
        """don't use this.

        no effect.
        """
        mods = [v for k, v in self._modules.items()]
        fused_mods = []
        idx = 0
        while idx < len(mods):
            if is_sparse_conv(mods[idx]):
                if idx < len(mods) - 1 and isinstance(mods[idx + 1], nn.BatchNorm1d):
                    new_module = SparseConvolution(ndim=mods[idx].ndim, in_channels=mods[idx].in_channels, out_channels=mods[idx].out_channels, kernel_size=mods[idx].kernel_size, stride=mods[idx].stride, padding=mods[idx].padding, dilation=mods[idx].dilation, groups=mods[idx].groups, bias=True, subm=mods[idx].subm, output_padding=mods[idx].output_padding, transposed=mods[idx].transposed, inverse=mods[idx].inverse, indice_key=mods[idx].indice_key, fused_bn=True)
                    new_module.load_state_dict(mods[idx].state_dict(), False)
                    new_module
                    conv = new_module
                    bn = mods[idx + 1]
                    conv.bias.data.zero_()
                    conv.weight.data[:] = conv.weight.data * bn.weight.data / (torch.sqrt(bn.running_var) + bn.eps)
                    conv.bias.data[:] = (conv.bias.data - bn.running_mean) * bn.weight.data / (torch.sqrt(bn.running_var) + bn.eps) + bn.bias.data
                    fused_mods.append(conv)
                    idx += 2
                else:
                    fused_mods.append(mods[idx])
                    idx += 1
            else:
                fused_mods.append(mods[idx])
                idx += 1
        return SparseSequential(*fused_mods)


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor."""

    def forward(self, x: SparseConvTensor):
        return x.dense()


class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer."""

    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x


class _dynamic_scatter(Function):

    @staticmethod
    def forward(ctx, feats, coors, reduce_type='max'):
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats: [N, C] float tensor. points features to be reduced
                into voxels.
            coors: [N, ndim] int tensor. corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type: str. reduce op. support 'max', 'sum' and 'mean'
        Returns:
            tuple
            voxel_feats: [M, C] float tensor. reduced features. input features
                that shares the same voxel coordinates are reduced to one row
            coordinates: [M, ndim] int tensor, voxel coordinates.
        """
        results = dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        voxel_feats, voxel_coors, point2voxel_map, voxel_points_count = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map, voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx, grad_voxel_feats, grad_voxel_coors=None):
        feats, voxel_feats, point2voxel_map, voxel_points_count = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        dynamic_point_to_voxel_backward(grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats, point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter = _dynamic_scatter.apply


class DynamicScatter(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, average_points: bool):
        super(DynamicScatter, self).__init__()
        """Scatters points into voxels, used in the voxel encoder with
           dynamic voxelization

        **Note**: The CPU and GPU implementation get the same output, but
        have numerical difference after summation and division (e.g., 5e-7).

        Args:
            average_points (bool): whether to use avg pooling to scatter
                points into voxel voxel_size (list): list [x, y, z] size
                of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points, coors):
        """
        Args:
            input: NC points
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(points[inds], coors[inds][:, 1:])
                coor_pad = nn.functional.pad(voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)
            return features, feature_coors

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', average_points=' + str(self.average_points)
        tmpstr += ')'
        return tmpstr


class _Voxelization(Function):

    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
            voxel_num = hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3, deterministic)
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
        return voxelization(input, self.voxel_size, self.point_cloud_range, self.max_num_points, max_voxels, self.deterministic)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionEmbeddingLearned,
     lambda: ([], {'input_channel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (RemoveGrid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SparseSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_mit_han_lab_bevfusion(_paritybench_base):
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

