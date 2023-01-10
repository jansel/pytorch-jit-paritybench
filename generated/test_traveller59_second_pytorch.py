import sys
_module = sys.modules[__name__]
del sys
second = _module
builder = _module
anchor_generator_builder = _module
dataset_builder = _module
dbsampler_builder = _module
preprocess_builder = _module
similarity_calculator_builder = _module
target_assigner_builder = _module
voxel_builder = _module
core = _module
anchor_generator = _module
box_coders = _module
box_np_ops = _module
geometry = _module
inference = _module
non_max_suppression = _module
nms_cpu = _module
nms_gpu = _module
preprocess = _module
region_similarity = _module
sample_ops = _module
target_assigner = _module
target_ops = _module
create_data = _module
data = _module
all_dataset = _module
dataset = _module
kitti_common = _module
kitti_dataset = _module
nusc_eval = _module
nuscenes_dataset = _module
framework = _module
test = _module
kittiviewer = _module
backend = _module
main = _module
control_panel = _module
glwidget = _module
viewer = _module
protos = _module
anchors_pb2 = _module
box_coder_pb2 = _module
input_reader_pb2 = _module
losses_pb2 = _module
model_pb2 = _module
optimizer_pb2 = _module
pipeline_pb2 = _module
preprocess_pb2 = _module
sampler_pb2 = _module
second_pb2 = _module
similarity_pb2 = _module
target_pb2 = _module
train_pb2 = _module
voxel_generator_pb2 = _module
pytorch = _module
box_coder_builder = _module
input_reader_builder = _module
losses_builder = _module
lr_scheduler_builder = _module
optimizer_builder = _module
second_builder = _module
box_coders = _module
box_torch_ops = _module
ghm_loss = _module
losses = _module
inference = _module
models = _module
middle = _module
net_multi_head = _module
pointpillars = _module
resnet = _module
rpn = _module
voxel_encoder = _module
voxelnet = _module
train = _module
utils = _module
script = _module
script_server = _module
bbox_plot = _module
check = _module
config_tool = _module
eval = _module
find = _module
loader = _module
log_tool = _module
merge_result = _module
model_tool = _module
progress_bar = _module
simplevis = _module
timer = _module
torchplus = _module
metrics = _module
nn = _module
functional = _module
modules = _module
common = _module
normalization = _module
ops = _module
array_ops = _module
tools = _module
checkpoint = _module
fastai_optim = _module
learning_schedules = _module
learning_schedules_fastai = _module
optim = _module

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


import time


import torch


import numpy as np


from torch.utils.data import Dataset


from torch import nn


from functools import partial


import math


from functools import reduce


from torch import stack as tstack


from abc import ABCMeta


from abc import abstractmethod


from torch.autograd import Variable


from torch.nn import functional as F


from enum import Enum


from torchvision.models import resnet


import copy


import re


import torch.nn.functional as F


from collections import OrderedDict


import functools


import inspect


import logging


from collections import Iterable


from collections import defaultdict


from copy import deepcopy


from itertools import chain


from torch._utils import _unflatten_dense_tensors


from torch.nn.utils import parameters_to_vector


from torch.optim.optimizer import Optimizer


class Empty(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def change_default_args(**kwargs):

    def layer_wrapper(layer_class):


        class DefaultArgLayer(layer_class):

            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)
        return DefaultArgLayer
    return layer_wrapper


def register_middle(cls, name=None):
    global REGISTERED_MIDDLE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MIDDLE_CLASSES, f'exist class: {REGISTERED_MIDDLE_CLASSES}'
    REGISTERED_MIDDLE_CLASSES[name] = cls
    return cls


class SparseMiddleExtractor(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []
        num_filters = [num_input_features] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d1:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False, indice_key='subm0'))
            middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(spconv.SparseConv3d(num_filters[-1], num_filters[-1], (3, 1, 1), (2, 1, 1), bias=False))
        middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False, indice_key='subm1'))
            middle_layers.append(BatchNorm1d(o))
            middle_layers.append(nn.ReLU())
        middle_layers.append(spconv.SparseConv3d(num_filters[-1], num_filters[-1], (3, 1, 1), (2, 1, 1), bias=False))
        middle_layers.append(BatchNorm1d(num_filters[-1]))
        middle_layers.append(nn.ReLU())
        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHD(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, indice_key='subm0'), BatchNorm1d(16), nn.ReLU(), SubMConv3d(16, 16, 3, indice_key='subm0'), BatchNorm1d(16), nn.ReLU(), SpConv3d(16, 32, 3, 2, padding=1), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1'), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1'), BatchNorm1d(32), nn.ReLU(), SpConv3d(32, 64, 3, 2, padding=1), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU())
        self.max_batch_size = 6

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHDPeople(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddleFHD'):
        super(SpMiddleFHDPeople, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, indice_key='subm0'), BatchNorm1d(16), nn.ReLU(), SubMConv3d(16, 16, 3, indice_key='subm0'), BatchNorm1d(16), nn.ReLU(), SpConv3d(16, 32, 3, 2, padding=1), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1'), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1'), BatchNorm1d(32), nn.ReLU(), SpConv3d(32, 64, 3, 2, padding=[0, 1, 1]), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU())
        self.max_batch_size = 6

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddle2K(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddle2K'):
        super(SpMiddle2K, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 8, 3, indice_key='subm0'), BatchNorm1d(8), nn.ReLU(), SubMConv3d(8, 8, 3, indice_key='subm0'), BatchNorm1d(8), nn.ReLU(), SpConv3d(8, 16, 3, 2, padding=1), BatchNorm1d(16), nn.ReLU(), SubMConv3d(16, 16, 3, indice_key='subm1'), BatchNorm1d(16), nn.ReLU(), SubMConv3d(16, 16, 3, indice_key='subm1'), BatchNorm1d(16), nn.ReLU(), SpConv3d(16, 32, 3, 2, padding=1), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm2'), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm2'), BatchNorm1d(32), nn.ReLU(), SpConv3d(32, 64, 3, 2, padding=1), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm4'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm4'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm4'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU())
        self.max_batch_size = 3
        self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size, self.grid)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHDLite(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddleFHDLite'):
        super(SpMiddleFHDLite, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(SpConv3d(num_input_features, 16, 3, 2, padding=1), BatchNorm1d(16), nn.ReLU(), SpConv3d(16, 32, 3, 2, padding=1), BatchNorm1d(32), nn.ReLU(), SpConv3d(32, 64, 3, 2, padding=[0, 1, 1]), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU())

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHDLiteHRZ(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddleFHDLite'):
        super(SpMiddleFHDLiteHRZ, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(SpConv3d(num_input_features, 32, 3, 2, padding=1), BatchNorm1d(32), nn.ReLU(), SpConv3d(32, 64, 3, 2, padding=1), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, 3, 2, padding=1), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU())

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHDHRZ(nn.Module):

    def __init__(self, output_shape, use_norm=True, num_input_features=128, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddleFHD'):
        super(SpMiddleFHDHRZ, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        else:
            BatchNorm1d = Empty
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        None
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, indice_key='subm0'), BatchNorm1d(16), nn.ReLU(), SubMConv3d(16, 16, 3, indice_key='subm0'), BatchNorm1d(16), nn.ReLU(), SpConv3d(16, 32, 3, 2, padding=1), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1'), BatchNorm1d(32), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1'), BatchNorm1d(32), nn.ReLU(), SpConv3d(32, 64, 3, 2, padding=1), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, 3, 2, padding=1), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm4'), BatchNorm1d(64), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm4'), BatchNorm1d(64), nn.ReLU(), SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)), BatchNorm1d(64), nn.ReLU())

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SmallObjectHead(nn.Module):

    def __init__(self, num_filters, num_class, num_anchor_per_loc, box_code_size, num_direction_bins, use_direction_classifier, encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.net = nn.Sequential(nn.Conv2d(num_filters, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        final_num_filters = 64
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        x = self.net(x)
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc, self._num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
        ret_dict = {'box_preds': box_preds.view(batch_size, -1, self._box_code_size), 'cls_preds': cls_preds.view(batch_size, -1, self._num_class)}
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict


class DefaultHead(nn.Module):

    def __init__(self, num_filters, num_class, num_anchor_per_loc, box_code_size, num_direction_bins, use_direction_classifier, encode_background_as_zeros):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        final_num_filters = num_filters
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc, self._num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
        ret_dict = {'box_preds': box_preds.view(batch_size, -1, self._box_code_size), 'cls_preds': cls_preds.view(batch_size, -1, self._num_class)}
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds.view(batch_size, -1, self._num_direction_bins)
        return ret_dict


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        if use_norm:
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
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


def register_vfe(cls, name=None):
    global REGISTERED_VFE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_VFE_CLASSES, f'exist class: {REGISTERED_VFE_CLASSES}'
    REGISTERED_VFE_CLASSES[name] = cls
    return cls


class PillarFeatureNetOld(nn.Module):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = features[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 3].unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].unsqueeze(1) * self.vy + self.y_offset)
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


class PillarFeatureNet(nn.Module):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].unsqueeze(1) * self.vx + self.x_offset)
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


class PillarFeatureNetRadius(nn.Module):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetRadius'
        assert len(num_filters) > 0
        num_input_features += 5
        num_input_features -= 1
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].unsqueeze(1) * self.vy + self.y_offset)
        features_radius = torch.norm(features[:, :, :2], p=2, dim=2, keepdim=True)
        features_radius = torch.cat([features_radius, features[:, :, 2:]], dim=2)
        features_ls = [features_radius, f_cluster, f_center]
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


class PillarFeatureNetRadiusHeight(nn.Module):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetRadiusHeight'
        assert len(num_filters) > 0
        num_input_features += 6
        num_input_features -= 1
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        pp_min = features[:, :, 2:3].min(dim=1, keepdim=True)[0]
        pp_max = features[:, :, 2:3].max(dim=1, keepdim=True)[0]
        pp_height = pp_max - pp_min
        f_height = torch.zeros_like(features[:, :, :1])
        f_height[:] = pp_height
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].unsqueeze(1) * self.vy + self.y_offset)
        features_radius = torch.norm(features[:, :, :2], p=2, dim=2, keepdim=True)
        features_radius = torch.cat([features_radius, features[:, :, 2:]], dim=2)
        features_ls = [features_radius, f_cluster, f_center, f_height]
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

    def __init__(self, output_shape, use_norm=True, num_input_features=64, num_filters_down1=[64], num_filters_down2=[64, 64], name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device)
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        return batch_canvas


class GroupNorm(torch.nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-05, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


class Sequential(torch.nn.Module):
    """A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
        
        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
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

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError('name exists')
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


def register_rpn(cls, name=None):
    global REGISTERED_RPN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_RPN_CLASSES, f'exist class: {REGISTERED_RPN_CLASSES}'
    REGISTERED_RPN_CLASSES[name] = cls
    return cls


class RPN(nn.Module):

    def __init__(self, use_norm=True, num_class=2, layer_nums=(3, 5, 5), layer_strides=(2, 2, 2), num_filters=(128, 128, 256), upsample_strides=(1, 2, 4), num_upsample_filters=(256, 256, 256), num_input_features=128, num_anchor_per_loc=2, encode_background_as_zeros=True, use_direction_classifier=True, use_groupnorm=False, num_groups=32, box_code_size=7, num_direction_bins=2, name='rpn'):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [np.round(u).astype(np.int64) for u in upsample_strides]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([(x == factors[0]) for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(num_groups=num_groups, eps=0.001)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(nn.ZeroPad2d(1), Conv2d(num_input_features, num_filters[0], 3, stride=layer_strides[0]), BatchNorm2d(num_filters[0]), nn.ReLU())
        for i in range(layer_nums[0]):
            self.block1.add(Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(ConvTranspose2d(num_filters[0], num_upsample_filters[0], upsample_strides[0], stride=upsample_strides[0]), BatchNorm2d(num_upsample_filters[0]), nn.ReLU())
        self.block2 = Sequential(nn.ZeroPad2d(1), Conv2d(block2_input_filters, num_filters[1], 3, stride=layer_strides[1]), BatchNorm2d(num_filters[1]), nn.ReLU())
        for i in range(layer_nums[1]):
            self.block2.add(Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(ConvTranspose2d(num_filters[1], num_upsample_filters[1], upsample_strides[1], stride=upsample_strides[1]), BatchNorm2d(num_upsample_filters[1]), nn.ReLU())
        self.block3 = Sequential(nn.ZeroPad2d(1), Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]), BatchNorm2d(num_filters[2]), nn.ReLU())
        for i in range(layer_nums[2]):
            self.block3.add(Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(ConvTranspose2d(num_filters[2], num_upsample_filters[2], upsample_strides[2], stride=upsample_strides[2]), BatchNorm2d(num_upsample_filters[2]), nn.ReLU())
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * num_direction_bins, 1)
        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)

    def forward(self, x):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {'box_preds': box_preds, 'cls_preds': cls_preds}
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['rc_preds'] = rc_preds
        return ret_dict


class RPNNoHeadBase(nn.Module):

    def __init__(self, use_norm=True, num_class=2, layer_nums=(3, 5, 5), layer_strides=(2, 2, 2), num_filters=(128, 128, 256), upsample_strides=(1, 2, 4), num_upsample_filters=(256, 256, 256), num_input_features=128, num_anchor_per_loc=2, encode_background_as_zeros=True, use_direction_classifier=True, use_groupnorm=False, num_groups=32, box_code_size=7, num_direction_bins=2, name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(num_groups=num_groups, eps=0.001)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []
        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(in_filters[i], num_filters[i], layer_num, stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(ConvTranspose2d(num_out_filters, num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride), BatchNorm2d(num_upsample_filters[i - self._upsample_start_idx]), nn.ReLU())
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(Conv2d(num_out_filters, num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride), BatchNorm2d(num_upsample_filters[i - self._upsample_start_idx]), nn.ReLU())
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, x):
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f'up{i}'] = up
        for i, out in enumerate(stage_outputs):
            res[f'stage{i}'] = out
        res['out'] = x
        return res


class RPNBase(RPNNoHeadBase):

    def __init__(self, use_norm=True, num_class=2, layer_nums=(3, 5, 5), layer_strides=(2, 2, 2), num_filters=(128, 128, 256), upsample_strides=(1, 2, 4), num_upsample_filters=(256, 256, 256), num_input_features=128, num_anchor_per_loc=2, encode_background_as_zeros=True, use_direction_classifier=True, use_groupnorm=False, num_groups=32, box_code_size=7, num_direction_bins=2, name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(use_norm=use_norm, num_class=num_class, layer_nums=layer_nums, layer_strides=layer_strides, num_filters=num_filters, upsample_strides=upsample_strides, num_upsample_filters=num_upsample_filters, num_input_features=num_input_features, num_anchor_per_loc=num_anchor_per_loc, encode_background_as_zeros=encode_background_as_zeros, use_direction_classifier=use_direction_classifier, use_groupnorm=use_groupnorm, num_groups=num_groups, box_code_size=box_code_size, num_direction_bins=num_direction_bins, name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(final_num_filters, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        res = super().forward(x)
        x = res['out']
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc, self._num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
        ret_dict = {'box_preds': box_preds, 'cls_preds': cls_preds}
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(0, 1, 3, 4, 2).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds
        return ret_dict


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetRPN(RPNBase):

    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(ResNetRPN, self).__init__(*args, **kw)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self.inplanes == -1:
            self.inplanes = self._num_input_features
        block = resnet.BasicBlock
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers), self.inplanes


class RPNV2(RPNBase):

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(num_groups=self._num_groups, eps=0.001)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        block = Sequential(nn.ZeroPad2d(1), Conv2d(inplanes, planes, 3, stride=stride), BatchNorm2d(planes), nn.ReLU())
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())
        return block, planes


class RPNNoHead(RPNNoHeadBase):

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(num_groups=self._num_groups, eps=0.001)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        block = Sequential(nn.ZeroPad2d(1), Conv2d(inplanes, planes, 3, stride=stride), BatchNorm2d(planes), nn.ReLU())
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())
        return block, planes


class VFELayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = F.relu(x)
        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        repeated = aggregated.repeat(1, voxel_count, 1)
        concatenated = torch.cat([pointwise, repeated], dim=2)
        return concatenated


class VoxelFeatureExtractor(nn.Module):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=[32, 128], with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1), name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)
        x *= mask
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


class VoxelFeatureExtractorV2(nn.Module):
    """VoxelFeatureExtractor with arbitrary number of VFE. deprecated.
    """

    def __init__(self, num_input_features=4, use_norm=True, num_filters=[32, 128], with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1), name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.ModuleList([VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


class SimpleVoxel(nn.Module):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=[32, 128], with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1), name='VoxelFeatureExtractor'):
        super(SimpleVoxel, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors):
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class SimpleVoxelRadius(nn.Module):
    """Simple voxel encoder. only keep r, z and reflection feature.
    """

    def __init__(self, num_input_features=4, use_norm=True, num_filters=(32, 128), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1), name='SimpleVoxelRadius'):
        super(SimpleVoxelRadius, self).__init__()
        self.num_input_features = num_input_features
        self.name = name

    def forward(self, features, num_voxels, coors):
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        feature = torch.norm(points_mean[:, :2], p=2, dim=1, keepdim=True)
        res = torch.cat([feature, points_mean[:, 2:self.num_input_features]], dim=1)
        return res


class LossNormType(Enum):
    NormByNumPositives = 'norm_by_num_positives'
    NormByNumExamples = 'norm_by_num_examples'
    NormByNumPosNeg = 'norm_by_num_pos_neg'
    DontNorm = 'dont_norm'


class Loss(object):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __call__(self, prediction_tensor, target_tensor, ignore_nan_targets=False, scope=None, **params):
        """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor), prediction_tensor, target_tensor)
        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
        pass


class WeightedSmoothL1LocalizationLoss(Loss):
    """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = torch.from_numpy(self._code_weights)
        else:
            self._code_weights = None
        self._codewise = codewise

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            code_weights = self._code_weights.type_as(prediction_tensor)
            diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / self._sigma ** 2).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + (abs_diff - 0.5 / self._sigma ** 2) * (1.0 - abs_diff_lt_1)
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


class WeightedSoftmaxClassificationLoss(Loss):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(prediction_tensor, self._logit_scale)
        per_row_cross_ent = _softmax_cross_entropy_with_logits(labels=target_tensor.view(-1, num_classes), logits=prediction_tensor.view(-1, num_classes))
        return per_row_cross_ent.view(weights.shape) * weights


def _get_pos_neg_loss(cls_loss, labels):
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor, cls_loss_ftor, box_preds, cls_preds, cls_targets, cls_weights, reg_targets, reg_weights, num_class, encode_background_as_zeros=True, encode_rad_error_by_sin=True, sin_error_factor=1.0, box_code_size=7, num_direction_bins=2):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets, box_preds[..., 6:7], reg_targets[..., 6:7], sin_error_factor)
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)
    cls_losses = cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)
    return loc_losses, cls_losses


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = box_torch_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets


def prepare_loss_weights(labels, pos_cls_weight=1.0, neg_cls_weight=1.0, loss_norm_type=LossNormType.NormByNumPositives, dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)
        cls_normalizer = (pos_neg * normalizer).sum(-1)
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise ValueError(f'unknown loss norm type. available: {list(LossNormType)}')
    return cls_weights, reg_weights, cared


def register_voxelnet(cls, name=None):
    global REGISTERED_NETWORK_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_NETWORK_CLASSES, f'exist class: {REGISTERED_NETWORK_CLASSES}'
    REGISTERED_NETWORK_CLASSES[name] = cls
    return cls


class Scalar(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))

    def forward(self, scalar):
        if not scalar.eq(0.0):
            self.count += 1
            self.total += scalar.data.float()
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Accuracy(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5, encode_background_as_zeros=True):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold
        self._encode_background_as_zeros = encode_background_as_zeros

    def forward(self, labels, preds, weights=None):
        if self._encode_background_as_zeros:
            scores = torch.sigmoid(preds)
            labels_pred = torch.max(preds, dim=self._dim)[1] + 1
            pred_labels = torch.where((scores > self._threshold).any(self._dim), labels_pred, torch.tensor(0).type_as(labels_pred))
        else:
            pred_labels = torch.max(preds, dim=self._dim)[1]
        N, *Ds = labels.shape
        labels = labels.view(N, int(np.prod(Ds)))
        pred_labels = pred_labels.view(N, int(np.prod(Ds)))
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        num_examples = torch.sum(weights)
        num_examples = torch.clamp(num_examples, min=1.0).float()
        total = torch.sum((pred_labels == labels.long()).float())
        self.count += num_examples
        self.total += total
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Precision(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold

    def forward(self, labels, preds, weights=None):
        if preds.shape[self._dim] == 1:
            pred_labels = (torch.sigmoid(preds) > self._threshold).long().squeeze(self._dim)
        else:
            assert preds.shape[self._dim] == 2, 'precision only support 2 class'
            pred_labels = torch.max(preds, dim=self._dim)[1]
        N, *Ds = labels.shape
        labels = labels.view(N, int(np.prod(Ds)))
        pred_labels = pred_labels.view(N, int(np.prod(Ds)))
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        pred_trues = pred_labels > 0
        pred_falses = pred_labels == 0
        trues = labels > 0
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).float()).sum()
        true_negatives = (weights * (falses & pred_falses).float()).sum()
        false_positives = (weights * (falses & pred_trues).float()).sum()
        false_negatives = (weights * (trues & pred_falses).float()).sum()
        count = true_positives + false_positives
        if count > 0:
            self.count += count
            self.total += true_positives
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Recall(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer('total', torch.FloatTensor([0.0]))
        self.register_buffer('count', torch.FloatTensor([0.0]))
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._threshold = threshold

    def forward(self, labels, preds, weights=None):
        if preds.shape[self._dim] == 1:
            pred_labels = (torch.sigmoid(preds) > self._threshold).long().squeeze(self._dim)
        else:
            assert preds.shape[self._dim] == 2, 'precision only support 2 class'
            pred_labels = torch.max(preds, dim=self._dim)[1]
        N, *Ds = labels.shape
        labels = labels.view(N, int(np.prod(Ds)))
        pred_labels = pred_labels.view(N, int(np.prod(Ds)))
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        pred_trues = pred_labels == 1
        pred_falses = pred_labels == 0
        trues = labels == 1
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).float()).sum()
        true_negatives = (weights * (falses & pred_falses).float()).sum()
        false_positives = (weights * (falses & pred_trues).float()).sum()
        false_negatives = (weights * (trues & pred_falses).float()).sum()
        count = true_positives + false_negatives
        if count > 0:
            self.count += count
            self.total += true_positives
        return self.value.cpu()

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


def _calc_binary_metrics(labels, scores, weights=None, ignore_idx=-1, threshold=0.5):
    pred_labels = (scores > threshold).long()
    N, *Ds = labels.shape
    labels = labels.view(N, int(np.prod(Ds)))
    pred_labels = pred_labels.view(N, int(np.prod(Ds)))
    pred_trues = pred_labels > 0
    pred_falses = pred_labels == 0
    trues = labels > 0
    falses = labels == 0
    true_positives = (weights * (trues & pred_trues).float()).sum()
    true_negatives = (weights * (falses & pred_falses).float()).sum()
    false_positives = (weights * (falses & pred_trues).float()).sum()
    false_negatives = (weights * (trues & pred_falses).float()).sum()
    return true_positives, true_negatives, false_positives, false_negatives


class PrecisionRecall(nn.Module):

    def __init__(self, dim=1, ignore_idx=-1, thresholds=0.5, use_sigmoid_score=False, encode_background_as_zeros=True):
        super().__init__()
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]
        self.register_buffer('prec_total', torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer('prec_count', torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer('rec_total', torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer('rec_count', torch.FloatTensor(len(thresholds)).zero_())
        self._ignore_idx = ignore_idx
        self._dim = dim
        self._thresholds = thresholds
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros

    def forward(self, labels, preds, weights=None):
        if self._encode_background_as_zeros:
            assert self._use_sigmoid_score is True
            total_scores = torch.sigmoid(preds)
        elif self._use_sigmoid_score:
            total_scores = torch.sigmoid(preds)[..., 1:]
        else:
            total_scores = F.softmax(preds, dim=-1)[..., 1:]
        """
        if preds.shape[self._dim] == 1:  # BCE
            scores = torch.sigmoid(preds)
        else:
            # assert preds.shape[
            #     self._dim] == 2, "precision only support 2 class"
            # TODO: add support for [N, C, ...] format.
            # TODO: add multiclass support
            if self._use_sigmoid_score:
                scores = torch.sigmoid(preds)[:, ..., 1:].sum(-1)
            else:
                scores = F.softmax(preds, dim=self._dim)[:, ..., 1:].sum(-1)
        """
        scores = torch.max(total_scores, dim=-1)[0]
        if weights is None:
            weights = (labels != self._ignore_idx).float()
        else:
            weights = weights.float()
        for i, thresh in enumerate(self._thresholds):
            tp, tn, fp, fn = _calc_binary_metrics(labels, scores, weights, self._ignore_idx, thresh)
            rec_count = tp + fn
            prec_count = tp + fp
            if rec_count > 0:
                self.rec_count[i] += rec_count
                self.rec_total[i] += tp
            if prec_count > 0:
                self.prec_count[i] += prec_count
                self.prec_total[i] += tp
        return self.value

    @property
    def value(self):
        prec_count = torch.clamp(self.prec_count, min=1.0)
        rec_count = torch.clamp(self.rec_count, min=1.0)
        return (self.prec_total / prec_count).cpu(), (self.rec_total / rec_count).cpu()

    @property
    def thresholds(self):
        return self._thresholds

    def clear(self):
        self.rec_count.zero_()
        self.prec_count.zero_()
        self.prec_total.zero_()
        self.rec_total.zero_()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DefaultHead,
     lambda: ([], {'num_filters': 4, 'num_class': 4, 'num_anchor_per_loc': 4, 'box_code_size': 4, 'num_direction_bins': 4, 'use_direction_classifier': 4, 'encode_background_as_zeros': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Empty,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (GroupNorm,
     lambda: ([], {'num_channels': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PFNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ResNetRPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     False),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleVoxel,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SimpleVoxelRadius,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SmallObjectHead,
     lambda: ([], {'num_filters': 4, 'num_class': 4, 'num_anchor_per_loc': 4, 'box_code_size': 4, 'num_direction_bins': 4, 'use_direction_classifier': 4, 'encode_background_as_zeros': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VFELayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_traveller59_second_pytorch(_paritybench_base):
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

