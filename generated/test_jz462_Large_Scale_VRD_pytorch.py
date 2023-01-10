import sys
_module = sys.modules[__name__]
del sys
core = _module
config = _module
test = _module
test_engine = _module
test_engine_rel = _module
test_rel = _module
datasets = _module
dataset_catalog = _module
dataset_catalog_rel = _module
json_dataset = _module
json_dataset_evaluator = _module
json_dataset_rel = _module
roidb = _module
roidb_rel = _module
task_evaluation = _module
task_evaluation_rel = _module
voc_eval_rel = _module
model = _module
nms = _module
_ext = _module
nms = _module
build = _module
nms_gpu = _module
nms_wrapper = _module
roi_align = _module
roi_align = _module
build = _module
functions = _module
roi_align = _module
modules = _module
roi_align = _module
roi_crop = _module
crop_resize = _module
roi_crop = _module
build = _module
crop_resize = _module
gridgen = _module
roi_crop = _module
gridgen = _module
roi_crop = _module
roi_pooling = _module
roi_pooling = _module
build = _module
roi_pool = _module
roi_pool = _module
utils = _module
net_utils = _module
FPN = _module
ResNet = _module
VGG16 = _module
modeling = _module
collect_and_distribute_fpn_rpn_proposals = _module
fast_rcnn_heads = _module
generate_anchors = _module
generate_proposal_labels = _module
generate_proposals = _module
generate_rel_proposal_labels = _module
get_dataset_counts_rel = _module
model_builder = _module
model_builder_rel = _module
reldn_heads = _module
relpn_heads = _module
roi_xfrom = _module
roi_align = _module
build = _module
roi_align = _module
roi_align = _module
rpn_heads = _module
sparse_targets_rel = _module
nn = _module
functional = _module
init = _module
affine = _module
normalization = _module
upsample = _module
parallel = _module
_functions = _module
data_parallel = _module
parallel_apply = _module
replicate = _module
scatter_gather = _module
roi_data = _module
data_utils = _module
fast_rcnn = _module
fast_rcnn_rel = _module
loader = _module
loader_rel = _module
minibatch = _module
minibatch_rel = _module
rpn = _module
setup = _module
blob = _module
boxes = _module
collections = _module
colormap = _module
detectron_weight_helper = _module
env = _module
fpn = _module
image = _module
io = _module
logging = _module
logging_rel = _module
misc = _module
net = _module
resnet_weights_helper = _module
subprocess = _module
timer = _module
training_stats = _module
training_stats_rel = _module
_init_paths = _module
test_net = _module
test_net_rel = _module
train_net_step = _module
train_net_step_rel = _module

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


import copy


import numpy as np


import torch


import torch.nn as nn


from torch.nn import init


from collections import defaultdict


from torch.autograd import Variable


import logging


from numpy import linalg as la


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import torch.nn.functional as F


import torchvision.models as models


import random


import collections


from collections import OrderedDict


from torch import nn


import torch.nn.init as init


from functools import wraps


import math


from functools import reduce


import torch.cuda.comm as comm


from torch.nn import Module


import re


from torch._six import string_classes


import numpy.random as npr


import torch.utils.data as data


import torch.utils.data.sampler as torch_sampler


from torch.utils.data.dataloader import default_collate


from collections import Iterable


from copy import deepcopy


from itertools import chain


import time


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, self.sampling_ratio, features, rois, output)
        else:
            raise NotImplementedError
        return output

    def backward(self, grad_output):
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, self.sampling_ratio, grad_output, self.rois, grad_input)
        return grad_input, None


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width, self.spatial_scale, self.sampling_ratio)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale, self.sampling_ratio)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale, self.sampling_ratio)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


class AffineGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = input1.new(torch.Size([input1.size(0)]) + self.grid.size()).zero_()
        self.batchgrid = input1.new(torch.Size([input1.size(0)]) + self.grid.size()).zero_()
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid.astype(self.batchgrid[i])
        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height * self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)
        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height * self.width, 2), 1, 2), self.batchgrid.view(-1, self.height * self.width, 3))
        return grad_input1


class _AffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(_AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr

    def forward(self, input):
        return self.f(input)


class AffineGridGenV2(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(AffineGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
        output = torch.bmm(self.batchgrid.view(-1, self.height * self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)
        return output


class CylinderGridGenV2(Module):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.grid.size())
        for i in range(input.size(0)):
            self.batchgrid[i, :, :, :] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        input_u = input.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output0 = self.batchgrid[:, :, :, 0:1]
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (self.batchgrid[:, :, :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.mul(self.batchgrid, input1[:, :, :, 0:3])
        y = torch.mul(self.batchgrid, input1[:, :, :, 3:6])
        output = torch.cat([torch.sum(x, 3), torch.sum(y, 3)], 3)
        return output


class DenseAffine3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, 0] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, 1] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, 0] = self.x
        self.grid3d[:, :, 1] = self.y
        self.grid3d[:, :, 2] = self.z
        self.grid3d[:, :, 3] = self.grid[:, :, 2]

    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class DenseAffine3DGridGen_rotate(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffine3DGridGen_rotate, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, 0] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, 1] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, 0] = self.x
        self.grid3d[:, :, 1] = self.y
        self.grid3d[:, :, 2] = self.z
        self.grid3d[:, :, 3] = self.grid[:, :, 2]

    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        input_u = input2.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, 0] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, 1] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, 0] = self.x
        self.grid3d[:, :, 1] = self.y
        self.grid3d[:, :, 2] = self.z
        self.grid3d[:, :, 3] = self.grid[:, :, 2]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        y = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        input_u = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen_with_mask(Module):

    def __init__(self, height, width, lr=1, aux_loss=False, ray_tracing=False):
        super(Depth3DGridGen_with_mask, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.ray_tracing = ray_tracing
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, 0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=0).T, 0)
        self.grid[:, :, 1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=0), 0)
        self.grid[:, :, 2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, 0] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, 1] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4], dtype=np.float32))
        self.grid3d[:, :, 0] = self.x
        self.grid3d[:, :, 1] = self.y
        self.grid3d[:, :, 2] = self.z
        self.grid3d[:, :, 3] = self.grid[:, :, 2]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if depth.is_cuda:
            self.batchgrid = self.batchgrid
            self.batchgrid3d = self.batchgrid3d
        x_ = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        y_ = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        rotate_z = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1) * np.pi
        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        if depth.is_cuda:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        else:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


class RoICropFunction(Function):

    def forward(self, input1, input2):
        self.input1 = input1.clone()
        self.input2 = input2.clone()
        output = input2.new(input2.size()[0], input1.size()[1], input2.size()[1], input2.size()[2]).zero_()
        assert output.get_device() == input1.get_device(), 'output and input1 must on the same device'
        assert output.get_device() == input2.get_device(), 'output and input2 must on the same device'
        roi_crop.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input2 = self.input2.new(self.input2.size()).zero_()
        roi_crop.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        return grad_input1, grad_input2


class _RoICrop(Module):

    def __init__(self, layout='BHWD'):
        super(_RoICrop, self).__init__()

    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)


class RoIPoolFunction(Function):

    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, features, rois, output, ctx.argmax)
        return output

    def backward(ctx, grad_output):
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, grad_output, ctx.rois, grad_input, ctx.argmax)
        return grad_input, None


class _RoIPooling(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features, rois)


HIGHEST_BACKBONE_LVL = 5


LOWEST_BACKBONE_LVL = 2


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = LOWEST_BACKBONE_LVL
    max_level = HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level


class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""

    def __init__(self, dim_in_top, dim_in_lateral):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        if cfg.FPN.USE_GN:
            self.conv_lateral = nn.Sequential(nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False), nn.GroupNorm(net_utils.get_group_gn(self.dim_out), self.dim_out, eps=cfg.GROUP_NORM.EPSILON))
        else:
            self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)
        self._init_weights()

    def _init_weights(self):
        if cfg.FPN.USE_GN:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral
        if cfg.FPN.ZERO_INIT_LATERAL:
            init.constant_(conv.weight, 0)
        else:
            mynn.init.XavierFill(conv.weight)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        lat = self.conv_lateral(lateral_blob)
        td = F.upsample(top_blob, scale_factor=2, mode='nearest')
        return lat + td


class fpn(nn.Module):
    """Add FPN connections based on the model described in the FPN paper.

    fpn_output_blobs is in reversed order: e.g [fpn5, fpn4, fpn3, fpn2]
    similarly for fpn_level_info.dims: e.g [2048, 1024, 512, 256]
    similarly for spatial_scale: e.g [1/32, 1/16, 1/8, 1/4]
    """

    def __init__(self, conv_body_func, fpn_level_info, P2only=False):
        super().__init__()
        self.fpn_level_info = fpn_level_info
        self.P2only = P2only
        self.dim_out = fpn_dim = cfg.FPN.DIM
        min_level, max_level = get_min_max_levels()
        self.num_backbone_stages = len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
        fpn_dim_lateral = fpn_level_info.dims
        self.spatial_scale = []
        self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        if cfg.FPN.USE_GN:
            self.conv_top = nn.Sequential(nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0, bias=False), nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim, eps=cfg.GROUP_NORM.EPSILON))
        else:
            self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        self.topdown_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()
        for i in range(self.num_backbone_stages - 1):
            self.topdown_lateral_modules.append(topdown_lateral_module(fpn_dim, fpn_dim_lateral[i + 1]))
        for i in range(self.num_backbone_stages):
            if cfg.FPN.USE_GN:
                self.posthoc_modules.append(nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False), nn.GroupNorm(net_utils.get_group_gn(fpn_dim), fpn_dim, eps=cfg.GROUP_NORM.EPSILON)))
            else:
                self.posthoc_modules.append(nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1))
            self.spatial_scale.append(fpn_level_info.spatial_scales[i])
        if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_BACKBONE_LVL + 1:
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)
        if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
            self.extra_pyramid_modules = nn.ModuleList()
            dim_in = fpn_level_info.dims[0]
            for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
                self.extra_pyramid_modules(nn.Conv2d(dim_in, fpn_dim, 3, 2, 1))
                dim_in = fpn_dim
                self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)
        if self.P2only:
            self.spatial_scale = self.spatial_scale[-1]
        self._init_weights()
        self.conv_body = conv_body_func()

    def _init_weights(self):

        def init_func(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for child_m in self.children():
            if not isinstance(child_m, nn.ModuleList) or not isinstance(child_m[0], topdown_lateral_module):
                child_m.apply(init_func)

    def detectron_weight_mapping(self):
        conv_body_mapping, orphan_in_detectron = self.conv_body.detectron_weight_mapping()
        mapping_to_detectron = {}
        for key, value in conv_body_mapping.items():
            mapping_to_detectron['conv_body.' + key] = value
        d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[0]
        if cfg.FPN.USE_GN:
            mapping_to_detectron['conv_top.0.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.1.weight'] = d_prefix + '_gn_s'
            mapping_to_detectron['conv_top.1.bias'] = d_prefix + '_gn_b'
        else:
            mapping_to_detectron['conv_top.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.bias'] = d_prefix + '_b'
        for i in range(self.num_backbone_stages - 1):
            p_prefix = 'topdown_lateral_modules.%d.conv_lateral' % i
            d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[i + 1] + '_lateral'
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({(p_prefix + '.0.weight'): d_prefix + '_w', (p_prefix + '.1.weight'): d_prefix + '_gn_s', (p_prefix + '.1.bias'): d_prefix + '_gn_b'})
            else:
                mapping_to_detectron.update({(p_prefix + '.weight'): d_prefix + '_w', (p_prefix + '.bias'): d_prefix + '_b'})
        for i in range(self.num_backbone_stages):
            p_prefix = 'posthoc_modules.%d' % i
            d_prefix = 'fpn_' + self.fpn_level_info.blobs[i]
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({(p_prefix + '.0.weight'): d_prefix + '_w', (p_prefix + '.1.weight'): d_prefix + '_gn_s', (p_prefix + '.1.bias'): d_prefix + '_gn_b'})
            else:
                mapping_to_detectron.update({(p_prefix + '.weight'): d_prefix + '_w', (p_prefix + '.bias'): d_prefix + '_b'})
        if hasattr(self, 'extra_pyramid_modules'):
            for i in len(self.extra_pyramid_modules):
                p_prefix = 'extra_pyramid_modules.%d' % i
                d_prefix = 'fpn_%d' % (HIGHEST_BACKBONE_LVL + 1 + i)
                mapping_to_detectron.update({(p_prefix + '.weight'): d_prefix + '_w', (p_prefix + '.bias'): d_prefix + '_b'})
        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        conv_body_blobs = [self.conv_body.res1(x)]
        for i in range(1, self.conv_body.convX):
            conv_body_blobs.append(getattr(self.conv_body, 'res%d' % (i + 1))(conv_body_blobs[-1]))
        fpn_inner_blobs = [self.conv_top(conv_body_blobs[-1])]
        for i in range(self.num_backbone_stages - 1):
            fpn_inner_blobs.append(self.topdown_lateral_modules[i](fpn_inner_blobs[-1], conv_body_blobs[-(i + 2)]))
        fpn_output_blobs = []
        for i in range(self.num_backbone_stages):
            fpn_output_blobs.append(self.posthoc_modules[i](fpn_inner_blobs[i]))
        if hasattr(self, 'maxpool_p6'):
            fpn_output_blobs.insert(0, self.maxpool_p6(fpn_output_blobs[0]))
        if hasattr(self, 'extra_pyramid_modules'):
            blob_in = conv_body_blobs[-1]
            fpn_output_blobs.insert(0, self.extra_pyramid_modules(blob_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.insert(0, module(F.relu(fpn_output_blobs[0], inplace=True)))
        if self.P2only:
            return fpn_output_blobs[-1]
        else:
            return fpn_output_blobs


def collect(inputs, is_training):
    cfg_key = 'TRAIN' if is_training else 'TEST'
    post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls:]
    rois = np.concatenate(roi_inputs)
    scores = np.concatenate(score_inputs).squeeze()
    inds = np.argsort(-scores)[:post_nms_topN]
    rois = rois[inds, :]
    return rois


def distribute(rois, label_blobs):
    """To understand the output blob order see return value of
    roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
    output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    outputs = [None] * len(output_blob_names)
    outputs[0] = rois
    rois_idx_order = np.empty((0,))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        outputs[output_idx + 1] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    outputs[-1] = rois_idx_restore.astype(np.int32)
    return dict(zip(output_blob_names, outputs))


class CollectAndDistributeFpnRpnProposalsOp(nn.Module):
    """Merge RPN proposals generated at multiple FPN levels and then
    distribute those proposals to their appropriate FPN levels. An anchor
    at one FPN level may predict an RoI that will map to another level,
    hence the need to redistribute the proposals.

    This function assumes standard blob names for input and output blobs.

    Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                  rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
        - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
        documentation from GenerateProposals.
        - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
        level i; see rpn_roi_probs documentation from GenerateProposals.

    If used during training, then the input blobs will also include:
        [roidb, im_info] (see GenerateProposalLabels).

    Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                   rois_idx_restore]
        - rois_fpn<i> are the RPN proposals for FPN level i
        - rois_idx_restore is a permutation on the concatenation of all
        rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
        restored to their original order in the input blobs.

    If used during training, then the output blobs will also include:
        [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, roidb, im_info):
        """
        Args:
            inputs: a list of [rpn_rois_fpn2, ..., rpn_rois_fpn6,
                               rpn_roi_probs_fpn2, ..., rpn_roi_probs_fpn6]
            im_info: [[im_height, im_width, im_scale], ...]
        """
        rois = collect(inputs, self.training)
        if self.training:
            im_scales = im_info.data.numpy()[:, 2]
            json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)
            output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names()
            blobs = {k: [] for k in output_blob_names}
            roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
        else:
            blobs = distribute(rois, None)
        return blobs


def _filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image.
  """
    min_size *= im_info[2]
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ws / 2.0
    y_ctr = boxes[:, 1] + hs / 2.0
    keep = np.where((ws >= min_size) & (hs >= min_size) & (x_ctr < im_info[1]) & (y_ctr < im_info[0]))[0]
    return keep


class GenerateProposalsOp(nn.Module):

    def __init__(self, anchors, spatial_scale):
        super().__init__()
        self._anchors = anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1.0 / spatial_scale

    def forward(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        """Type conversion"""
        scores = rpn_cls_prob.data.cpu().numpy()
        bbox_deltas = rpn_bbox_pred.data.cpu().numpy()
        im_info = im_info.data.cpu().numpy()
        height, width = scores.shape[-2:]
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        num_images = scores.shape[0]
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 4))
        rois = np.empty((0, 5), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :], scores[im_i, :, :, :])
            batch_inds = im_i * np.ones((im_i_boxes.shape[0], 1), dtype=np.float32)
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            rois = np.append(rois, im_i_rois, axis=0)
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)
        return rois, roi_probs

    def proposals_for_one_image(self, im_info, all_anchors, bbox_deltas, scores):
        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))
        scores = scores.transpose((1, 2, 0)).reshape((-1, 1))
        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            inds = np.argpartition(-scores.squeeze(), pre_nms_topN)[:pre_nms_topN]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]
        proposals = box_utils.bbox_transform(all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0))
        proposals = box_utils.clip_tiled_boxes(proposals, im_info[:2])
        keep = _filter_boxes(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]
        if nms_thresh > 0:
            keep = box_utils.nms(np.hstack((proposals, scores)), nms_thresh)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]
        return proposals, scores


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
    return anchors


def generate_anchors(stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(stride, np.array(sizes, dtype=np.float) / stride, np.array(aspect_ratios, dtype=np.float))


class fpn_rpn_outputs(nn.Module):
    """Add RPN on FPN specific outputs."""

    def __init__(self, dim_in, spatial_scales):
        super().__init__()
        self.dim_in = dim_in
        self.spatial_scales = spatial_scales
        self.dim_out = self.dim_in
        num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)
        self.FPN_RPN_conv = nn.Conv2d(dim_in, self.dim_out, 3, 1, 1)
        dim_score = num_anchors * 2 if cfg.RPN.CLS_ACTIVATION == 'softmax' else num_anchors
        self.FPN_RPN_cls_score = nn.Conv2d(self.dim_out, dim_score, 1, 1, 0)
        self.FPN_RPN_bbox_pred = nn.Conv2d(self.dim_out, 4 * num_anchors, 1, 1, 0)
        self.GenerateProposals_modules = nn.ModuleList()
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        for lvl in range(k_min, k_max + 1):
            sc = self.spatial_scales[k_max - lvl]
            lvl_anchors = generate_anchors(stride=2.0 ** lvl, sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2.0 ** (lvl - k_min),), aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS)
            self.GenerateProposals_modules.append(GenerateProposalsOp(lvl_anchors, sc))
        self.CollectAndDistributeFpnRpnProposals = CollectAndDistributeFpnRpnProposalsOp()
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.FPN_RPN_conv.weight, std=0.01)
        init.constant_(self.FPN_RPN_conv.bias, 0)
        init.normal_(self.FPN_RPN_cls_score.weight, std=0.01)
        init.constant_(self.FPN_RPN_cls_score.bias, 0)
        init.normal_(self.FPN_RPN_bbox_pred.weight, std=0.01)
        init.constant_(self.FPN_RPN_bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        k_min = cfg.FPN.RPN_MIN_LEVEL
        mapping_to_detectron = {'FPN_RPN_conv.weight': 'conv_rpn_fpn%d_w' % k_min, 'FPN_RPN_conv.bias': 'conv_rpn_fpn%d_b' % k_min, 'FPN_RPN_cls_score.weight': 'rpn_cls_logits_fpn%d_w' % k_min, 'FPN_RPN_cls_score.bias': 'rpn_cls_logits_fpn%d_b' % k_min, 'FPN_RPN_bbox_pred.weight': 'rpn_bbox_pred_fpn%d_w' % k_min, 'FPN_RPN_bbox_pred.bias': 'rpn_bbox_pred_fpn%d_b' % k_min}
        return mapping_to_detectron, []

    def forward(self, blobs_in, im_info, roidb=None):
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        assert len(blobs_in) == k_max - k_min + 1
        return_dict = {}
        rois_blobs = []
        score_blobs = []
        for lvl in range(k_min, k_max + 1):
            slvl = str(lvl)
            bl_in = blobs_in[k_max - lvl]
            fpn_rpn_conv = F.relu(self.FPN_RPN_conv(bl_in), inplace=True)
            fpn_rpn_cls_score = self.FPN_RPN_cls_score(fpn_rpn_conv)
            fpn_rpn_bbox_pred = self.FPN_RPN_bbox_pred(fpn_rpn_conv)
            return_dict['rpn_cls_logits_fpn' + slvl] = fpn_rpn_cls_score
            return_dict['rpn_bbox_pred_fpn' + slvl] = fpn_rpn_bbox_pred
            if not self.training or cfg.MODEL.FASTER_RCNN:
                if cfg.RPN.CLS_ACTIVATION == 'softmax':
                    B, C, H, W = fpn_rpn_cls_score.size()
                    fpn_rpn_cls_probs = F.softmax(fpn_rpn_cls_score.view(B, 2, C // 2, H, W), dim=1)
                    fpn_rpn_cls_probs = fpn_rpn_cls_probs[:, 1].squeeze(dim=1)
                else:
                    fpn_rpn_cls_probs = F.sigmoid(fpn_rpn_cls_score)
                fpn_rpn_rois, fpn_rpn_roi_probs = self.GenerateProposals_modules[lvl - k_min](fpn_rpn_cls_probs, fpn_rpn_bbox_pred, im_info)
                rois_blobs.append(fpn_rpn_rois)
                score_blobs.append(fpn_rpn_roi_probs)
                return_dict['rpn_rois_fpn' + slvl] = fpn_rpn_rois
                return_dict['rpn_rois_prob_fpn' + slvl] = fpn_rpn_roi_probs
        if cfg.MODEL.FASTER_RCNN:
            blobs_out = self.CollectAndDistributeFpnRpnProposals(rois_blobs + score_blobs, roidb, im_info)
            return_dict.update(blobs_out)
        return return_dict


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None
    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(inplanes, outplanes, innerplanes, stride, dilation=dilation, group=cfg.RESNETS.NUM_GROUPS, downsample=downsample)
    return res_block


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(inplanes, outplanes, innerplanes, dilation, stride))
        inplanes = outplanes
        stride = 1
    return nn.Sequential(*res_blocks), outplanes


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.RESNETS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'
            mapping_to_detectron[my_prefix + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
    return mapping_to_detectron, orphan_in_detectron


class ResNet_convX_body(nn.Module):

    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2
        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0], dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1], dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2], dilation=1, stride_init=2)
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3], cfg.RESNETS.RES5_DILATION, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16
        self.dim_out = dim_in
        self._init_modules()

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {'res1.conv1.weight': 'conv1_w', 'res1.gn1.weight': 'conv1_gn_s', 'res1.gn1.bias': 'conv1_gn_b'}
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {'res1.conv1.weight': 'conv1_w', 'res1.bn1.weight': 'res_conv1_bn_s', 'res1.bn1.bias': 'res_conv1_bn_b'}
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']
        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(getattr(self, stage_name), stage_name, self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)
        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        self.training = mode
        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


class ResNet_roi_conv5_head(nn.Module):

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3, dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)
        self._init_modules()

    def _init_modules(self):
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(x, rpn_ret, blob_rois='rois', method=cfg.FAST_RCNN.ROI_XFORM_METHOD, resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION, spatial_scale=self.spatial_scale, sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x


class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()
        str1x1, str3x3 = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = mynn.AffineChannel2d(innerplanes)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False, padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = mynn.AffineChannel2d(innerplanes)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = mynn.AffineChannel2d(outplanes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

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


class bottleneck_gn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1, downsample=None):
        super().__init__()
        str1x1, str3x3 = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv2 = nn.Conv2d(innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False, padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(net_utils.get_group_gn(innerplanes), innerplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.conv3 = nn.Conv2d(innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(net_utils.get_group_gn(outplanes), outplanes, eps=cfg.GROUP_NORM.EPSILON)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.gn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


vgg = models.vgg16()


class VGG16_conv_body(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_layers = 16
        self.spatial_scale = 1.0 / 16.0
        self.dim_out = 512
        self._init_modules()

    def _init_modules(self):
        self.convs = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        for layer in range(10):
            for p in self.convs[layer].parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.convs(x)


class VGG16_roi_conv5_head(nn.Module):

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = 4096
        self.dim_roi_out = dim_in
        self._init_modules()

    def _init_modules(self):
        self.heads = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    def forward(self, x, rpn_ret, rois_name='rois', use_relu=True):
        x = self.roi_xform(x, rpn_ret, blob_rois=rois_name, method=cfg.FAST_RCNN.ROI_XFORM_METHOD, resolution=7, spatial_scale=self.spatial_scale, sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)
        feat = x.view(x.size(0), -1)
        if use_relu:
            for layer in list(self.heads.children()):
                feat = layer(feat)
        else:
            for layer in list(self.heads.children())[:-2]:
                feat = layer(feat)
        return feat


class fast_rcnn_outputs(nn.Module):

    def __init__(self, dim_in):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {'cls_score.weight': 'cls_score_w', 'cls_score.bias': 'cls_score_b', 'bbox_pred.weight': 'bbox_pred_w', 'bbox_pred.bias': 'bbox_pred_b'}
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred


class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {'fc1.weight': 'fc6_w', 'fc1.bias': 'fc6_b', 'fc2.weight': 'fc7_w', 'fc2.bias': 'fc7_b'}
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret, rois_name='rois', use_relu=True):
        x = self.roi_xform(x, rpn_ret, blob_rois=rois_name, method=cfg.FAST_RCNN.ROI_XFORM_METHOD, resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION, spatial_scale=self.spatial_scale, sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        if use_relu:
            x = F.relu(self.fc2(x), inplace=True)
        else:
            x = self.fc2(x)
        return x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([nn.Conv2d(dim_in, hidden_dim, 3, 1, 1), nn.ReLU(inplace=True)])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)
        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)
        self._init_weights()

    def _init_weights(self):

        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({('convs.%d.weight' % (i * 2)): 'head_conv%d_w' % (i + 1), ('convs.%d.bias' % (i * 2)): 'head_conv%d_b' % (i + 1)})
        mapping.update({'fc.weight': 'fc6_w', 'fc.bias': 'fc6_b'})
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(x, rpn_ret, blob_rois='rois', method=cfg.FAST_RCNN.ROI_XFORM_METHOD, resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION, spatial_scale=self.spatial_scale, sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False), nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim, eps=cfg.GROUP_NORM.EPSILON), nn.ReLU(inplace=True)])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)
        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)
        self._init_weights()

    def _init_weights(self):

        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({('convs.%d.weight' % (i * 3)): 'head_conv%d_w' % (i + 1), ('convs.%d.weight' % (i * 3 + 1)): 'head_conv%d_gn_s' % (i + 1), ('convs.%d.bias' % (i * 3 + 1)): 'head_conv%d_gn_b' % (i + 1)})
        mapping.update({'fc.weight': 'fc6_w', 'fc.bias': 'fc6_b'})
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(x, rpn_ret, blob_rois='rois', method=cfg.FAST_RCNN.ROI_XFORM_METHOD, resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION, spatial_scale=self.spatial_scale, sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO)
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class GenerateProposalLabelsOp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, rpn_rois, roidb, im_info):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        im_scales = im_info.data.numpy()[:, 2]
        output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names()
        json_dataset.add_proposals(roidb, rpn_rois, im_scales, crowd_thresh=0)
        blobs = {k: [] for k in output_blob_names}
        roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
        return blobs


class GenerateRelProposalLabelsOp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sbj_rois, obj_rois, det_rois, roidb, im_info):
        im_scales = im_info.data.numpy()[:, 2]
        json_dataset_rel.add_rel_proposals(roidb, sbj_rois, obj_rois, det_rois, im_scales)
        output_blob_names = ['sbj_rois', 'obj_rois', 'rel_rois', 'fg_prd_labels_int32', 'all_prd_labels_int32', 'fg_size']
        if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.USE_SEPARATE_SO_SCORES:
            output_blob_names += ['all_sbj_labels_int32']
            output_blob_names += ['all_obj_labels_int32']
        blobs = {k: [] for k in output_blob_names}
        roi_data.fast_rcnn_rel.add_rel_blobs(blobs, im_scales, roidb)
        return blobs


def check_inference(net_func):

    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.Set the network in inference mode by net.eval().')
    return wrapper


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger


logger = setup_logging(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        if len(parts) == 1:
            return globals()[parts[0]]
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def get_obj_prd_vecs(dataset_name):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(cfg.DATA_DIR + '/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
    logger.info('Model loaded.')
    all_keys = list(word2vec_model.vocab.keys())
    for key in all_keys:
        new_key = key.lower()
        word2vec_model.vocab[new_key] = word2vec_model.vocab.pop(key)
    logger.info('Wiki words converted to lowercase.')
    if dataset_name.find('vrd') >= 0:
        with open(cfg.DATA_DIR + '/vrd/objects.json') as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vrd/predicates.json') as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg') >= 0:
        with open(cfg.DATA_DIR + '/vg/objects.json') as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg/predicates.json') as f:
            prd_cats = json.load(f)
    else:
        raise NotImplementedError
    prd_cats.insert(0, 'unknown')
    all_obj_vecs = np.zeros((len(obj_cats), 300), dtype=np.float32)
    for r, obj_cat in enumerate(obj_cats):
        obj_words = obj_cat.split()
        for word in obj_words:
            raw_vec = word2vec_model[word]
            all_obj_vecs[r] += raw_vec / la.norm(raw_vec)
        all_obj_vecs[r] /= len(obj_words)
    logger.info('Object label vectors loaded.')
    all_prd_vecs = np.zeros((len(prd_cats), 300), dtype=np.float32)
    for r, prd_cat in enumerate(prd_cats):
        prd_words = prd_cat.split()
        for word in prd_words:
            raw_vec = word2vec_model[word]
            all_prd_vecs[r] += raw_vec / la.norm(raw_vec)
        all_prd_vecs[r] /= len(prd_words)
    logger.info('Predicate label vectors loaded.')
    return all_obj_vecs, all_prd_vecs


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]
    overlaps = box_utils.bbox_overlaps(boxes.astype(np.float32), boxes.astype(np.float32)) > 0
    np.fill_diagonal(overlaps, 0)
    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)
    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))
        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def get_rel_counts(ds_name, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data: 
    :param must_overlap: 
    :return: 
    """
    if ds_name.find('vg') >= 0:
        with open(cfg.DATA_DIR + '/vg/rel_annotations_train.json') as f:
            train_data = json.load(f)
    elif ds_name.find('vrd') >= 0:
        with open(cfg.DATA_DIR + '/vrd/new_annotations_train.json') as f:
            train_data = json.load(f)
    else:
        raise NotImplementedError
    fg_matrix = np.zeros((cfg.MODEL.NUM_CLASSES - 1, cfg.MODEL.NUM_CLASSES - 1, cfg.MODEL.NUM_PRD_CLASSES + 1), dtype=np.int64)
    bg_matrix = np.zeros((cfg.MODEL.NUM_CLASSES - 1, cfg.MODEL.NUM_CLASSES - 1), dtype=np.int64)
    for _, im_rels in train_data.items():
        gt_box_to_label = {}
        for i, rel in enumerate(im_rels):
            sbj_box = box_utils.y1y2x1x2_to_x1y1x2y2(rel['subject']['bbox'])
            obj_box = box_utils.y1y2x1x2_to_x1y1x2y2(rel['object']['bbox'])
            sbj_lbl = rel['subject']['category']
            obj_lbl = rel['object']['category']
            prd_lbl = rel['predicate']
            if tuple(sbj_box) not in gt_box_to_label:
                gt_box_to_label[tuple(sbj_box)] = sbj_lbl
            if tuple(obj_box) not in gt_box_to_label:
                gt_box_to_label[tuple(obj_box)] = obj_lbl
            fg_matrix[sbj_lbl, obj_lbl, prd_lbl + 1] += 1
        if cfg.MODEL.USE_OVLP_FILTER:
            if len(gt_box_to_label):
                gt_boxes = np.array(list(gt_box_to_label.keys()), dtype=np.int32)
                gt_classes = np.array(list(gt_box_to_label.values()), dtype=np.int32)
                o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
                for o1, o2 in o1o2_total:
                    bg_matrix[o1, o2] += 1
        else:
            for b1, l1 in gt_box_to_label.items():
                for b2, l2 in gt_box_to_label.items():
                    if b1 == b2:
                        continue
                    bg_matrix[l1, l2] += 1
    return fg_matrix, bg_matrix


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, ds_name, eps=0.001):
        super(FrequencyBias, self).__init__()
        if ds_name.find('vg') >= 0:
            ds_name = 'vg'
        elif ds_name.find('vrd') >= 0:
            ds_name = 'vrd'
        else:
            raise NotImplementedError
        if cfg.MODEL.USE_OVLP_FILTER:
            must_overlap = True
        else:
            must_overlap = False
        fg_matrix, bg_matrix = get_rel_counts(ds_name, must_overlap=must_overlap)
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + 1e-08) + eps)
        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])
        self.rel_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.rel_baseline.weight.data = pred_dist
        logger.info('Frequency bias tables loaded.')

    def rel_index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.rel_baseline(labels[:, 0] * self.num_objs + labels[:, 1])


class reldn_head(nn.Module):

    def __init__(self, dim_in, all_obj_vecs=None, all_prd_vecs=None):
        super().__init__()
        num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1
        if cfg.MODEL.RUN_BASELINE:
            self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
            return
        self.obj_vecs = all_obj_vecs
        self.prd_vecs = all_prd_vecs
        self.prd_feats = nn.Sequential(nn.Linear(dim_in, 1024), nn.LeakyReLU(0.1))
        self.prd_vis_embeddings = nn.Sequential(nn.Linear(1024 * 3, 1024), nn.LeakyReLU(0.1), nn.Linear(1024, 1024))
        if not cfg.MODEL.USE_SEM_CONCAT:
            self.prd_sem_embeddings = nn.Sequential(nn.Linear(300, 1024), nn.LeakyReLU(0.1), nn.Linear(1024, 1024))
        else:
            self.prd_sem_hidden = nn.Sequential(nn.Linear(300, 1024), nn.LeakyReLU(0.1), nn.Linear(1024, 1024))
            self.prd_sem_embeddings = nn.Linear(3 * 1024, 1024)
        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
        self.so_sem_embeddings = nn.Sequential(nn.Linear(300, 1024), nn.LeakyReLU(0.1), nn.Linear(1024, 1024))
        if cfg.MODEL.USE_FREQ_BIAS:
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, spo_feat, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None):
        device_id = spo_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64')))
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64')))
        if cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
            return prd_cls_scores, None, None, None, None, None
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)
        sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
        obj_vis_embeddings = self.so_vis_embeddings(obj_feat)
        prd_hidden = self.prd_feats(spo_feat)
        prd_features = torch.cat((sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)
        prd_vis_embeddings = self.prd_vis_embeddings(prd_features)
        ds_obj_vecs = self.obj_vecs
        ds_obj_vecs = Variable(torch.from_numpy(ds_obj_vecs.astype('float32')))
        so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)
        so_sem_embeddings = F.normalize(so_sem_embeddings, p=2, dim=1)
        so_sem_embeddings.t_()
        sbj_vis_embeddings = F.normalize(sbj_vis_embeddings, p=2, dim=1)
        sbj_sim_matrix = torch.mm(sbj_vis_embeddings, so_sem_embeddings)
        sbj_cls_scores = cfg.MODEL.NORM_SCALE * sbj_sim_matrix
        obj_vis_embeddings = F.normalize(obj_vis_embeddings, p=2, dim=1)
        obj_sim_matrix = torch.mm(obj_vis_embeddings, so_sem_embeddings)
        obj_cls_scores = cfg.MODEL.NORM_SCALE * obj_sim_matrix
        if not cfg.MODEL.USE_SEM_CONCAT:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32')))
            prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=1)
            prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)
            prd_sim_matrix = torch.mm(prd_vis_embeddings, prd_sem_embeddings.t_())
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
        else:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32')))
            prd_sem_hidden = self.prd_sem_hidden(ds_prd_vecs)
            sbj_vecs = self.obj_vecs[sbj_labels]
            sbj_vecs = Variable(torch.from_numpy(sbj_vecs.astype('float32')))
            if len(list(sbj_vecs.size())) == 1:
                sbj_vecs.unsqueeze_(0)
            sbj_sem_embeddings = self.so_sem_embeddings(sbj_vecs)
            sbj_sem_embeddings = sbj_sem_embeddings.unsqueeze(1).expand(sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)
            obj_vecs = self.obj_vecs[obj_labels]
            obj_vecs = Variable(torch.from_numpy(obj_vecs.astype('float32')))
            if len(list(obj_vecs.size())) == 1:
                obj_vecs.unsqueeze_(0)
            obj_sem_embeddings = self.so_sem_embeddings(obj_vecs)
            obj_sem_embeddings = obj_sem_embeddings.unsqueeze(1).expand(obj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)
            prd_sem_hidden = prd_sem_hidden.unsqueeze(0).expand(sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)
            spo_sem_feat = torch.cat((sbj_sem_embeddings.detach(), prd_sem_hidden, obj_sem_embeddings.detach()), dim=2)
            prd_sem_embeddings = self.prd_sem_embeddings(spo_sem_feat)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=2)
            prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)
            prd_vis_embeddings = prd_vis_embeddings.unsqueeze(-1)
            prd_sim_matrix = torch.bmm(prd_sem_embeddings, prd_vis_embeddings).squeeze(-1)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
        if cfg.MODEL.USE_FREQ_BIAS:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = prd_cls_scores + self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
        if not self.training:
            sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
        return prd_cls_scores, sbj_cls_scores, obj_cls_scores


class single_scale_relpn_outputs(nn.Module):
    """Add RelPN outputs to a single scale model (i.e., no FPN)."""

    def __init__(self):
        super().__init__()
        self.RelPN_GenerateProposalLabels = GenerateRelProposalLabelsOp()
        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]

    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds

    def remove_self_pairs(self, det_size, sbj_inds, obj_inds):
        mask = np.ones(sbj_inds.shape[0], dtype=bool)
        for i in range(det_size):
            mask[i + det_size * i] = False
        keeps = np.where(mask)[0]
        sbj_inds = sbj_inds[keeps]
        obj_inds = obj_inds[keeps]
        return sbj_inds, obj_inds

    def forward(self, det_rois, det_labels, det_scores, im_info, dataset_name, roidb=None):
        """
        det_rois: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """
        if roidb is not None:
            assert len(roidb) == 1
        sbj_inds = np.repeat(np.arange(det_rois.shape[0]), det_rois.shape[0])
        obj_inds = np.tile(np.arange(det_rois.shape[0]), det_rois.shape[0])
        if det_rois.shape[0] > 1:
            sbj_inds, obj_inds = self.remove_self_pairs(det_rois.shape[0], sbj_inds, obj_inds)
        sbj_rois = det_rois[sbj_inds]
        obj_rois = det_rois[obj_inds]
        im_scale = im_info.data.numpy()[:, 2][0]
        sbj_boxes = sbj_rois[:, 1:] / im_scale
        obj_boxes = obj_rois[:, 1:] / im_scale
        if cfg.MODEL.USE_OVLP_FILTER:
            ovlp_so = box_utils.bbox_pair_overlaps(sbj_boxes.astype(dtype=np.float32, copy=False), obj_boxes.astype(dtype=np.float32, copy=False))
            ovlp_inds = np.where(ovlp_so > 0)[0]
            sbj_inds = sbj_inds[ovlp_inds]
            obj_inds = obj_inds[ovlp_inds]
            sbj_rois = sbj_rois[ovlp_inds]
            obj_rois = obj_rois[ovlp_inds]
            sbj_boxes = sbj_boxes[ovlp_inds]
            obj_boxes = obj_boxes[ovlp_inds]
        return_dict = {}
        if self.training:
            blobs_out = self.RelPN_GenerateProposalLabels(sbj_rois, obj_rois, det_rois, roidb, im_info)
            return_dict.update(blobs_out)
        else:
            sbj_labels = det_labels[sbj_inds]
            obj_labels = det_labels[obj_inds]
            sbj_scores = det_scores[sbj_inds]
            obj_scores = det_scores[obj_inds]
            rel_rois = box_utils.rois_union(sbj_rois, obj_rois)
            return_dict['det_rois'] = det_rois
            return_dict['sbj_inds'] = sbj_inds
            return_dict['obj_inds'] = obj_inds
            return_dict['sbj_rois'] = sbj_rois
            return_dict['obj_rois'] = obj_rois
            return_dict['rel_rois'] = rel_rois
            return_dict['sbj_labels'] = sbj_labels
            return_dict['obj_labels'] = obj_labels
            return_dict['sbj_scores'] = sbj_scores
            return_dict['obj_scores'] = obj_scores
            return_dict['fg_size'] = np.array([sbj_rois.shape[0]], dtype=np.int32)
            im_scale = im_info.data.numpy()[:, 2][0]
            im_w = im_info.data.numpy()[:, 1][0]
            im_h = im_info.data.numpy()[:, 0][0]
            if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE or cfg.MODEL.USE_SEM_CONCAT:
                return_dict['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False) - 1
                return_dict['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False) - 1
            if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                lvl_min = cfg.FPN.ROI_MIN_LEVEL
                lvl_max = cfg.FPN.ROI_MAX_LEVEL
                rois_blob_names = ['det_rois', 'rel_rois']
                for rois_blob_name in rois_blob_names:
                    target_lvls = fpn_utils.map_rois_to_fpn_levels(return_dict[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                    fpn_utils.add_multilevel_roi_blobs(return_dict, rois_blob_name, return_dict[rois_blob_name], target_lvls, lvl_min, lvl_max)
        return return_dict


class single_scale_rpn_outputs(nn.Module):
    """Add RPN outputs to a single scale model (i.e., no FPN)."""

    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_in if cfg.RPN.OUT_DIM_AS_IN_DIM else cfg.RPN.OUT_DIM
        anchors = generate_anchors(stride=1.0 / spatial_scale, sizes=cfg.RPN.SIZES, aspect_ratios=cfg.RPN.ASPECT_RATIOS)
        num_anchors = anchors.shape[0]
        self.RPN_conv = nn.Conv2d(self.dim_in, self.dim_out, 3, 1, 1)
        self.n_score_out = num_anchors * 2 if cfg.RPN.CLS_ACTIVATION == 'softmax' else num_anchors
        self.RPN_cls_score = nn.Conv2d(self.dim_out, self.n_score_out, 1, 1, 0)
        self.RPN_bbox_pred = nn.Conv2d(self.dim_out, num_anchors * 4, 1, 1, 0)
        self.RPN_GenerateProposals = GenerateProposalsOp(anchors, spatial_scale)
        self.RPN_GenerateProposalLabels = GenerateProposalLabelsOp()
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.RPN_conv.weight, std=0.01)
        init.constant_(self.RPN_conv.bias, 0)
        init.normal_(self.RPN_cls_score.weight, std=0.01)
        init.constant_(self.RPN_cls_score.bias, 0)
        init.normal_(self.RPN_bbox_pred.weight, std=0.01)
        init.constant_(self.RPN_bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {'RPN_conv.weight': 'conv_rpn_w', 'RPN_conv.bias': 'conv_rpn_b', 'RPN_cls_score.weight': 'rpn_cls_logits_w', 'RPN_cls_score.bias': 'rpn_cls_logits_b', 'RPN_bbox_pred.weight': 'rpn_bbox_pred_w', 'RPN_bbox_pred.bias': 'rpn_bbox_pred_b'}
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, im_info, roidb=None):
        """
        x: feature maps from the backbone network. (Variable)
        im_info: (CPU Variable)
        roidb: (list of ndarray)
        """
        rpn_conv = F.relu(self.RPN_conv(x), inplace=True)
        rpn_cls_logits = self.RPN_cls_score(rpn_conv)
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv)
        return_dict = {'rpn_cls_logits': rpn_cls_logits, 'rpn_bbox_pred': rpn_bbox_pred}
        if not self.training or cfg.MODEL.FASTER_RCNN:
            if cfg.RPN.CLS_ACTIVATION == 'softmax':
                B, C, H, W = rpn_cls_logits.size()
                rpn_cls_prob = F.softmax(rpn_cls_logits.view(B, 2, C // 2, H, W), dim=1)
                rpn_cls_prob = rpn_cls_prob[:, 1].squeeze(dim=1)
            else:
                rpn_cls_prob = F.sigmoid(rpn_cls_logits)
            rpn_rois, rpn_rois_prob = self.RPN_GenerateProposals(rpn_cls_prob, rpn_bbox_pred, im_info)
            return_dict['rpn_rois'] = rpn_rois
            return_dict['rpn_roi_probs'] = rpn_rois_prob
        if cfg.MODEL.FASTER_RCNN:
            if self.training:
                blobs_out = self.RPN_GenerateProposalLabels(rpn_rois, roidb, im_info)
                return_dict.update(blobs_out)
            else:
                return_dict['rois'] = return_dict['rpn_rois']
        return return_dict


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)


class GroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return myF.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, affine={affine}'.format(**self.__dict__)


class BilinearInterpolation2d(nn.Module):
    """Bilinear interpolation in space of scale.

    Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """

    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        assert in_channels == out_channels
        assert up_scale % 2 == 0, 'Scale should be even'
        self.in_channes = in_channels
        self.out_channels = out_channels
        self.up_scale = int(up_scale)
        self.padding = up_scale // 2

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)
        kernel = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
        kernel[range(in_channels), range(out_channels), :, :] = bil_filt
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=self.up_scale, padding=self.padding)
        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)


class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs))
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)


def _get_stream(device):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.cuda.device_count()
    if _streams[device] is None:
        _streams[device] = torch.Stream(device)
    return _streams[device]


class Scatter(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        ctx.target_gpus = target_gpus
        ctx.chunk_sizes = chunk_sizes
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.is_cuda else -1
        streams = None
        if ctx.input_device == -1:
            streams = [_get_stream(device) for device in ctx.target_gpus]
        outputs = comm.scatter(input, ctx.target_gpus, ctx.chunk_sizes, ctx.dim, streams)
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.device(ctx.target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


def scatter(inputs, target_gpus, dim=0):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), 'Tensors not supported in scatter.'
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
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


class DataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    .. warning::
        Forward and backwrad hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
        cpu_keywords: list of argument keywords that could be used in `forward` to
            indicating not moving the argument to gpu. Currently, only support
            argument of type: Variable

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0, cpu_keywords=[], minibatch=False, batch_outputs=True):
        super(DataParallel, self).__init__()
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module
        self.cpu_keywords = cpu_keywords
        self.minibatch = minibatch
        self.batch_outputs = batch_outputs

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.minibatch:
            inputs_list, kwargs_list = [], []
            for i, device_id in enumerate(self.device_ids):
                mini_inputs = [x[i] for x in inputs]
                mini_kwargs = dict([(k, v[i]) for k, v in kwargs.items()])
                a, b = self._minibatch_scatter(device_id, *mini_inputs, **mini_kwargs)
                inputs_list.append(a)
                kwargs_list.append(b)
            inputs = inputs_list
            kwargs = kwargs_list
        else:
            kwargs_cpu = {}
            for k in kwargs:
                if k in self.cpu_keywords:
                    v = kwargs[k]
                    kwargs_cpu[k] = v
            for k in self.cpu_keywords:
                kwargs.pop(k, None)
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            for k, v in kwargs_cpu.items():
                split_size = v.size(self.dim) / len(self.device_ids)
                assert split_size.is_integer()
                kwargs_cpu[k] = list(map(Variable, torch.split(v.data, int(split_size), self.dim)))
            kwargs_cpu = list(map(dict, zip(*[[(k, v) for v in vs] for k, vs in kwargs_cpu.items()])))
            for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
                d_gpu.update(d_cpu)
        if len(self.device_ids) == 1:
            outputs = [self.module(*inputs[0], **kwargs[0])]
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, kwargs)
        if self.batch_outputs:
            return self.gather(outputs, self.output_device)
        else:
            return [self.gather([x], self.output_device) for x in outputs]

    def _minibatch_scatter(self, device_id, *inputs, **kwargs):
        kwargs_cpu = {}
        for k in kwargs:
            if k in self.cpu_keywords:
                kwargs_cpu[k] = kwargs[k]
        for k in self.cpu_keywords:
            kwargs.pop(k, None)
        inputs, kwargs = self.scatter(inputs, kwargs, [device_id])
        kwargs_cpu = [kwargs_cpu]
        for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
            d_gpu.update(d_cpu)
        return inputs[0], kwargs[0]

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AffineChannel2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Depth3DGridGen,
     lambda: ([], {'height': 4, 'width': 4}),
     lambda: ([torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Depth3DGridGen_with_mask,
     lambda: ([], {'height': 4, 'width': 4}),
     lambda: ([torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGG16_conv_body,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_jz462_Large_Scale_VRD_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

