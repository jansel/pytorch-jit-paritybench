import sys
_module = sys.modules[__name__]
del sys
core = _module
config = _module
test = _module
test_engine = _module
datasets = _module
cityscapes = _module
coco_to_cityscapes_id = _module
convert_cityscapes_to_coco = _module
convert_coco_model_to_cityscapes = _module
cityscapes_json_dataset_evaluator = _module
dataset_catalog = _module
dis_eval = _module
dummy_datasets = _module
json_dataset = _module
json_dataset_evaluator = _module
roidb = _module
task_evaluation = _module
voc_dataset_evaluator = _module
voc_eval = _module
model = _module
nms = _module
_ext = _module
build = _module
nms_gpu = _module
nms_wrapper = _module
pcl = _module
pcl = _module
pcl_losses = _module
functions = _module
modules = _module
pcl_losses = _module
roi_align = _module
roi_align = _module
roi_crop = _module
crop_resize = _module
gridgen = _module
gridgen = _module
roi_crop = _module
roi_pooling = _module
roi_pool = _module
roi_pool = _module
utils = _module
net_utils = _module
modeling = _module
model_builder = _module
pcl_heads = _module
roi_xfrom = _module
roi_align = _module
vgg16 = _module
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
loader = _module
minibatch = _module
setup = _module
blob = _module
boxes = _module
collections = _module
colormap = _module
detectron_weight_helper = _module
env = _module
image = _module
io = _module
logging = _module
misc = _module
net = _module
subprocess = _module
timer = _module
training_stats = _module
vgg_weights_helper = _module
vis = _module
_init_paths = _module
download_imagenet_weights = _module
reeval = _module
test_net = _module
train_net_step = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import copy


import numpy as np


import torch


import torch.nn as nn


from torch.nn import init


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import random


from functools import wraps


import logging


import torch.nn.init as init


from collections import OrderedDict


import math


from functools import reduce


from torch.nn import Module


from collections import defaultdict


class PCLLosses(torch.autograd.Function):

    def forward(ctx, pcl_probs, labels, cls_loss_weights, gt_assignment,
        pc_labels, pc_probs, pc_count, img_cls_loss_weights, im_labels):
        (ctx.pcl_probs, ctx.labels, ctx.cls_loss_weights, ctx.gt_assignment,
            ctx.pc_labels, ctx.pc_probs, ctx.pc_count, ctx.
            img_cls_loss_weights, ctx.im_labels) = (pcl_probs, labels,
            cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count,
            img_cls_loss_weights, im_labels)
        batch_size, channels = pcl_probs.size()
        loss = 0
        ctx.mark_non_differentiable(labels, cls_loss_weights, gt_assignment,
            pc_labels, pc_probs, pc_count, img_cls_loss_weights, im_labels)
        for c in range(channels):
            if im_labels[0, c] != 0:
                if c == 0:
                    for i in range(batch_size):
                        if labels[0, i] == 0:
                            loss -= cls_loss_weights[0, i] * torch.log(
                                pcl_probs[i, c])
                else:
                    for i in range(pc_labels.size(0)):
                        if pc_probs[0, i] == c:
                            loss -= img_cls_loss_weights[0, i] * torch.log(
                                pc_probs[0, i])
        return loss / batch_size

    def backward(ctx, grad_output):
        (pcl_probs, labels, cls_loss_weights, gt_assignment, pc_labels,
            pc_probs, pc_count, img_cls_loss_weights, im_labels) = (ctx.
            pcl_probs, ctx.labels, ctx.cls_loss_weights, ctx.gt_assignment,
            ctx.pc_labels, ctx.pc_probs, ctx.pc_count, ctx.
            img_cls_loss_weights, ctx.im_labels)
        grad_input = grad_output.new(pcl_probs.size()).zero_()
        batch_size, channels = pcl_probs.size()
        for i in range(batch_size):
            for c in range(channels):
                grad_input[i, c] = 0
                if im_labels[0, c] != 0:
                    if c == 0:
                        if labels[0, i] == 0:
                            grad_input[i, c] = -cls_loss_weights[0, i
                                ] / pcl_probs[i, c]
                    elif labels[0, i] == c:
                        pc_index = int(gt_assignment[0, i].item())
                        if c != pc_labels[0, pc_index]:
                            print('labels mismatch.')
                        grad_input[i, c] = -img_cls_loss_weights[0, pc_index
                            ] / (pc_count[0, pc_index] * pc_probs[0, pc_index])
        grad_input /= batch_size
        return grad_input, grad_output.new(labels.size()).zero_(
            ), grad_output.new(cls_loss_weights.size()).zero_(
            ), grad_output.new(gt_assignment.size()).zero_(), grad_output.new(
            pc_labels.size()).zero_(), grad_output.new(pc_probs.size()).zero_(
            ), grad_output.new(pc_count.size()).zero_(), grad_output.new(
            img_cls_loss_weights.size()).zero_(), grad_output.new(im_labels
            .size()).zero_()


class _PCL_Losses(Module):

    def forward(self, pcl_prob, labels, cls_loss_weights, gt_assignment,
        pc_labels, pc_probs, pc_count, img_cls_loss_weights, im_labels_real):
        return PCLLosses()(pcl_prob, labels, cls_loss_weights,
            gt_assignment, pc_labels, pc_probs, pc_count,
            img_cls_loss_weights, im_labels_real)


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height,
            self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.
                aligned_width, self.spatial_scale, features, rois, output)
        else:
            raise NotImplementedError
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height,
            data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.
            aligned_width, self.spatial_scale, grad_output, self.rois,
            grad_input)
        return grad_input, None


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
            self.spatial_scale)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


class AffineGridGenFunction(Function):

    def __init__(self, height, width, lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.input1 = input1
        output = input1.new(torch.Size([input1.size(0)]) + self.grid.size()
            ).zero_()
        self.batchgrid = input1.new(torch.Size([input1.size(0)]) + self.
            grid.size()).zero_()
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid.astype(self.batchgrid[i])
        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height * self.
                width, 3), torch.transpose(input1, 1, 2)).view(-1, self.
                height, self.width, 2)
        return output

    def backward(self, grad_output):
        grad_input1 = self.input1.new(self.input1.size()).zero_()
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(
            grad_output.view(-1, self.height * self.width, 2), 1, 2), self.
            batchgrid.view(-1, self.height * self.width, 3))
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
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if input1.is_cuda:
            self.batchgrid = self.batchgrid
        output = torch.bmm(self.batchgrid.view(-1, self.height * self.width,
            3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.
            width, 2)
        return output


class CylinderGridGenV2(Module):

    def __init__(self, height, width, lr=1):
        super(CylinderGridGenV2, self).__init__()
        self.height, self.width = height, width
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input):
        self.batchgrid = torch.zeros(torch.Size([input.size(0)]) + self.
            grid.size())
        for i in range(input.size(0)):
            self.batchgrid[(i), :, :, :] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        input_u = input.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1)
        output0 = self.batchgrid[:, :, :, 0:1]
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (self.batchgrid[:, :,
            :, 1:2] + self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (
            np.pi / 2)
        output = torch.cat([output0, output1], 3)
        return output


class DenseAffineGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(DenseAffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))

    def forward(self, input1):
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
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
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
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
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, input1, input2):
        self.batchgrid3d = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid3d.size())
        for i in range(input1.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.
            grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 0:4]), 3)
        y = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 4:8]), 3)
        z = torch.sum(torch.mul(self.batchgrid3d, input1[:, :, :, 8:]), 3)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = input2.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
        output2 = torch.cat([output[:, :, :, 0:1], output1], 3)
        return output2


class Depth3DGridGen(Module):

    def __init__(self, height, width, lr=1, aux_loss=False):
        super(Depth3DGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.lr = lr
        self.grid = np.zeros([self.height, self.width, 3], dtype=np.float32)
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        x = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
            FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).type(
            torch.FloatTensor))
        phi = phi / np.pi
        input_u = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.width, 1
            )
        output = torch.cat([theta, phi], 3)
        output1 = torch.atan(torch.tan(np.pi / 2.0 * (output[:, :, :, 1:2] +
            self.batchgrid[:, :, :, 2:] * input_u[:, :, :, :]))) / (np.pi / 2)
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
        self.grid[:, :, (0)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.height), 0), repeats=self.width, axis=
            0).T, 0)
        self.grid[:, :, (1)] = np.expand_dims(np.repeat(np.expand_dims(np.
            arange(-1, 1, 2.0 / self.width), 0), repeats=self.height, axis=
            0), 0)
        self.grid[:, :, (2)] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        self.theta = self.grid[:, :, (0)] * np.pi / 2 + np.pi / 2
        self.phi = self.grid[:, :, (1)] * np.pi
        self.x = torch.sin(self.theta) * torch.cos(self.phi)
        self.y = torch.sin(self.theta) * torch.sin(self.phi)
        self.z = torch.cos(self.theta)
        self.grid3d = torch.from_numpy(np.zeros([self.height, self.width, 4
            ], dtype=np.float32))
        self.grid3d[:, :, (0)] = self.x
        self.grid3d[:, :, (1)] = self.y
        self.grid3d[:, :, (2)] = self.z
        self.grid3d[:, :, (3)] = self.grid[:, :, (2)]

    def forward(self, depth, trans0, trans1, rotate):
        self.batchgrid3d = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid3d.size())
        for i in range(depth.size(0)):
            self.batchgrid3d[i] = self.grid3d
        self.batchgrid3d = Variable(self.batchgrid3d)
        self.batchgrid = torch.zeros(torch.Size([depth.size(0)]) + self.
            grid.size())
        for i in range(depth.size(0)):
            self.batchgrid[i] = self.grid
        self.batchgrid = Variable(self.batchgrid)
        if depth.is_cuda:
            self.batchgrid = self.batchgrid
            self.batchgrid3d = self.batchgrid3d
        x_ = self.batchgrid3d[:, :, :, 0:1] * depth + trans0.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        y_ = self.batchgrid3d[:, :, :, 1:2] * depth + trans1.view(-1, 1, 1, 1
            ).repeat(1, self.height, self.width, 1)
        z = self.batchgrid3d[:, :, :, 2:3] * depth
        rotate_z = rotate.view(-1, 1, 1, 1).repeat(1, self.height, self.
            width, 1) * np.pi
        x = x_ * torch.cos(rotate_z) - y_ * torch.sin(rotate_z)
        y = x_ * torch.sin(rotate_z) + y_ * torch.cos(rotate_z)
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-05
        theta = torch.acos(z / r) / (np.pi / 2) - 1
        if depth.is_cuda:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                cuda.FloatTensor) * (y.ge(0).type(torch.cuda.FloatTensor) -
                y.lt(0).type(torch.cuda.FloatTensor))
        else:
            phi = torch.atan(y / (x + 1e-05)) + np.pi * x.lt(0).type(torch.
                FloatTensor) * (y.ge(0).type(torch.FloatTensor) - y.lt(0).
                type(torch.FloatTensor))
        phi = phi / np.pi
        output = torch.cat([theta, phi], 3)
        return output


defines = []


extra_objects = ['src/nms_cuda_kernel.cu.o']


headers = []


sources = []


with_cuda = False


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
        output = features.new(num_rois, num_channels, ctx.pooled_height,
            ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height,
            ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.
                pooled_width, ctx.spatial_scale, _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.
                pooled_width, ctx.spatial_scale, features, rois, output,
                ctx.argmax)
        return output

    def backward(ctx, grad_output):
        assert ctx.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height,
            data_width).zero_()
        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.
            pooled_width, ctx.spatial_scale, grad_output, ctx.rois,
            grad_input, ctx.argmax)
        return grad_input, None


class _RoIPooling(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.
            spatial_scale)(features, rois)


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = box_utils.bbox_overlaps(boxes.astype(dtype=np.float32, copy=
        False), boxes.astype(dtype=np.float32, copy=False))
    return (overlaps > iou_threshold).astype(np.float32)


_global_config['TRAIN'] = 4


_global_config['RNG_SEED'] = 4


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER, random_state=
        cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)
    index = np.where(kmeans.labels_ == high_score_label)[0]
    if len(index) == 0:
        index = np.array([np.argmax(probs)])
    return index


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[(0), :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, (i)].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape
                (-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[(idxs), :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]
            graph = _build_graph(boxes_tmp, cfg.TRAIN.GRAPH_IOU_THRESHOLD)
            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[(tmp), :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))
                graph[:, (inds)] = 0
                graph[(inds), :] = 0
                count = count - len(inds)
                if count <= 5:
                    break
            gt_boxes_tmp = boxes_tmp[(keep_idxs), :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()
            keep_idxs_new = np.argsort(gt_scores_tmp)[-1:-1 - min(len(
                gt_scores_tmp), cfg.TRAIN.MAX_PC_NUM):-1]
            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[(keep_idxs_new), :]))
            gt_scores = np.vstack((gt_scores, gt_scores_tmp[keep_idxs_new].
                reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((len(
                keep_idxs_new), 1), dtype=np.int32)))
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][
                keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new],
                axis=0)
    proposals = {'gt_boxes': gt_boxes, 'gt_classes': gt_classes,
        'gt_scores': gt_scores}
    return proposals


def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = box_utils.bbox_overlaps(all_rois.astype(dtype=np.float32,
        copy=False), gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0
    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1
    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])
    return (labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs,
        pc_count, img_cls_loss_weights)


def PCL(boxes, cls_prob, im_labels, cls_prob_new):
    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-09
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps
    proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(), im_labels
        .copy())
    (labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count,
        img_cls_loss_weights) = (_get_proposal_clusters(boxes.copy(),
        proposals, im_labels.copy(), cls_prob_new.copy()))
    return {'labels': labels.reshape(1, -1).astype(np.float32).copy(),
        'cls_loss_weights': cls_loss_weights.reshape(1, -1).astype(np.
        float32).copy(), 'gt_assignment': gt_assignment.reshape(1, -1).
        astype(np.float32).copy(), 'pc_labels': pc_labels.reshape(1, -1).
        astype(np.float32).copy(), 'pc_probs': pc_probs.reshape(1, -1).
        astype(np.float32).copy(), 'pc_count': pc_count.reshape(1, -1).
        astype(np.float32).copy(), 'img_cls_loss_weights':
        img_cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        'im_labels_real': np.hstack((np.array([[1]]), im_labels)).astype(np
        .float32).copy()}


_global_config['PYTORCH_VERSION_LESS_THAN_040'] = 4


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
            raise ValueError(
                'You should call this function only on inference.Set the network in inference mode by net.eval().'
                )
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


_global_config['FAST_RCNN'] = 4


_global_config['REFINE_TIMES'] = 4


_global_config['CROP_RESIZE_WITH_MAX_POOL'] = 4


_global_config['MODEL'] = 4


class Generalized_RCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(self.Conv_Body
            .dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_MIL_Outs = pcl_heads.mil_outputs(self.Box_Head.dim_out,
            cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = pcl_heads.refine_outputs(self.Box_Head.
            dim_out, cfg.MODEL.NUM_CLASSES + 1)
        self.Refine_Losses = [PCLLosses() for i in range(cfg.REFINE_TIMES)]
        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, rois, labels):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels)

    def _forward(self, data, rois, labels):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)
        device_id = im_data.get_device()
        return_dict = {}
        blob_conv = self.Conv_Body(im_data)
        if not self.training:
            return_dict['blob_conv'] = blob_conv
        box_feat = self.Box_Head(blob_conv, rois)
        mil_score = self.Box_MIL_Outs(box_feat)
        refine_score = self.Box_Refine_Outs(box_feat)
        if self.training:
            return_dict['losses'] = {}
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            return_dict['losses']['loss_im_cls'] = loss_im_cls
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]
            for i_refine, refine in enumerate(refine_score):
                if i_refine == 0:
                    pcl_output = PCL(boxes, mil_score, im_labels, refine)
                else:
                    pcl_output = PCL(boxes, refine_score[i_refine - 1],
                        im_labels, refine)
                refine_loss = self.Refine_Losses[i_refine](refine, Variable
                    (torch.from_numpy(pcl_output['labels'])), Variable(
                    torch.from_numpy(pcl_output['cls_loss_weights'])),
                    Variable(torch.from_numpy(pcl_output['gt_assignment'])),
                    Variable(torch.from_numpy(pcl_output['pc_labels'])),
                    Variable(torch.from_numpy(pcl_output['pc_probs'])),
                    Variable(torch.from_numpy(pcl_output['pc_count'])),
                    Variable(torch.from_numpy(pcl_output[
                    'img_cls_loss_weights'])), Variable(torch.from_numpy(
                    pcl_output['im_labels_real'])))
                return_dict['losses']['refine_loss%d' % i_refine
                    ] = refine_loss.clone()
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
        else:
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score
        return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
        resolution=7, spatial_scale=1.0 / 16.0, sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'
            }, 'Unknown pooling method: {}'.format(method)
        if method == 'RoIPoolF':
            xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(
                blobs_in, rois)
        elif method == 'RoICrop':
            grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:],
                self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, (1)], grid_xy.data
                [:, :, :, (0)]], 3).contiguous()
            xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                xform_out = F.max_pool2d(xform_out, 2, 2)
        elif method == 'RoIAlign':
            xform_out = RoIAlignFunction(resolution, resolution,
                spatial_scale, sampling_ratio)(blobs_in, rois)
        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}
            d_orphan = []
            for name, m_child in self.named_children():
                if list(m_child.parameters()):
                    child_map, child_orphan = m_child.detectron_weight_mapping(
                        )
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan
        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value


class mil_outputs(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mil_score0 = nn.Linear(dim_in, dim_out)
        self.mil_score1 = nn.Linear(dim_in, dim_out)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.mil_score0.weight, std=0.01)
        init.constant_(self.mil_score0.bias, 0)
        init.normal_(self.mil_score1.weight, std=0.01)
        init.constant_(self.mil_score1.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {'mil_score0.weight': 'mil_score0_w',
            'mil_score0.bias': 'mil_score0_b', 'mil_score1.weight':
            'mil_score1_w', 'mil_score1.bias': 'mil_score1_b'}
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        mil_score0 = self.mil_score0(x)
        mil_score1 = self.mil_score1(x)
        mil_score = F.softmax(mil_score0, dim=0) * F.softmax(mil_score1, dim=1)
        return mil_score


class refine_outputs(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.refine_score = []
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)
        self._init_weights()

    def _init_weights(self):
        for i_refine in range(cfg.REFINE_TIMES):
            init.normal_(self.refine_score[i_refine].weight, std=0.01)
            init.constant_(self.refine_score[i_refine].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        for i_refine in range(cfg.REFINE_TIMES):
            detectron_weight_mapping.update({('refine_score.%d.weight' %
                i_refine): 'refine_score%d_w' % i_refine, (
                'refine_score.%d.bias' % i_refine): 'refine_score%d_b' %
                i_refine})
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        refine_score = [F.softmax(refine(x), dim=1) for refine in self.
            refine_score]
        return refine_score


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale,
        sampling_ratio):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
            self.spatial_scale, self.sampling_ratio)(features, rois)


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale,
        sampling_ratio):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale, self.sampling_ratio)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale,
        sampling_ratio):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 
            1, self.spatial_scale, self.sampling_ratio)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


_global_config['VGG'] = 4


class dilated_conv5_body(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
            padding=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(64, 64,
            kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(inplace
            =True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride
            =1, padding=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(128,
            128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(
            inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3,
            stride=1, padding=1, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            stride=1, padding=1, bias=True), nn.ReLU(inplace=True), nn.
            MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3,
            stride=1, padding=1, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3,
            stride=1, padding=1, bias=True), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
            stride=1, padding=2, dilation=2, bias=True), nn.ReLU(inplace=
            True), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2,
            dilation=2, bias=True), nn.ReLU(inplace=True), nn.Conv2d(512, 
            512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True))
        self.dim_out = 512
        self.spatial_scale = 1.0 / 8.0
        self._init_modules()

    def _init_modules(self):
        assert cfg.VGG.FREEZE_AT in [0, 2, 3, 4, 5]
        for i in range(1, cfg.VGG.FREEZE_AT + 1):
            freeze_params(getattr(self, 'conv%d' % i))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {'conv1.0.weight': 'conv1_0_w',
            'conv1.0.bias': 'conv1_0_b', 'conv1.2.weight': 'conv1_2_w',
            'conv1.2.bias': 'conv1_2_b', 'conv2.0.weight': 'conv2_0_w',
            'conv2.0.bias': 'conv2_0_b', 'conv2.2.weight': 'conv2_2_w',
            'conv2.2.bias': 'conv2_2_b', 'conv3.0.weight': 'conv3_0_w',
            'conv3.0.bias': 'conv3_0_b', 'conv3.2.weight': 'conv3_2_w',
            'conv3.2.bias': 'conv3_2_b', 'conv3.4.weight': 'conv3_4_w',
            'conv3.4.bias': 'conv3_4_b', 'conv4.0.weight': 'conv4_0_w',
            'conv4.0.bias': 'conv4_0_b', 'conv4.2.weight': 'conv4_2_w',
            'conv4.2.bias': 'conv4_2_b', 'conv4.4.weight': 'conv4_4_w',
            'conv4.4.bias': 'conv4_4_b', 'conv5.0.weight': 'conv5_0_w',
            'conv5.0.bias': 'conv5_0_b', 'conv5.2.weight': 'conv5_2_w',
            'conv5.2.bias': 'conv5_2_b', 'conv5.4.weight': 'conv5_4_w',
            'conv5.4.bias': 'conv5_4_b'}
        orphan_in_detectron = []
        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        self.training = mode
        for i in range(cfg.VGG.FREEZE_AT + 1, 6):
            getattr(self, 'conv%d' % i).train(mode)

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, 'conv%d' % i)(x)
        return x


class roi_2mlp_head(nn.Module):

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size ** 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {'fc1.weight': 'fc6_w', 'fc1.bias':
            'fc6_b', 'fc2.weight': 'fc7_w', 'fc2.bias': 'fc7_b'}
        return detectron_weight_mapping, []

    def forward(self, x, rois):
        x = self.roi_xform(x, rois, method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION, spatial_scale=
            self.spatial_scale, sampling_ratio=cfg.FAST_RCNN.
            ROI_XFORM_SAMPLING_RATIO)
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x


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
        return x * self.weight.view(1, self.num_features, 1, 1
            ) + self.bias.view(1, self.num_features, 1, 1)


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
        return myF.group_norm(x, self.num_groups, self.weight, self.bias,
            self.eps)

    def extra_repr(self):
        return ('{num_groups}, {num_channels}, eps={eps}, affine={affine}'.
            format(**self.__dict__))


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
            return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] -
                center) / factor)
        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)
        kernel = np.zeros((in_channels, out_channels, kernel_size,
            kernel_size), dtype=np.float32)
        kernel[(range(in_channels)), (range(out_channels)), :, :] = bil_filt
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=self.up_scale, padding=self.padding)
        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)


class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(map(lambda i: i.is_cuda, inputs))
        ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs))
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + Scatter.apply(ctx.input_gpus, ctx.input_sizes,
            ctx.dim, grad_output)


def _get_stream(device):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.cuda.device_count()
    if _streams[device] is None:
        _streams[device] = torch.cuda.Stream(device)
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
        outputs = comm.scatter(input, ctx.target_gpus, ctx.chunk_sizes, ctx
            .dim, streams)
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(ctx.target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *
            grad_output)


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

    def __init__(self, module, device_ids=None, output_device=None, dim=0,
        cpu_keywords=[], minibatch=False, batch_outputs=True):
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
                a, b = self._minibatch_scatter(device_id, *mini_inputs, **
                    mini_kwargs)
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
                kwargs_cpu[k] = list(map(Variable, torch.split(v.data, int(
                    split_size), self.dim)))
            kwargs_cpu = list(map(dict, zip(*[[(k, v) for v in vs] for k,
                vs in kwargs_cpu.items()])))
            for d_gpu, d_cpu in zip(kwargs, kwargs_cpu):
                d_gpu.update(d_cpu)
        if len(self.device_ids) == 1:
            outputs = [self.module(*inputs[0], **kwargs[0])]
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(
                inputs)])
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
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:
            len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ppengtang_pcl_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(AffineChannel2d(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Depth3DGridGen(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Depth3DGridGen_with_mask(*[], **{'height': 4, 'width': 4}), [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(mil_outputs(*[], **{'dim_in': 4, 'dim_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(refine_outputs(*[], **{'dim_in': 4, 'dim_out': 4}), [torch.rand([4, 4, 4, 4])], {})

