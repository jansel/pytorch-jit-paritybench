import sys
_module = sys.modules[__name__]
del sys
vkitti_utils = _module
bulb = _module
net = _module
saver = _module
utils = _module
derender3d = _module
data_loader = _module
datasets = _module
models = _module
derenderer = _module
renderer = _module
transforms = _module
cityscapes = _module
config = _module
convert_from_keras = _module
demo = _module
model = _module
nms = _module
build = _module
nms_wrapper = _module
pth_nms = _module
roialign = _module
roi_align = _module
build = _module
crop_and_resize = _module
roi_align = _module
utils = _module
visualize = _module
vkitti = _module
neural_renderer = _module
cross = _module
get_points_from_angles = _module
lighting = _module
load_obj = _module
look = _module
look_at = _module
mesh = _module
optimizers = _module
perspective = _module
rasterize = _module
save_obj = _module
vertices_to_faces = _module
main = _module
nn = _module
modules = _module
batchnorm = _module
comm = _module
replicate = _module
test_numeric_batchnorm = _module
test_sync_batchnorm = _module
unittest = _module
parallel = _module
data_parallel = _module
data = _module
dataloader = _module
dataset = _module
distributed = _module
sampler = _module
th = _module
models = _module
resnet = _module
utils = _module
vkitti_dataset = _module
vkitti_eval = _module
vkitti_test = _module
vkitti_train = _module
base_data_loader = _module
base_dataset = _module
cityscapes_dataset = _module
cityscapes_labels = _module
custom_dataset_data_loader = _module
image_folder = _module
vkitti_dataset = _module
edit_benchmark = _module
edit_vkitti = _module
base_model = _module
models = _module
networks = _module
pix2pixHD_model = _module
ui_model = _module
options = _module
base_options = _module
edit_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
util = _module
html = _module
image_pool = _module
util = _module
util2 = _module
visualizer = _module

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


import itertools


import torch


import collections


import numpy as np


import matplotlib.cm


import pandas as pd


import torch.nn.functional as F


import torchvision


import random


import scipy.ndimage


import scipy.special


from torch import Tensor


from torch import IntTensor


from torch.distributions import Categorical


from torch.nn.modules import Module


from torch.autograd import Function


import time


from torchvision import transforms


import math


import matplotlib


import matplotlib.pyplot as plt


import re


import torch.nn as nn


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


from torch import nn


import scipy.misc


import functools


from torch.nn import functional as F


from torch.nn.parallel import DataParallel


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


from torch.nn.parallel.data_parallel import DataParallel


import torch.cuda as cuda


from torch.nn.parallel._functions import Gather


import torch.multiprocessing as multiprocessing


from torch._C import _set_worker_signal_handlers


from torch._C import _remove_worker_pids


from torch._C import _error_if_any_worker_fails


from torch._six import string_classes


from torch._six import int_classes


import warnings


from torch._utils import _accumulate


from torch import randperm


from torch.distributed import get_world_size


from torch.distributed import get_rank


from scipy.io import loadmat


import torch.utils.data as data


import torchvision.transforms as transforms


from math import pi


from collections import OrderedDict


from torchvision import models


import inspect


from scipy.ndimage.interpolation import zoom


class Derenderer(Module):
    in_size = 4
    hidden_size = 256

    def __init__(self, num_classes=8, grid_size=4):
        super(Derenderer, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.out_sizes = {'_theta_deltas': 2, '_translation2ds': 2, '_log_scales': 3, '_log_depths': 1, '_class_probs': num_classes, '_ffd_coeffs': num_classes * grid_size ** 3 * 3}
        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.net.fc = torch.nn.Linear(512, Derenderer.hidden_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(self.hidden_size + self.in_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self._fc3 = torch.nn.Linear(self.hidden_size, sum(self.out_sizes.values()))

    def forward(self, images, mroi_norms, droi_norms):
        x = self.net(images)
        x = self.relu(x)
        x = torch.cat([x, mroi_norms, droi_norms], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self._fc3(x)
        _theta_deltas, _translation2ds, _log_scales, _log_depths, _class_probs, _ffd_coeffs = torch.split(x, list(self.out_sizes.values()), dim=1)
        _theta_deltas = _theta_deltas / torch.norm(_theta_deltas, p=2, dim=1, keepdim=True)
        _class_probs = torch.nn.functional.softmax(_class_probs, dim=1)
        _ffd_coeffs = _ffd_coeffs.view(-1, self.num_classes, self.grid_size ** 3 * 3)
        return {'_theta_deltas': _theta_deltas, '_translation2ds': _translation2ds, '_log_scales': _log_scales, '_log_depths': _log_depths, '_class_probs': _class_probs, '_ffd_coeffs': _ffd_coeffs}


class FFD(Module):


    class Constraint:


        class Type:
            symmetry = 0
            homogeneity = 1


        class Axis:
            x = 0
            y = 1
            z = 2

        @staticmethod
        def symmetry(axis):
            c = FFD.Constraint(FFD.Constraint.Type.symmetry)
            c.axis = axis
            return c

        @staticmethod
        def homogeneity(axis, index):
            c = FFD.Constraint(FFD.Constraint.Type.homogeneity)
            c.axis = axis
            c.index = index
            return c

        def __init__(self, type):
            self.type = type

    @staticmethod
    def flip(x, dim):
        shape = x.shape
        index = torch.arange(shape[dim] - 1, -1, -1, dtype=torch.long)
        return torch.index_select(x, dim, index)

    def __init__(self, vertices, num_grids=4, constraints=None):
        super(FFD, self).__init__()
        assert num_grids % 2 == 0
        self.num_grids = num_grids
        self.constraints = constraints
        grids = np.arange(num_grids)
        binoms = Tensor(scipy.special.binom(num_grids - 1, grids))
        grid_1ds = Tensor(grids)
        grid_3ds = Tensor(np.meshgrid(grids, grids, grids, indexing='ij'))
        coeff = binoms * torch.pow(torch.unsqueeze(0.5 + vertices, dim=2), grid_1ds) * torch.pow(torch.unsqueeze(0.5 - vertices, dim=2), num_grids - 1 - grid_1ds)
        self.B = torch.einsum('ni,nj,nk->nijk', torch.unbind(coeff, dim=1))
        self.B = torch.unsqueeze(self.B, dim=1)
        self.P0 = grid_3ds / (num_grids - 1) - 0.5

    def forward(self, ffd_coeff):
        dP = ffd_coeff.view(3, self.num_grids, self.num_grids, self.num_grids)
        for constraint in self.constraints:
            if constraint.type == FFD.Constraint.Type.symmetry:
                _dP = FFD.flip(dP, dim=constraint.axis + 1)
                _dPx, _dPy, _dPz = torch.unbind(_dP, dim=0)
                _dP = torch.stack([_dPx, _dPy, -_dPz], dim=0)
                dP = (dP + _dP) / 2
            elif constraint.type == FFD.Constraint.Type.homogeneity:
                dPs = torch.unbind(dP, dim=constraint.axis + 1)
                _dPs = [dPs[index] for index in constraint.index]
                _dP_mean = sum(_dPs) / len(_dPs)
                _dPs = []
                for index in range(self.num_grids):
                    if index in constraint.index:
                        _dP = _dP_mean.clone()
                        _dP[constraint.axis] = dPs[index][constraint.axis]
                    else:
                        _dP = dPs[index]
                    _dPs.append(_dP)
                dP = torch.stack(_dPs, dim=constraint.axis + 1)
        PB = (self.P0 + dP) * self.B
        V = PB.view(-1, 3, self.num_grids * self.num_grids * self.num_grids).sum(dim=2)
        return V


class PerspectiveTransform(Module):

    def forward(self, vertices, scales=None, rotations=None, translations=None, perspective_translations=None, zooms=None, zoom_tos=None):
        if scales is not None:
            scales = scales.unsqueeze(dim=1)
            vertices = vertices * scales
        if rotations is not None:
            a, b, c, d = torch.unbind(rotations, dim=1)
            T = torch.stack([a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c, 2 * b * c + 2 * a * d, a * a - b * b + c * c - d * d, 2 * c * d - 2 * a * b, 2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a - b * b - c * c + d * d], dim=1).view(-1, 3, 3)
            vertices = torch.matmul(vertices, torch.transpose(T, dim0=1, dim1=2))
        if translations is not None:
            translations = translations.unsqueeze(dim=1)
            vertices = vertices + translations
        if perspective_translations is not None:
            perspective_translations = perspective_translations.unsqueeze(dim=1)
        else:
            perspective_translations = translations
        x, y, z = torch.unbind(vertices, dim=2)
        x0, y0, z0 = torch.unbind(perspective_translations, dim=2)
        x = x - x0 / z0 * z
        y = y - y0 / z0 * z
        if zoom_tos is not None:
            zooms = torch.min(torch.abs(z) / torch.max(torch.abs(x), torch.abs(y)), dim=1, keepdim=True)[0] * zoom_tos
        z = z / zooms
        vertices = torch.stack([x, y, z], dim=2)
        if zoom_tos is None:
            return vertices
        else:
            return vertices, zooms


class RenderType:
    RGB = 0
    Silhouette = 1
    Depth = 2
    Normal = 3


class RenderFunction(Function):

    @staticmethod
    def torch2numpy(torch_tensor):
        return torch_tensor.detach().cpu().numpy()

    @staticmethod
    def chainer2numpy(chainer_tensor):
        return chainer.cuda.to_cpu(chainer_tensor)

    @staticmethod
    def torch2chainer(torch_tensor):
        device = torch_tensor.get_device()
        return chainer.cuda.to_gpu(RenderFunction.torch2numpy(torch_tensor), device=device)

    @staticmethod
    def chainer2torch(chainer_tensor):
        device = chainer.cuda.get_device_from_array(chainer_tensor).id
        return torch.Tensor(RenderFunction.chainer2numpy(chainer_tensor))

    @staticmethod
    def forward(ctx, vertices, faces, textures, renderer, render_type, eye, camera_mode, camera_direction, camera_up):
        _vertices = chainer.Variable(RenderFunction.torch2chainer(vertices))
        _faces = chainer.Variable(RenderFunction.torch2chainer(faces))
        _textures = None
        _eye = chainer.Variable(RenderFunction.torch2chainer(eye))
        _camera_direction = chainer.Variable(RenderFunction.torch2chainer(camera_direction))
        _camera_up = chainer.Variable(RenderFunction.torch2chainer(camera_up))
        if render_type == RenderType.RGB:
            _textures = chainer.Variable(RenderFunction.torch2chainer(textures))
        renderer.eye = _eye
        renderer.camera_mode = camera_mode
        renderer.camera_direction = _camera_direction
        renderer.up = _camera_up
        if render_type == RenderType.RGB:
            _images = renderer.render(_vertices, _faces, _textures)
        elif render_type == RenderType.Silhouette:
            _images = renderer.render_silhouettes(_vertices, _faces)
            _images = chainer.functions.expand_dims(_images, axis=1)
        elif render_type == RenderType.Depth:
            _images = renderer.render_depth(_vertices, _faces)
            _images = chainer.functions.expand_dims(_images, axis=1)
        elif render_type == RenderType.Normal:
            _images = renderer.render_normal(_vertices, _faces)
        ctx._vertices = _vertices
        ctx._textures = _textures
        ctx._render_type = render_type
        ctx._images = _images
        images = RenderFunction.chainer2torch(_images.data)
        return images

    @staticmethod
    def backward(ctx, grad_images):
        _grad_images = chainer.Variable(RenderFunction.torch2chainer(grad_images.data))
        ctx._images.grad_var = _grad_images
        ctx._images.backward()
        grad_vertices = None
        if ctx.needs_input_grad[0]:
            grad_vertices = RenderFunction.chainer2torch(ctx._vertices.grad_var.data)
        grad_textures = None
        if ctx.needs_input_grad[2] and ctx._render_type == RenderType.RGB:
            grad_textures = RenderFunction.chainer2torch(ctx._textures.grad_var.data)
        return grad_vertices, None, grad_textures, None, None, None, None, None, None


class Renderer(Module):

    def __init__(self, image_size=256, viewing_angle=30):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.viewing_angle = viewing_angle
        self.eye = Tensor([0, 0, 0])
        self.camera_mode = 'look'
        self.camera_direction = Tensor([0, 0, -1])
        self.camera_up = Tensor([0, 1, 0])

    def forward(self, vertices, faces, textures=None, render_type=RenderType.RGB):
        _renderer = _Renderer()
        _renderer.image_size = self.image_size
        _renderer.viewing_angle = self.viewing_angle
        vertices = vertices * Tensor([-1, 1, 1])
        eye = self.eye
        camera_mode = self.camera_mode
        camera_direction = self.camera_direction
        camera_up = self.camera_up
        batch_size = len(vertices)
        eye = eye[(None), :].expand(batch_size, -1)
        camera_direction = camera_direction[(None), :].expand(batch_size, -1)
        camera_up = camera_up[(None), :].expand(batch_size, -1)
        images = RenderFunction.apply(vertices, faces, textures, _renderer, render_type, eye, camera_mode, camera_direction, camera_up)
        if render_type == RenderType.Normal:
            x, y, z = torch.unbind(images, dim=1)
            images = torch.stack([-x, y, z], dim=1)
        return images


class TargetType:
    geometry = 1 << 0
    reproject = 1 << 1
    normal = 1 << 2
    depth = 1 << 3
    pretrain = geometry
    finetune = reproject
    full = geometry | reproject
    extend = geometry | reproject | normal | depth


class Derenderer3d(Module):

    def __init__(self, mode, image_size, render_size):
        super(Derenderer3d, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.render_size = render_size
        self.derenderer = Derenderer()
        self._force_no_sample = False
        if mode & TargetType.reproject:
            self.objs = [ShapenetObj(class_id='02958343', obj_id='137f67657cdc9da5f985cd98f7d73e9a'), ShapenetObj(class_id='02958343', obj_id='5343e944a7753108aa69dfdc5532bb13'), ShapenetObj(class_id='02958343', obj_id='3776e4d1e2587fd3253c03b7df20edd5'), ShapenetObj(class_id='02958343', obj_id='3ba5bce1b29f0be725f689444c7effe2'), ShapenetObj(class_id='02958343', obj_id='53a031dd120e81dc3aa562f24645e326'), ShapenetObj(class_id='02924116', obj_id='7905d83af08a0ca6dafc1d33c05cbcf8'), ShapenetObj(class_id='02958343', obj_id='a0fe4aac120d5f8a5145cad7315443b3'), ShapenetObj(class_id='02958343', obj_id='cd7feedd6041209131ac5fb37e6c8324')]
            self.ffds = [FFD(obj.vertices, constraints=[FFD.Constraint.symmetry(axis=FFD.Constraint.Axis.z), FFD.Constraint.homogeneity(axis=FFD.Constraint.Axis.y, index=[0, 1])]) for obj in self.objs]
            self.perspective_transform = PerspectiveTransform()
            self.renderer = Renderer(image_size=render_size)

    def forward(self, images, roi_norms, focals):
        _mroi_norms = torch.stack([roi_norms[:, (2)] + roi_norms[:, (0)], roi_norms[:, (3)] + roi_norms[:, (1)]], dim=1) / 2.0
        _droi_norms = torch.stack([roi_norms[:, (2)] - roi_norms[:, (0)], roi_norms[:, (3)] - roi_norms[:, (1)]], dim=1)
        _blob = {'_roi_norms': roi_norms, '_mroi_norms': _mroi_norms, '_droi_norms': _droi_norms, '_focals': focals}
        _blob.update(self.derenderer(images, _mroi_norms, _droi_norms))
        if not self.mode & TargetType.reproject:
            return _blob
        _blob.update(self.render(_blob))
        return _blob

    def render(self, blob):
        _mroi_norms = blob['_mroi_norms']
        _droi_norms = blob['_droi_norms']
        _focals = blob['_focals']
        _theta_deltas = blob['_theta_deltas']
        _translation2ds = blob['_translation2ds']
        _log_scales = blob['_log_scales']
        _log_depths = blob['_log_depths']
        _class_probs = blob['_class_probs']
        _ffd_coeffs = blob['_ffd_coeffs']
        batch_size = len(_focals)
        _thetas = torch.unsqueeze(torch.atan2(_theta_deltas[:, (1)], _theta_deltas[:, (0)]), dim=1)
        _rotations = torch.cat([torch.cos(_thetas / 2), torch.zeros(batch_size, 1), torch.sin(_thetas / 2), torch.zeros(batch_size, 1)], dim=1)
        _areas = torch.unsqueeze(_droi_norms[:, (0)] * _droi_norms[:, (1)], dim=1)
        _scales = torch.exp(_log_scales)
        _depths = torch.sqrt(torch.exp(_log_depths) / _areas)
        _center2ds = _mroi_norms + _translation2ds * _droi_norms
        _translation_units = torch.stack([_center2ds[:, (1)], -_center2ds[:, (0)], -torch.ones(batch_size)], dim=1)
        _translation_units = _translation_units / torch.norm(_translation_units, p=2, dim=1, keepdim=True)
        _translations = _depths * _translation_units
        _alphas = -(_thetas - torch.atan(_translations[:, 0:1] / _translations[:, 2:3]))
        _alphas = torch.remainder(_alphas + np.pi, 2 * np.pi) - np.pi
        if self.training and not self._force_no_sample:
            _class_dists = Categorical(_class_probs)
            _class_samples = _class_dists.sample()
            _class_log_probs = _class_dists.log_prob(_class_samples)
        else:
            _class_max_probs, _class_samples = torch.max(_class_probs, dim=1)
            _class_log_probs = torch.log(_class_max_probs)
        if self.training:
            _perspective_translation_units = torch.stack([_mroi_norms[:, (1)], -_mroi_norms[:, (0)], -torch.ones(batch_size)], dim=1)
            _perspective_translation_units = _perspective_translation_units / torch.norm(_perspective_translation_units, p=2, dim=1, keepdim=True)
            _perspective_translations = _depths * _perspective_translation_units
            _zooms = self.image_size / _focals / torch.max(_droi_norms, dim=1, keepdim=True)[0]
        else:
            _zoom_tos = self.render_size / (2.0 * _focals)
            _zooms = []
        device = torch.cuda.current_device()
        _masks = []
        _normals = []
        _depth_maps = []
        for size in range(batch_size):
            None
            sys.stdout.flush()
            _class_sample = int(_class_samples[size])
            _ffd_coeff = _ffd_coeffs[size][_class_sample]
            vertices = self.ffds[_class_sample](_ffd_coeff)
            faces = self.objs[_class_sample].faces
            __vertices = vertices.unsqueeze(dim=0)
            __faces = faces.unsqueeze(dim=0)
            __scales = _scales[size].unsqueeze(dim=0)
            __rotations = _rotations[size].unsqueeze(dim=0)
            __translations = _translations[size].unsqueeze(dim=0)
            if self.training:
                __perspective_translations = _perspective_translations[size].unsqueeze(dim=0)
                __zooms = _zooms[size].unsqueeze(dim=0)
                __vertices = self.perspective_transform(__vertices, scales=__scales, rotations=__rotations, translations=__translations, perspective_translations=__perspective_translations, zooms=__zooms)
            else:
                __zoom_tos = _zoom_tos[size].unsqueeze(dim=0)
                __vertices, __zooms = self.perspective_transform(__vertices, scales=__scales, rotations=__rotations, translations=__translations, perspective_translations=__translations, zoom_tos=__zoom_tos)
                _zooms.append(__zooms)
            self.renderer.viewing_angle = np.arctan(self.render_size / (2.0 * _focals[size].item())) / np.pi * 180
            __masks = self.renderer(vertices=__vertices, faces=__faces, render_type=RenderType.Silhouette)
            _masks.append(__masks)
            if self.mode & TargetType.normal:
                __normals = self.renderer(vertices=__vertices, faces=__faces, render_type=RenderType.Normal)
                _normals.append(__normals)
            if self.mode & TargetType.depth:
                __depth_maps = self.renderer(vertices=__vertices, faces=__faces, render_type=RenderType.Depth)
                _depth_maps.append(__depth_maps)
        if not self.training:
            _zooms = torch.cat(_zooms, dim=0)
        _masks = torch.cat(_masks, dim=0)
        if self.mode & TargetType.normal:
            _normals = torch.cat(_normals, dim=0)
        if self.mode & TargetType.depth:
            _depth_maps = torch.cat(_depth_maps, dim=0)
        return {'_thetas': _thetas, '_alphas': _alphas, '_rotations': _rotations, '_scales': _scales, '_depths': _depths, '_center2ds': _center2ds, '_translations': _translations, '_class_log_probs': _class_log_probs, '_zooms': _zooms, '_masks': _masks, '_normals': _normals, '_depth_maps': _depth_maps}


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = (out_width - 1) * self.stride[0] + self.kernel_size[0] - in_width
        pad_along_height = (out_height - 1) * self.stride[1] + self.kernel_size[1] - in_height
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


class TopDownLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, y):
        y = F.upsample(y, scale_factor=2)
        x = self.conv1(x)
        return self.conv2(self.padding2(x + y))


class FPN(nn.Module):

    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1), nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1), nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1), nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1), nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)
        p6_out = self.P6(p5_out)
        return [p2_out, p3_out, p4_out, p5_out, p6_out]


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


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


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
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
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
        assert results[0][0] == 0, 'The first result should belongs to the master.'
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


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.001, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None
        self._moving_average_fraction = 1.0 - momentum
        self.register_buffer('_tmp_running_mean', torch.zeros(self.num_features))
        self.register_buffer('_tmp_running_var', torch.ones(self.num_features))
        self.register_buffer('_running_iter', torch.ones(1))
        self._tmp_running_mean = self.running_mean.clone() * self._running_iter
        self._tmp_running_var = self.running_var.clone() * self._running_iter

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))
        return outputs

    def _add_weighted(self, dest, delta, alpha=1, beta=1, bias=0):
        """return *dest* by `dest := dest*alpha + delta*beta + bias`"""
        return dest * alpha + delta * beta + bias

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        self._tmp_running_mean = self._add_weighted(self._tmp_running_mean, mean.data, alpha=self._moving_average_fraction)
        self._tmp_running_var = self._add_weighted(self._tmp_running_var, unbias_var.data, alpha=self._moving_average_fraction)
        self._running_iter = self._add_weighted(self._running_iter, 1, alpha=self._moving_average_fraction)
        self.running_mean = self._tmp_running_mean / self._running_iter
        self.running_var = self._tmp_running_var / self._running_iter
        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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
        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = SynchronizedBatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = SynchronizedBatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), SynchronizedBatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth
        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.conv_shared(self.padding(x)))
        rpn_class_logits = self.conv_class(x)
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)
        rpn_probs = self.softmax(rpn_class_logits)
        rpn_bbox = self.conv_bbox(x)
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)
        return [rpn_class_logits, rpn_probs, rpn_bbox]


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)
        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(image, boxes, box_ind, self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _backend.crop_and_resize_forward(image, boxes, box_ind, self.extrapolation_value, self.crop_height, self.crop_width, crops)
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)
        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)
        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(grad_outputs, boxes, box_ind, grad_image)
        else:
            _backend.crop_and_resize_backward(grad_outputs, boxes, box_ind, grad_image)
        return grad_image, None, None


def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2
    return torch.log(x) / ln2


def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)
    boxes = inputs[0]
    feature_maps = inputs[1:]
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, (0)]
        level_boxes = boxes[(ix.data), :]
        box_to_level.append(ix.data)
        level_boxes = level_boxes.detach()
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind
        feature_maps[i] = feature_maps[i].unsqueeze(0)
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)
    pooled = torch.cat(pooled, dim=0)
    box_to_level = torch.cat(box_to_level, dim=0)
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[(box_to_level), :, :]
    return pooled


class Classifier(nn.Module):

    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)
        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)
        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):

    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array([image_id] + list(image_shape) + list(window) + list(active_class_ids))
    return meta


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    if target_class_ids.size():
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, (0)]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)
        target_bbox = target_bbox[(indices[:, (0)].data), :]
        pred_bbox = pred_bbox[(indices[:, (0)].data), (indices[:, (1)].data), :]
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    if target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size():
        positive_ix = torch.nonzero(target_class_ids > 0)[:, (0)]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        y_true = target_masks[(indices[:, (0)].data), :, :]
        y_pred = pred_masks[(indices[:, (0)].data), (indices[:, (1)].data), :, :]
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    rpn_match = rpn_match.squeeze(2)
    indices = torch.nonzero(rpn_match == 1)
    rpn_bbox = rpn_bbox[indices.data[:, (0)], indices.data[:, (1)]]
    target_bbox = target_bbox[(0), :rpn_bbox.size()[0], :]
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)
    return loss


def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    rpn_match = rpn_match.squeeze(2)
    anchor_class = (rpn_match == 1).long()
    indices = torch.nonzero(rpn_match != 0)
    rpn_class_logits = rpn_class_logits[(indices.data[:, (0)]), (indices.data[:, (1)]), :]
    anchor_class = anchor_class[indices.data[:, (0)], indices.data[:, (1)]]
    loss = F.cross_entropy(rpn_class_logits, anchor_class)
    return loss


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask):
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]


def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, (0)]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    height = boxes[:, (2)] - boxes[:, (0)]
    width = boxes[:, (3)] - boxes[:, (1)]
    center_y = boxes[:, (0)] + 0.5 * height
    center_x = boxes[:, (1)] + 0.5 * width
    center_y += deltas[:, (0)] * height
    center_x += deltas[:, (1)] * width
    height *= torch.exp(deltas[:, (2)])
    width *= torch.exp(deltas[:, (3)])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, (0)] = boxes[:, (0)].clamp(float(window[0]), float(window[2]))
    boxes[:, (1)] = boxes[:, (1)].clamp(float(window[1]), float(window[3]))
    boxes[:, (2)] = boxes[:, (2)].clamp(float(window[0]), float(window[2]))
    boxes[:, (3)] = boxes[:, (3)].clamp(float(window[1]), float(window[3]))
    return boxes


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def pth_nms(dets, thresh):
    """
    dets has to be a tensor
    """
    if not dets.is_cuda:
        x1 = dets[:, (1)]
        y1 = dets[:, (0)]
        x2 = dets[:, (3)]
        y2 = dets[:, (2)]
        scores = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.cpu_nms(keep, num_out, dets, order, areas, thresh)
        return keep[:num_out[0]]
    else:
        x1 = dets[:, (1)]
        y1 = dets[:, (0)]
        x2 = dets[:, (3)]
        y2 = dets[:, (2)]
        scores = dets[:, (4)]
        dets_temp = torch.FloatTensor(dets.size())
        dets_temp[:, (0)] = dets[:, (1)]
        dets_temp[:, (1)] = dets[:, (0)]
        dets_temp[:, (2)] = dets[:, (3)]
        dets_temp[:, (3)] = dets[:, (2)]
        dets_temp[:, (4)] = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        dets = dets[order].contiguous()
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.gpu_nms(keep, num_out, dets_temp, thresh)
        return order[keep[:num_out[0]]].contiguous()


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.
    Accept dets as tensor"""
    return pth_nms(dets, thresh)


def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    _, class_ids = torch.max(probs, dim=1)
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale
    refined_rois *= scale
    refined_rois = clip_to_window(window, refined_rois)
    refined_rois = torch.round(refined_rois)
    keep_bool = class_ids > 0
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, (0)]
    pre_nms_class_ids = class_ids[keep.data]
    pre_nms_scores = class_scores[keep.data]
    pre_nms_rois = refined_rois[keep.data]
    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, (0)]
        ix_rois = pre_nms_rois[ixs.data]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[(order.data), :]
        class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        class_keep = keep[ixs[order[class_keep].data].data]
        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]
    result = torch.cat((refined_rois[keep.data], class_ids[keep.data].unsqueeze(1).float(), class_scores[keep.data].unsqueeze(1)), dim=1)
    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """
    rois = rois.squeeze(0)
    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)
    return detections


def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, (0)]
    x1 = torch.max(b1_x1, b2_x1)[:, (0)]
    y2 = torch.min(b1_y2, b2_y2)[:, (0)]
    x2 = torch.min(b1_x2, b2_x2)[:, (0)]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, (0)] + b2_area[:, (0)] - intersection
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    if (gt_class_ids < 0).any():
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, (0)]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, (0)]
        crowd_boxes = gt_boxes[(crowd_ix.data), :]
        crowd_masks = gt_masks[(crowd_ix.data), :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[(non_crowd_ix.data), :]
        gt_masks = gt_masks[(non_crowd_ix.data), :]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool
    overlaps = bbox_overlaps(proposals, gt_boxes)
    roi_iou_max = torch.max(overlaps, dim=1)[0]
    positive_roi_bool = roi_iou_max >= 0.5
    if positive_roi_bool.any():
        positive_indices = torch.nonzero(positive_roi_bool)[:, (0)]
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[(positive_indices.data), :]
        positive_overlaps = overlaps[(positive_indices.data), :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[(roi_gt_box_assignment.data), :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev
        deltas /= std_dev
        roi_masks = gt_masks[(roi_gt_box_assignment.data), :, :]
        boxes = positive_rois
        if config.USE_MINI_MASK:
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT:
            box_ids = box_ids
        masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)
        masks = masks.squeeze(1)
        masks = torch.round(masks)
    else:
        positive_count = 0
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    if negative_roi_bool.any() and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, (0)]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[(negative_indices.data), :]
    else:
        negative_count = 0
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros
        roi_gt_class_ids = torch.cat([roi_gt_class_ids.int(), zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        masks = zeros
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois
            roi_gt_class_ids = roi_gt_class_ids
            deltas = deltas
            masks = masks
    return rois, roi_gt_class_ids, deltas, masks


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += 'shape: {:20}  min: {:10.5f}  max: {:10.5f}'.format(str(array.shape), array.min() if array.size else '', array.max() if array.size else '')
    None


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    None
    if iteration == total:
        None


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack([boxes[:, (0)].clamp(float(window[0]), float(window[2])), boxes[:, (1)].clamp(float(window[1]), float(window[3])), boxes[:, (2)].clamp(float(window[0]), float(window[2])), boxes[:, (3)].clamp(float(window[1]), float(window[3]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)
    scores = inputs[0][:, (1)]
    deltas = inputs[1]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev
    deltas = deltas * std_dev
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[(order.data), :]
    anchors = anchors[(order.data), :]
    boxes = apply_box_deltas(anchors, deltas)
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[(keep), :]
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm
    normalized_boxes = boxes / norm
    normalized_boxes = normalized_boxes.unsqueeze(0)
    return normalized_boxes


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception('Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc. ')
        resnet = ResNet('resnet101', stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, config.BACKBONE_SHAPES, config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        self.epoch = 0
        now = datetime.datetime.now()
        if model_path:
            regex = '.*/\\w+(\\d{4})(\\d{2})(\\d{2})T(\\d{2})(\\d{2})/mask\\_rcnn\\_\\w+(\\d{4})\\.pth'
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))
        self.log_dir = os.path.join(self.model_dir, '{}{:%Y%m%dT%H%M}'.format(self.config.NAME.lower(), now))
        self.checkpoint_path = os.path.join(self.log_dir, 'mask_rcnn_{}_*epoch*.pth'.format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace('*epoch*', '{:04d}')

    def load_last_model(self):
        """Find the last checkpoint file of the model directory, and load the weights.
        """
        checkpoints = next(os.walk(self.model_dir))[2]
        checkpoints = filter(lambda f: f.startswith('mask_rcnn'), checkpoints)
        checkpoints = sorted(checkpoints)
        if checkpoints:
            checkpoint = os.path.join(self.model_dir, checkpoints[-1])
            self.load_weights(checkpoint)
            None

    def find_last(self):
        """Find the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith('mask_rcnn'), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, transfer=0):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            load_dict = torch.load(filepath)
            if transfer:
                remove_list = ['classifier.linear_bbox.bias', 'classifier.linear_bbox.weight', 'classifier.linear_class.bias', 'classifier.linear_class.weight', 'mask.conv5.weight', 'mask.conv5.bias']
                for remove_weight in remove_list:
                    load_dict.pop(remove_weight)
                state_dict = self.state_dict()
                state_dict.update(load_dict)
                self.load_state_dict(state_dict)
            else:
                self.load_state_dict(load_dict)
        else:
            None
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        molded_images, image_metas, windows = self.mold_inputs(images)
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()
        if self.config.GPU_COUNT:
            molded_images = molded_images
        molded_images = Variable(molded_images, volatile=True)
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(detections[i], mrcnn_mask[i], image.shape, windows[i])
            results.append({'rois': final_rois, 'class_ids': final_class_ids, 'scores': final_scores, 'masks': final_masks})
        return results

    def generate_inst_maps(self, images):
        """Generate instance maps out of images.

        images: of shape (N, H, W, 3).

        Returns:
        inst_tensors: (N, H, W) numpy instance maps. Each (H, W)instance map maps pixel to 0 (background), 1, 2, 3, etc.
        """
        inst_tensors = []
        for n in range(images.shape[0]):
            detection = self.detect([images[n]])[0]
            masks = detection['masks']
            inst_tensor = np.zeros(masks.shape[:2])
            for i in range(masks.shape[2]):
                inst_tensor = np.maximum(inst_tensor, (i + 1) * masks[:, :, (i)].astype(int))
            inst_tensors.append(inst_tensor)
        inst_tensors = np.stack(inst_tensors, axis=0)
        return inst_tensors

    def generate_inst_map(self, images, return_more=False):
        """Generate instance maps out of images.

        images: List of images. Each image is of shape (H, W, 3).

        Returns:
        inst_tensors: [#images, H, W] numpy instance maps. Each [H, W] instance map maps pixel to 0 (background), 1, 2, 3, etc.
        """
        detection = self.detect(images)
        inst_tensors = []
        d = {}
        for image_dict in detection:
            masks = image_dict['masks']
            h = masks.shape[0]
            w = masks.shape[1]
            inst_tensor = np.zeros((h, w))
            for i in range(masks.shape[2]):
                inst_tensor = np.maximum(inst_tensor, (i + 1) * masks[:, :, (i)].astype(int))
                if return_more:
                    d[int(i + 1)] = {'rois': image_dict['rois'][i].tolist(), 'class_id': int(image_dict['class_ids'][i])}
            inst_tensors.append(inst_tensor)
        inst_tensors = np.stack(inst_tensors, axis=0)
        if return_more:
            return inst_tensors, d, detection
        else:
            return inst_tensors

    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]
        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.apply(set_bn_eval)
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == 'training' else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox], proposal_count=proposal_count, nms_threshold=self.config.RPN_NMS_THRESHOLD, anchors=self.anchors, config=self.config)
        if mode == 'inference':
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale
            detection_boxes = detections[:, :4] / scale
            detection_boxes = detection_boxes.unsqueeze(0)
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask]
        elif mode == 'training':
            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale
            gt_boxes = gt_boxes / scale
            rois, target_class_ids, target_deltas, target_mask = detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)
            if not rois.size():
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits
                    mrcnn_class = mrcnn_class
                    mrcnn_bbox = mrcnn_bbox
                    mrcnn_mask = mrcnn_mask
            else:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)
            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        transfer = False
        if layers == 'transfer':
            transfer = True
            layers = 'heads'
        layer_regex = {'heads': '(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)', '3+': '(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)', '4+': '(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)', '5+': '(fpn.C5.*)|(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)', 'all': '.*'}
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
        train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=16)
        val_set = Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=16)
        log('\nStarting at epoch {}. LR={}\n'.format(self.epoch + 1, learning_rate))
        log('Checkpoint Path: {}'.format(self.checkpoint_path))
        self.set_trainable(layers)
        trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        if transfer:
            transfer_layers = '(mask.conv5.*)|(classifier.linear_class.*)|(classifier.linear_bbox.*)'
            trainables_transfer = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name and re.fullmatch(transfer_layers, name)]
            trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name and not re.fullmatch(transfer_layers, name)]
            optimizer = optim.SGD([{'params': trainables_transfer, 'lr': 0.01, 'weight_decay': self.config.WEIGHT_DECAY}, {'params': trainables_wo_bn, 'lr': learning_rate, 'weight_decay': self.config.WEIGHT_DECAY}, {'params': trainables_only_bn, 'lr': learning_rate}], momentum=self.config.LEARNING_MOMENTUM)
        else:
            optimizer = optim.SGD([{'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY}, {'params': trainables_only_bn}], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
        for epoch in range(self.epoch + 1, epochs + 1):
            log('Epoch {}/{}.'.format(epoch, epochs))
            loss = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)
            val_loss = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)
            self.loss_history.append(loss)
            self.val_loss_history.append(val_loss)
            visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            if epoch % 5 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))
        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        step = 0
        for inputs in datagenerator:
            batch_count += 1
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            image_metas = image_metas.numpy()
            images = Variable(images)
            rpn_match = Variable(rpn_match)
            rpn_bbox = Variable(rpn_bbox)
            gt_class_ids = Variable(gt_class_ids)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)
            if self.config.GPU_COUNT:
                images = images
                rpn_match = rpn_match
                rpn_bbox = rpn_bbox
                gt_class_ids = gt_class_ids
                gt_boxes = gt_boxes
                gt_masks = gt_masks
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss
            if batch_count % self.config.BATCH_SIZE == 0:
                optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
            if batch_count % self.config.BATCH_SIZE == 0:
                optimizer.step()
                batch_count = 0
            if (step + 1) % self.config.DISPLAY_SIZE == 0:
                printProgressBar(step + 1, steps, prefix='\t{}/{}'.format(step + 1, steps), suffix='Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}'.format(loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0], mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0], mrcnn_mask_loss.data.cpu()[0]), length=10)
            loss_sum += loss.data.cpu()[0] / steps
            if step == steps - 1:
                break
            step += 1
        return loss_sum

    def valid_epoch(self, datagenerator, steps):
        step = 0
        loss_sum = 0
        for inputs in datagenerator:
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            image_metas = image_metas.numpy()
            images = Variable(images, volatile=True)
            rpn_match = Variable(rpn_match, volatile=True)
            rpn_bbox = Variable(rpn_bbox, volatile=True)
            gt_class_ids = Variable(gt_class_ids, volatile=True)
            gt_boxes = Variable(gt_boxes, volatile=True)
            gt_masks = Variable(gt_masks, volatile=True)
            if self.config.GPU_COUNT:
                images = images
                rpn_match = rpn_match
                rpn_bbox = rpn_bbox
                gt_class_ids = gt_class_ids
                gt_boxes = gt_boxes
                gt_masks = gt_masks
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')
            if not target_class_ids.size():
                continue
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss
            printProgressBar(step + 1, steps, prefix='\t{}/{}'.format(step + 1, steps), suffix='Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}'.format(loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0], mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0], mrcnn_mask_loss.data.cpu()[0]), length=10)
            loss_sum += loss.data.cpu()[0] / steps
            if step == steps - 1:
                break
            step += 1
        return loss_sum

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding = utils.resize_image(image, min_dim=self.config.IMAGE_MIN_DIM, max_dim=self.config.IMAGE_MAX_DIM, padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            image_meta = compose_image_meta(0, image.shape, window, np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        zero_ix = np.where(detections[:, (4)] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        boxes = detections[:N, :4]
        class_ids = detections[:N, (4)].astype(np.int32)
        scores = detections[:N, (5)]
        masks = mrcnn_mask[(np.arange(N)), :, :, (class_ids)]
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
        exclude_ix = np.where((boxes[:, (2)] - boxes[:, (0)]) * (boxes[:, (3)] - boxes[:, (1)]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]
        full_masks = []
        for i in range(N):
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty((0,) + masks.shape[1:3])
        return boxes, class_ids, scores, full_masks


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]
        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)
            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)
            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)
        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(featuremap, boxes, box_ind)


class Model(Derenderer3d):

    def __init__(self):
        super(Model, self).__init__(mode=FLAGS.mode, image_size=FLAGS.image_size, render_size=FLAGS.render_size)


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    """Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    """Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def dict_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU), with dictionary support.
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, Variable):
            if out.dim() == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            return Gather.apply(target_device, dim, *outputs)
        elif out is None:
            return None
        elif isinstance(out, collections.Mapping):
            return {k: gather_map([o[k] for o in outputs]) for k in out}
        elif isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)


class DictGatherDataParallel(nn.DataParallel):

    def gather(self, outputs, output_device):
        return dict_gather(outputs, output_device, dim=self.dim)


def _get_stream(device):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * cuda.device_count()
    if _streams[device] is None:
        _streams[device] = cuda.Stream(device)
    return _streams[device]


def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        obj = Variable(obj)
    if isinstance(obj, Variable):
        v = obj
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj


def _async_copy_stream(inputs, device_ids):
    nr_devs = len(device_ids)
    assert type(inputs) in (tuple, list)
    assert len(inputs) == nr_devs
    outputs = []
    streams = [_get_stream(d) for d in device_ids]
    for i, dev, stream in zip(inputs, device_ids, streams):
        with cuda.device(dev):
            main_stream = cuda.current_stream()
            with cuda.stream(stream):
                outputs.append(async_copy_to(i, dev, main_stream=main_stream))
            main_stream.wait_stream(stream)
    return outputs


class UserScatteredDataParallel(DictGatherDataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        assert len(inputs) == 1
        inputs = inputs[0]
        inputs = _async_copy_stream(inputs, device_ids)
        inputs = [[i] for i in inputs]
        assert len(kwargs) == 0
        kwargs = [{} for _ in range(len(inputs))]
        return inputs, kwargs


class SegmentationModuleBase(nn.Module):

    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):

    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        if segSize is None:
            if self.deep_sup_scale is not None:
                pred, pred_deepsup = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=False))
            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale
            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data']), segSize=segSize)
            return pred


class Resnet(nn.Module):

    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()
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

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


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

    def forward(self, x, return_feature_maps=False):
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
        return [x]


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(conv3x3(in_planes, out_planes, stride), SynchronizedBatchNorm2d(out_planes), nn.ReLU(inplace=True))


class C1BilinearDeepSup(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1BilinearDeepSup, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class C1Bilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.use_softmax = use_softmax
        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, x, segSize=None):
        conv5 = x[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinear, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class PPMBilinearDeepsup(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMBilinearDeepsup, self).__init__()
        self.use_softmax = use_softmax
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(nn.AdaptiveAvgPool2d(scale), nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True)))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)
        self.conv_last = nn.Sequential(nn.Conv2d(fc_dim + len(pool_scales) * 512, 512, kernel_size=3, padding=1, bias=False), SynchronizedBatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_class, kernel_size=1))
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if self.use_softmax:
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
            x = nn.functional.softmax(x, dim=1)
            return x
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)
        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)
        return x, _


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
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
        out += residual
        out = self.relu(out)
        return out


class BaseModel(torch.nn.Module):

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            None
            if network_label == 'G':
                raise 'Generator must exist!'
        else:
            None
            pretrained_dict = torch.load(save_path)
            if next(iter(pretrained_dict.keys()))[:6] == 'module':
                new_pretrained_dict = OrderedDict()
                for k, v in pretrained_dict.items():
                    new_pretrained_dict[k[7:]] = v
                pretrained_dict = new_pretrained_dict
            try:
                network.load_state_dict(pretrained_dict)
            except:
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    None
                except:
                    None
                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        not_initialized = Set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    None
                    network.load_state_dict(model_dict)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = self.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.copy_(torch.rand(batchSize, nz) * 2.0 - 1.0)
        elif random_type == 'gauss':
            z.copy_(torch.randn(batchSize, nz))
            z = Variable(z)
        return z

    def print_networks(self, verbose):
        None
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    None
                None
        None

    def update_learning_rate():
        pass


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = self.real_label_var is None or self.real_label_var.numel() != input.numel()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = self.fake_label_var is None or self.fake_label_var.numel() != input.numel()
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class LocalEnhancer(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        ngf_global = ngf * 2 ** n_local_enhancers
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = nn.Sequential(*model_global)
        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True), nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        output_prev = self.model(input_downsampled[-1])
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class Encoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d, isTrain=True):
        super(Encoder, self).__init__()
        self.isTrain = isTrain
        self.output_nc = output_nc
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)
        outputs_mean = outputs.clone()
        batchSize = inst.size(0)
        for i in range(batchSize):
            inst[i] = inst[i] * batchSize + i
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            i = int(i)
            assert np.any(inst == i), (inst_list, inst)
            indices = (inst == i).nonzero()
            for j in range(self.output_nc):
                output_ins = outputs[indices[:, (0)], indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]]
                mean_feat = torch.mean(output_ins).expand_as(output_ins)
                outputs_mean[indices[:, (0)], indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]] = mean_feat
        return (outputs_mean[:, :self.output_nc, :, :], 0) if self.isTrain else outputs_mean[:, :self.output_nc, :, :]

    def generate_feat_dict(self, input, inst):
        outputs = self.model(input)
        batchSize = inst.size(0)
        for i in range(batchSize):
            inst[i] = inst[i] * batchSize + i
        feat_dict = {}
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            i = int(i)
            feat_dict[i] = []
            indices = (inst == i).nonzero()
            for j in range(self.output_nc):
                output_ins = outputs[indices[:, (0)], indices[:, (1)] + j, indices[:, (2)], indices[:, (3)]]
                mean_feat = torch.mean(output_ins)
                feat_dict[i] += [float(mean_feat)]
        return feat_dict


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


class Pix2PixHDModel(BaseModel):

    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        self.no_global_encoder = opt.no_global_encoder
        input_nc = opt.label_nc if opt.label_nc != 0 else 3
        self.model_names = []
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        if opt.feat_pose:
            netG_input_nc += opt.feat_pose_num_bins + 1 if opt.feat_pose_num_bins else 2
        if opt.feat_normal:
            netG_input_nc += 3
        if opt.feat_depth:
            netG_input_nc += 1
        if not opt.no_global_encoder:
            netG_input_nc += opt.global_encoder_nz
        None
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.model_names += ['G']
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.model_names += ['D']
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids, isTrain=opt.isTrain)
            self.model_names += ['E']
        self.print_networks(verbose=True)
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
            if not self.no_global_encoder:
                self.load_network(self.netGlobalE, 'GlobalE', opt.which_epoch, pretrained_path)
        if self.isTrain:
            if opt.pool_size > 0 and len(self.gpu_ids) > 1:
                raise NotImplementedError('Fake Pool Not Implemented for MultiGPU')
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'G_L1', 'E_VAE', 'E_regress']
            if opt.niter_fix_global > 0:
                None
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params': [value], 'lr': opt.lr}]
                    else:
                        params += [{'params': [value], 'lr': 0.0}]
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            if not opt.no_global_encoder:
                params += list(self.netGlobalE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, pose_map=None, normal_map=None, depth_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data
        else:
            size = label_map.size()
            oneHot_size = size[0], self.opt.label_nc, size[2], size[3]
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long(), 1.0)
        if not self.opt.no_instance:
            inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        with torch.no_grad():
            input_label = Variable(input_label)
        if real_image is not None:
            real_image = Variable(real_image.data)
        if self.use_features:
            if self.opt.load_features:
                feat_map = Variable(feat_map.data)
            if self.opt.feat_pose:
                if self.opt.feat_pose_num_bins:
                    size = pose_map.size()
                    oneHot_size = size[0], self.opt.feat_pose_num_bins + 1, size[2], size[3]
                    _pose_map = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
                    pose_map = _pose_map.scatter_(1, pose_map.data.long(), 1.0)
                    pose_map = Variable(pose_map)
                else:
                    pose_map = Variable(pose_map.data)
            if self.opt.feat_normal:
                normal_map = Variable(normal_map.data)
            if self.opt.feat_depth:
                depth_map = Variable(depth_map.data)
        return input_label, inst_map, real_image, feat_map, pose_map, normal_map, depth_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, pose=None, normal=None, depth=None, infer=False):
        input_label, inst_map, real_image, feat_map, pose_map, normal_map, depth_map = self.encode_input(label, inst, image, feat, pose, normal, depth)
        input_concat = input_label
        if self.use_features:
            if not self.opt.load_features:
                feat_map, loss_E_VAE = self.netE.forward(real_image, inst_map)
                input_concat = torch.cat((input_concat, feat_map), dim=1)
            if self.opt.feat_pose:
                input_concat = torch.cat((input_concat, pose_map), dim=1)
            if self.opt.feat_normal:
                input_concat = torch.cat((input_concat, normal_map), dim=1)
            if self.opt.feat_depth:
                input_concat = torch.cat((input_concat, depth_map), dim=1)
            if not self.opt.no_global_encoder:
                mu, logvar = self.netGlobalE.forward(real_image)
                std = logvar.mul(0.5).exp_()
                eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
                z = eps.mul(std).add_(mu)
                z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), real_image.size(2), real_image.size(3))
                input_concat = torch.cat((input_concat, z), dim=1)
        fake_image = self.netG.forward(input_concat)
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        loss_G_L1 = 0
        if self.opt.lambda_L1 > 0:
            loss_G_L1 = self.criterionFeat(fake_image, real_image) * self.opt.lambda_L1
        loss_E_regress = 0
        if not self.opt.no_global_encoder:
            kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).mul_(-0.5)
            loss_E_VAE = loss_E_VAE + torch.sum(kl_element)
        loss_E_VAE = loss_E_VAE * self.opt.lambda_KL
        if isinstance(loss_E_VAE, float):
            loss_E_VAE = 0
        return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_L1, loss_E_VAE, loss_E_regress], None if not infer else fake_image]

    def fake_inference(self, image, label, inst, feat=None, pose=None, normal=None, depth=None):
        input_label, inst_map, real_image, feat_map, pose_map, normal_map, depth_map = self.encode_input(Variable(label), inst_map=Variable(inst), pose_map=Variable(pose), normal_map=Variable(normal), depth_map=Variable(depth), real_image=Variable(image), infer=True)
        if self.use_features:
            if feat is None:
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                feat_map = Variable(feat)
            input_concat = torch.cat((input_label, feat_map), dim=1)
            if self.opt.feat_pose:
                input_concat = torch.cat((input_concat, pose_map), dim=1)
            if self.opt.feat_normal:
                input_concat = torch.cat((input_concat, normal_map), dim=1)
            if self.opt.feat_depth:
                input_concat = torch.cat((input_concat, depth_map), dim=1)
            if not self.opt.no_global_encoder:
                mu, logvar = self.netGlobalE.forward(Variable(image))
                std = logvar.mul(0.5).exp_()
                eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
                z = eps.mul(std).add_(mu)
                z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), image.size(2), image.size(3))
                z = mu.view(mu.size(0), mu.size(1), 1, 1).expand(mu.size(0), mu.size(1), image.size(2), image.size(3))
                input_concat = torch.cat((input_concat, z), dim=1)
        else:
            input_concat = input_label
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def inference(self, label, inst):
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)
        if self.use_features:
            feat_map = self.sample_features(inst_map)
            feat_map = Variable(feat_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = torch.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 5000 else i // 5000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image, volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst)
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 5000 else i // 5000
            i = int(i)
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[(num // 2), :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        if not self.no_global_encoder:
            self.save_network(self.netGlobalE, 'GlobalE', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        None

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        None
        self.old_lr = lr


class UIModel(BaseModel):

    def name(self):
        return 'UIModel'

    def initialize(self, opt):
        assert not opt.isTrain
        BaseModel.initialize(self, opt)
        self.use_features = opt.instance_feat or opt.label_feat
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.load_network(self.netG, 'G', opt.which_epoch)
        None

    def toTensor(self, img, normalize=False):
        tensor = torch.from_numpy(np.array(img, np.int32, copy=False))
        tensor = tensor.view(1, img.size[1], img.size[0], len(img.mode))
        tensor = tensor.transpose(1, 2).transpose(1, 3).contiguous()
        if normalize:
            return (tensor.float() / 255.0 - 0.5) / 0.5
        return tensor.float()

    def load_image(self, label_path, inst_path, feat_path):
        opt = self.opt
        label_img = Image.open(label_path)
        if label_path.find('face') != -1:
            label_img = label_img.convert('L')
        ow, oh = label_img.size
        w = opt.loadSize
        h = int(w * oh / ow)
        label_img = label_img.resize((w, h), Image.NEAREST)
        label_map = self.toTensor(label_img)
        self.label_map = label_map
        oneHot_size = 1, opt.label_nc, h, w
        input_label = self.Tensor(torch.Size(oneHot_size)).zero_()
        self.input_label = input_label.scatter_(1, label_map.long(), 1.0)
        if not opt.no_instance:
            inst_img = Image.open(inst_path)
            inst_img = inst_img.resize((w, h), Image.NEAREST)
            self.inst_map = self.toTensor(inst_img)
            self.edge_map = self.get_edges(self.inst_map)
            self.net_input = Variable(torch.cat((self.input_label, self.edge_map), dim=1), volatile=True)
        else:
            self.net_input = Variable(self.input_label, volatile=True)
        self.features_clustered = np.load(feat_path).item()
        self.object_map = self.inst_map if opt.instance_feat else self.label_map
        object_np = self.object_map.cpu().numpy().astype(int)
        self.feat_map = self.Tensor(1, opt.feat_num, h, w).zero_()
        self.cluster_indices = np.zeros(self.opt.label_nc, np.uint8)
        for i in np.unique(object_np):
            label = i if i < 1000 else i // 1000
            if label in self.features_clustered:
                feat = self.features_clustered[label]
                np.random.seed(i + 1)
                cluster_idx = np.random.randint(0, feat.shape[0])
                self.cluster_indices[label] = cluster_idx
                idx = (self.object_map == i).nonzero()
                self.set_features(idx, feat, cluster_idx)
        self.net_input_original = self.net_input.clone()
        self.label_map_original = self.label_map.clone()
        self.feat_map_original = self.feat_map.clone()
        if not opt.no_instance:
            self.inst_map_original = self.inst_map.clone()

    def reset(self):
        self.net_input = self.net_input_prev = self.net_input_original.clone()
        self.label_map = self.label_map_prev = self.label_map_original.clone()
        self.feat_map = self.feat_map_prev = self.feat_map_original.clone()
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev = self.inst_map_original.clone()
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map

    def undo(self):
        self.net_input = self.net_input_prev
        self.label_map = self.label_map_prev
        self.feat_map = self.feat_map_prev
        if not self.opt.no_instance:
            self.inst_map = self.inst_map_prev
        self.object_map = self.inst_map if self.opt.instance_feat else self.label_map

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def change_labels(self, click_src, click_tgt):
        y_src, x_src = click_src[0], click_src[1]
        y_tgt, x_tgt = click_tgt[0], click_tgt[1]
        label_src = int(self.label_map[0, 0, y_src, x_src])
        inst_src = self.inst_map[0, 0, y_src, x_src]
        label_tgt = int(self.label_map[0, 0, y_tgt, x_tgt])
        inst_tgt = self.inst_map[0, 0, y_tgt, x_tgt]
        idx_src = (self.inst_map == inst_src).nonzero()
        if idx_src.shape:
            self.backup_current_state()
            self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_src, idx_src[:, (2)], idx_src[:, (3)]] = 0
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
            if inst_tgt > 1000:
                tgt_indices = (self.inst_map > label_tgt * 1000) & (self.inst_map < (label_tgt + 1) * 1000)
                inst_tgt = self.inst_map[tgt_indices].max() + 1
            self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = inst_tgt
            self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
            idx_tgt = (self.inst_map == inst_tgt).nonzero()
            if idx_tgt.shape:
                self.copy_features(idx_src, idx_tgt[(0), :])
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def add_strokes(self, click_src, label_tgt, bw, save):
        size = self.net_input.size()
        h, w = size[2], size[3]
        idx_src = torch.LongTensor(bw ** 2, 4).fill_(0)
        for i in range(bw):
            idx_src[i * bw:(i + 1) * bw, (2)] = min(h - 1, max(0, click_src[0] - bw // 2 + i))
            for j in range(bw):
                idx_src[i * bw + j, 3] = min(w - 1, max(0, click_src[1] - bw // 2 + j))
        idx_src = idx_src
        if idx_src.shape:
            if save:
                self.backup_current_state()
            self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            for k in range(self.opt.label_nc):
                self.net_input[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = 0
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
            self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
            self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
            if self.opt.instance_feat:
                feat = self.features_clustered[label_tgt]
                cluster_idx = self.cluster_indices[label_tgt]
                self.set_features(idx_src, feat, cluster_idx)
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def add_objects(self, click_src, label_tgt, mask, style_id=0):
        y, x = click_src[0], click_src[1]
        mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]
        idx_src = torch.from_numpy(mask).nonzero()
        idx_src[:, (2)] += y
        idx_src[:, (3)] += x
        self.backup_current_state()
        self.label_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
        for k in range(self.opt.label_nc):
            self.net_input[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = 0
        self.net_input[idx_src[:, (0)], idx_src[:, (1)] + label_tgt, idx_src[:, (2)], idx_src[:, (3)]] = 1
        self.inst_map[idx_src[:, (0)], idx_src[:, (1)], idx_src[:, (2)], idx_src[:, (3)]] = label_tgt
        self.net_input[:, (-1), :, :] = self.get_edges(self.inst_map)
        self.set_features(idx_src, self.feat, style_id)
        self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def single_forward(self, net_input, feat_map):
        net_input = torch.cat((net_input, feat_map), dim=1)
        fake_image = self.netG.forward(net_input)
        if fake_image.size()[0] == 1:
            return fake_image.data[0]
        return fake_image.data

    def style_forward(self, click_pt, style_id=-1):
        if click_pt is None:
            self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))
            self.crop = None
            self.mask = None
        else:
            instToChange = int(self.object_map[0, 0, click_pt[0], click_pt[1]])
            self.instToChange = instToChange
            label = instToChange if instToChange < 1000 else instToChange // 1000
            self.feat = self.features_clustered[label]
            self.fake_image = []
            self.mask = self.object_map == instToChange
            idx = self.mask.nonzero()
            self.get_crop_region(idx)
            if idx.size():
                if style_id == -1:
                    min_y, min_x, max_y, max_x = self.crop
                    for cluster_idx in range(self.opt.multiple_output):
                        self.set_features(idx, self.feat, cluster_idx)
                        fake_image = self.single_forward(self.net_input, self.feat_map)
                        fake_image = util.tensor2im(fake_image[:, min_y:max_y, min_x:max_x])
                        self.fake_image.append(fake_image)
                    """### To speed up previewing different style results, either crop or downsample the label maps
                    if instToChange > 1000:
                        (min_y, min_x, max_y, max_x) = self.crop
                        ### crop
                        _, _, h, w = self.net_input.size()
                        offset = 512
                        y_start, x_start = max(0, min_y-offset), max(0, min_x-offset)
                        y_end, x_end = min(h, (max_y + offset)), min(w, (max_x + offset))
                        y_region = slice(y_start, y_start+(y_end-y_start)//16*16)
                        x_region = slice(x_start, x_start+(x_end-x_start)//16*16)
                        net_input = self.net_input[:,:,y_region,x_region]
                        for cluster_idx in range(self.opt.multiple_output):
                            self.set_features(idx, self.feat, cluster_idx)
                            fake_image = self.single_forward(net_input, self.feat_map[:,:,y_region,x_region])
                            fake_image = util.tensor2im(fake_image[:,min_y-y_start:max_y-y_start,min_x-x_start:max_x-x_start])
                            self.fake_image.append(fake_image)
                    else:
                        ### downsample
                        (min_y, min_x, max_y, max_x) = [crop//2 for crop in self.crop]
                        net_input = self.net_input[:,:,::2,::2]
                        size = net_input.size()
                        net_input_batch = net_input.expand(self.opt.multiple_output, size[1], size[2], size[3])
                        for cluster_idx in range(self.opt.multiple_output):
                            self.set_features(idx, self.feat, cluster_idx)
                            feat_map = self.feat_map[:,:,::2,::2]
                            if cluster_idx == 0:
                                feat_map_batch = feat_map
                            else:
                                feat_map_batch = torch.cat((feat_map_batch, feat_map), dim=0)
                        fake_image_batch = self.single_forward(net_input_batch, feat_map_batch)
                        for i in range(self.opt.multiple_output):
                            self.fake_image.append(util.tensor2im(fake_image_batch[i,:,min_y:max_y,min_x:max_x]))"""
                else:
                    self.set_features(idx, self.feat, style_id)
                    self.cluster_indices[label] = style_id
                    self.fake_image = util.tensor2im(self.single_forward(self.net_input, self.feat_map))

    def backup_current_state(self):
        self.net_input_prev = self.net_input.clone()
        self.label_map_prev = self.label_map.clone()
        self.inst_map_prev = self.inst_map.clone()
        self.feat_map_prev = self.feat_map.clone()

    def get_crop_region(self, idx):
        size = self.net_input.size()
        h, w = size[2], size[3]
        min_y, min_x = idx[:, (2)].min(), idx[:, (3)].min()
        max_y, max_x = idx[:, (2)].max(), idx[:, (3)].max()
        crop_min = 128
        if max_y - min_y < crop_min:
            min_y = max(0, (max_y + min_y) // 2 - crop_min // 2)
            max_y = min(h - 1, min_y + crop_min)
        if max_x - min_x < crop_min:
            min_x = max(0, (max_x + min_x) // 2 - crop_min // 2)
            max_x = min(w - 1, min_x + crop_min)
        self.crop = min_y, min_x, max_y, max_x
        self.mask = self.mask[:, :, min_y:max_y, min_x:max_x]

    def update_features(self, cluster_idx, mask=None, click_pt=None):
        self.feat_map_prev = self.feat_map.clone()
        if mask is not None:
            y, x = click_pt[0], click_pt[1]
            mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]
            idx = torch.from_numpy(mask).nonzero()
            idx[:, (2)] += y
            idx[:, (3)] += x
        else:
            idx = (self.object_map == self.instToChange).nonzero()
        self.set_features(idx, self.feat, cluster_idx)

    def set_features(self, idx, feat, cluster_idx):
        for k in range(self.opt.feat_num):
            self.feat_map[idx[:, (0)], idx[:, (1)] + k, idx[:, (2)], idx[:, (3)]] = feat[cluster_idx, k]

    def copy_features(self, idx_src, idx_tgt):
        for k in range(self.opt.feat_num):
            val = self.feat_map[idx_tgt[0], idx_tgt[1] + k, idx_tgt[2], idx_tgt[3]]
            self.feat_map[idx_src[:, (0)], idx_src[:, (1)] + k, idx_src[:, (2)], idx_src[:, (3)]] = val

    def get_current_visuals(self, getLabel=False):
        mask = self.mask
        if self.mask is not None:
            mask = np.transpose(self.mask[0].cpu().float().numpy(), (1, 2, 0)).astype(np.uint8)
        dict_list = [('fake_image', self.fake_image), ('mask', mask)]
        if getLabel:
            label = util.tensor2label(self.net_input.data[0], self.opt.label_nc)
            dict_list += [('label', label)]
        return OrderedDict(dict_list)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (DataParallelWithCallback,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (Derenderer3d,
     lambda: ([], {'mode': 4, 'image_size': 4, 'render_size': 4}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DictGatherDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (GlobalGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (LocalEnhancer,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MultiscaleDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PPMBilinear,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4096, 64, 64])], {}),
     False),
    (RPN,
     lambda: ([], {'anchors_per_location': 4, 'anchor_stride': 1, 'depth': 1}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     False),
    (SamePad2d,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SynchronizedBatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SynchronizedBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SynchronizedBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TopDownLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UIModel,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
    (UserScatteredDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (VGGLoss,
     lambda: ([], {'gpu_ids': False}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_SynchronizedBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ysymyth_3D_SDN(_paritybench_base):
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

