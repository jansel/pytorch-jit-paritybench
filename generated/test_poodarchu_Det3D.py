import sys
_module = sys.modules[__name__]
del sys
det3d = _module
builder = _module
core = _module
anchor = _module
anchor_generator = _module
target_assigner = _module
target_ops = _module
bbox = _module
box_coders = _module
box_np_ops = _module
box_torch_ops = _module
geometry = _module
iou = _module
region_similarity = _module
evaluation = _module
bbox_overlaps = _module
class_names = _module
coco_utils = _module
mean_ap = _module
recall = _module
fp16 = _module
decorators = _module
hooks = _module
utils = _module
input = _module
voxel_generator = _module
sampler = _module
preprocess = _module
sample_ops = _module
dist_utils = _module
misc = _module
datasets = _module
custom = _module
dataset_factory = _module
dataset_wrappers = _module
kitti = _module
eval = _module
eval_hooks = _module
kitti_common = _module
loader = _module
build_loader = _module
sampler = _module
lyft = _module
lyft_common = _module
splits = _module
nuscenes = _module
nusc_common = _module
plot_results = _module
pipelines = _module
compose = _module
formating = _module
loading = _module
test_aug = _module
transforms = _module
registry = _module
create_gt_database = _module
distributed = _module
ground_plane_detection = _module
evaluate = _module
rotate_iou = _module
oss = _module
preprocess = _module
models = _module
backbones = _module
resnet = _module
scn = _module
senet = _module
ssd_vgg = _module
bbox_heads = _module
mg_head = _module
builder = _module
detectors = _module
base = _module
point_pillars = _module
single_stage = _module
voxelnet = _module
losses = _module
accuracy = _module
balanced_l1_loss = _module
cross_entropy_loss = _module
focal_loss = _module
ghm_loss = _module
iou_loss = _module
losses = _module
metrics = _module
mse_loss = _module
smooth_l1_loss = _module
utils = _module
necks = _module
fpn = _module
rpn = _module
readers = _module
cropped_voxel_encoder = _module
pillar_encoder = _module
voxel_encoder = _module
conv_module = _module
conv_ws = _module
misc = _module
norm = _module
scale = _module
weight_init = _module
ops = _module
align_feature_and_aggregation = _module
alignfeature = _module
check = _module
functions = _module
align_feature = _module
modules = _module
align_feature = _module
setup = _module
correlation = _module
check = _module
correlation = _module
correlation = _module
setup = _module
rotated_boxes = _module
iou3d_utils = _module
setup = _module
nms = _module
nms_cpu = _module
nms_gpu = _module
point_cloud = _module
bev_ops = _module
point_cloud_ops = _module
pointnet2_modules = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
roipool3d_utils = _module
setup = _module
RoI = _module
grad_check = _module
setup = _module
sigmoid_focal_loss = _module
sigmoid_focal_loss = _module
syncbn = _module
syncbn = _module
solver = _module
background = _module
fastai_optim = _module
learning_schedules = _module
learning_schedules_fastai = _module
optim = _module
torchie = _module
apis = _module
env = _module
train = _module
cnn = _module
weight_init = _module
fileio = _module
handlers = _module
json_handler = _module
pickle_handler = _module
yaml_handler = _module
io = _module
parse = _module
parallel = _module
_functions = _module
collate = _module
data_container = _module
data_parallel = _module
distributed = _module
scatter_gather = _module
trainer = _module
checkpoint = _module
closure = _module
hook = _module
iter_timer = _module
logger = _module
pavi = _module
tensorboard = _module
text = _module
lr_updater = _module
memory = _module
optimizer = _module
sampler_seed = _module
log_buffer = _module
parallel_test = _module
priority = _module
trainer = _module
utils = _module
config = _module
path = _module
progressbar = _module
timer = _module
buildtools = _module
command = _module
pybind11_build = _module
checkpoint = _module
config_tool = _module
collect_env = _module
dist_common = _module
find = _module
flops_counter = _module
imports = _module
print_utils = _module
registry = _module
utils = _module
version = _module
visualization = _module
netviz = _module
preds_vis = _module
show = _module
show_lidar_vtk = _module
simplevis = _module
QVTKRenderWindowInteractor = _module
vtk_visualizer = _module
plot3d = _module
pointobject = _module
renderwidget = _module
visualizercontrol = _module
lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn = _module
nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn = _module
kitti_point_pillars_mghead_syncbn = _module
nusc_all_point_pillars_mghead_syncbn = _module
kitti_all_vfev3_spmiddlefhd_rpn1_mghead_syncbn = _module
kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn = _module
setup = _module
analyze_logs = _module
create_data = _module
dist_test = _module
get_flops = _module
test = _module
train = _module

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


import logging


from functools import partial


import numpy as np


import torch


from torch import nn


from abc import ABCMeta


from abc import abstractmethod


from abc import abstractproperty


import math


from functools import reduce


from torch import stack as tstack


import functools


from inspect import getfullargspec


import copy


import torch.nn as nn


from collections import abc


from collections import OrderedDict


import torch.distributed as dist


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.utils.data import Dataset


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from torch.utils.data import DataLoader


from torch.utils.data.sampler import Sampler


from torch.utils.data import DistributedSampler as _DistributedSampler


import time


from collections import defaultdict


import itertools


import torch.utils.checkpoint as cp


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn import BatchNorm1d


from torch.nn import functional as F


from torch.utils import model_zoo


import torch.nn.functional as F


from enum import Enum


from torch.autograd import Variable


from torchvision.models import resnet


import warnings


import inspect


from torch.autograd.function import Function


from torch.nn import BatchNorm2d


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


from typing import List


from typing import Tuple


from torch.autograd import gradcheck


from queue import Queue


import torch.cuda.comm as comm


from collections import Iterable


from copy import deepcopy


from itertools import chain


from torch.nn.utils import parameters_to_vector


from torch.optim.optimizer import Optimizer


import random


import torch.multiprocessing as mp


import re


from torch.nn.parallel import DistributedDataParallel


from torch.nn.parallel._functions import _get_stream


import collections


from torch.utils.data.dataloader import default_collate


from torch.nn.parallel import DataParallel


from torch.nn.parallel._functions import Scatter as OrigScatter


import torchvision


from torch.nn.utils import clip_grad


import queue


from torch.utils.collect_env import get_pretty_env_info


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


from collections import namedtuple


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


conv_cfg = {'Conv': nn.Conv2d, 'ConvWS': ConvWS2d}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer
    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.
    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


class DistributedSyncBNFucntion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, sync=True):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.sync = sync
        if ctx.training:
            ex, exs = syncbn_gpu.batch_norm_collect_statistics(x)
            world_size = dist.get_world_size()
            if ctx.sync:
                ex_all_reduce = dist.all_reduce(ex, async_op=True)
                exs_all_reduce = dist.all_reduce(exs, async_op=True)
                ex_all_reduce.wait()
                exs_all_reduce.wait()
                ex /= world_size
                exs /= world_size
            n = x.numel() / x.shape[1] * world_size
            ctx.cf = n / (n - 1)
            var = (exs - ex ** 2) * ctx.cf
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * ex)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var)
            ctx.mark_dirty(running_mean, running_var)
            y = syncbn_gpu.batch_norm_transform_input(x, gamma, beta, ex, exs, ctx.eps, ctx.cf)
            ctx.save_for_backward(x, ex, exs, gamma, beta)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_ouput):
        x, ex, exs, gamma, beta = ctx.saved_tensors
        grad_gamma, grad_beta, grad_ex, grad_exs = syncbn_gpu.batch_norm_collect_grad_statistics(x, grad_ouput, gamma, ex, exs, ctx.eps, ctx.cf)
        if ctx.training:
            if ctx.sync:
                world_size = dist.get_world_size()
                grad_ex_all_reduce = dist.all_reduce(grad_ex, async_op=True)
                grad_exs_all_reduce = dist.all_reduce(grad_exs, async_op=True)
                grad_gamma_all_reduce = dist.all_reduce(grad_gamma, async_op=True)
                grad_beta_all_reduce = dist.all_reduce(grad_beta, async_op=True)
                grad_ex_all_reduce.wait()
                grad_exs_all_reduce.wait()
                grad_gamma_all_reduce.wait()
                grad_beta_all_reduce.wait()
                grad_ex /= world_size
                grad_exs /= world_size
        grad_input = syncbn_gpu.batch_norm_input_backward(x, grad_ouput, gamma, ex, exs, grad_ex, grad_exs, ctx.eps, ctx.cf)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None, None


class DistributedSyncBN(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True, sync=True):
        super(DistributedSyncBN, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.sync = sync

    def forward(self, x):
        if self.training and self.sync:
            return DistributedSyncBNFucntion.apply(x, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps, self.sync)
        else:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:
                        exponential_average_factor = self.momentum
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)


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


class NaiveSyncBatchNorm(BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).
    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if comm.get_world_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBatchNorm does not support empty input'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])
        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'BN1d': ('bn1d', nn.BatchNorm1d), 'GN': ('gn', nn.GroupNorm), 'SyncBN': ('bn', DistributedSyncBN), 'NaiveSyncBN': ('bn', NaiveSyncBatchNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, gcb=None, gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert gen_attention is None, 'Not implemented yet.'
        assert gcb is None, 'Not implemented yet.'
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


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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
        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbone')


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module('torchvision.models.{}'.format(name))
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=None):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    match_matrix = [(len(j) if i.endswith(j) else 0) for i in current_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    idxs[max_match_size == 0] = -1
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = '{: <{}} loaded from {: <{}} of shape {}'
    if logger is None:
        logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(log_str_template.format(key, max_size, key_old, max_size_loaded, tuple(loaded_state_dict[key_old].shape)))


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, '')] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, logger=None):
    model_state_dict = model.state_dict()
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix='module.')
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=logger)
    model.load_state_dict(model_state_dict)


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_url_dist(url):
    """ In distributed setting, this function only download checkpoint at
    local rank 0 """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url)
    return checkpoint


open_mmlab_model_urls = {'vgg16_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth', 'resnet50_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_caffe-788b5fa3.pth', 'resnet101_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth', 'resnext50_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50-32x4d-0ab1a123.pth', 'resnext101_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth', 'resnext101_64x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth', 'contrib/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_thangvubk-ad1730dd.pth', 'detectron/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn-9186a21c.pth', 'detectron/resnet101_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn-cac0ab98.pth', 'jhu/resnet50_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_ws-15beedd8.pth', 'jhu/resnet101_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn_ws-3e3c308c.pth', 'jhu/resnext50_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth', 'jhu/resnext101_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn_ws-34ac1a9e.pth', 'jhu/resnext50_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn-c7e8b754.pth', 'jhu/resnext101_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn-ac3bb84e.pth', 'msra/hrnetv2_w18': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w18-00eb2006.pth', 'msra/hrnetv2_w32': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth', 'msra/hrnetv2_w40': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w40-ed0b031c.pth', 'bninception_caffe': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/bn_inception_caffe-ed2e8665.pth', 'kin400/i3d_r50_f32s2_k400': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/i3d_r50_f32s2_k400-2c57e077.pth', 'kin400/nl3d_r50_f32s2_k400': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/nl3d_r50_f32s2_k400-fa7e7caa.pth'}


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, gcb=None, gen_attention=None, gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1])
    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb, gen_attention=gen_attention if 0 in gen_attention_blocks else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb, gen_attention=gen_attention if i in gen_attention_blocks else None))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
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
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """
    arch_settings = {(18): (BasicBlock, (2, 2, 2, 2)), (34): (BasicBlock, (3, 4, 6, 3)), (50): (Bottleneck, (3, 4, 6, 3)), (101): (Bottleneck, (3, 4, 23, 3)), (152): (Bottleneck, (3, 8, 36, 3))}

    def __init__(self, depth, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3), style='pytorch', frozen_stages=-1, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, dcn=None, stage_with_dcn=(False, False, False, False), gcb=None, stage_with_gcb=(False, False, False, False), gen_attention=None, stage_with_gen_attention=((), (), (), ()), with_cp=False, zero_init_residual=True):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self._make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2 ** i
            res_layer = make_res_layer(self.block, self.inplanes, planes, num_blocks, stride=stride, dilation=dilation, style=self.style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb, gen_attention=gen_attention, gen_attention_blocks=stage_with_gen_attention[i])
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(self.conv_cfg, 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)
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


class SpMiddleFHD(nn.Module):

    def __init__(self, num_input_features=128, norm_cfg=None, name='SpMiddleFHD', **kwargs):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        self.dcn = None
        self.zero_init_residual = False
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d', eps=0.001, momentum=0.01)
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, bias=False, indice_key='subm0'), build_norm_layer(norm_cfg, 16)[1], nn.ReLU(), SubMConv3d(16, 16, 3, bias=False, indice_key='subm0'), build_norm_layer(norm_cfg, 16)[1], nn.ReLU(), SparseConv3d(16, 32, 3, 2, padding=1, bias=False), build_norm_layer(norm_cfg, 32)[1], nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1', bias=False), build_norm_layer(norm_cfg, 32)[1], nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1', bias=False), build_norm_layer(norm_cfg, 32)[1], nn.ReLU(), SparseConv3d(32, 64, 3, 2, padding=1, bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2', bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2', bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2', bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SparseConv3d(64, 64, 3, 2, padding=[0, 1, 1], bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3', bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3', bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3', bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SparseConv3d(64, 64, (3, 1, 1), (2, 1, 1), bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU())

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleFHDNobn(nn.Module):

    def __init__(self, num_input_features=128, norm_cfg=None, name='SpMiddleFHD', **kwargs):
        super(SpMiddleFHDNobn, self).__init__()
        self.name = name
        self.dcn = None
        self.zero_init_residual = False
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d', eps=0.001, momentum=0.01)
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, bias=True, indice_key='subm0'), nn.ReLU(), SubMConv3d(16, 16, 3, bias=True, indice_key='subm0'), nn.ReLU(), SparseConv3d(16, 32, 3, 2, padding=1, bias=True), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1', bias=True), nn.ReLU(), SubMConv3d(32, 32, 3, indice_key='subm1', bias=True), nn.ReLU(), SparseConv3d(32, 64, 3, 2, padding=1, bias=True), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2', bias=True), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2', bias=True), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm2', bias=True), nn.ReLU(), SparseConv3d(64, 64, 3, 2, padding=[0, 1, 1], bias=True), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3', bias=True), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3', bias=True), nn.ReLU(), SubMConv3d(64, 64, 3, indice_key='subm3', bias=True), nn.ReLU(), SparseConv3d(64, 64, (3, 1, 1), (2, 1, 1), bias=True), nn.ReLU())

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)


class SpMiddleResNetFHD(nn.Module):

    def __init__(self, num_input_features=128, norm_cfg=None, name='SpMiddleResNetFHD', **kwargs):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name
        self.dcn = None
        self.zero_init_residual = False
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d', eps=0.001, momentum=0.01)
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, bias=False, indice_key='res0'), build_norm_layer(norm_cfg, 16)[1], nn.ReLU(), SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key='res0'), SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key='res0'), SparseConv3d(16, 32, 3, 2, padding=1, bias=False), build_norm_layer(norm_cfg, 32)[1], nn.ReLU(), SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key='res1'), SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key='res1'), SparseConv3d(32, 64, 3, 2, padding=1, bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key='res2'), SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key='res2'), SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1], bias=False), build_norm_layer(norm_cfg, 128)[1], nn.ReLU(), SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key='res3'), SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key='res3'), SparseConv3d(128, 128, (3, 1, 1), (2, 1, 1), bias=False), build_norm_layer(norm_cfg, 128)[1], nn.ReLU())

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class RCNNSpMiddleFHD(nn.Module):

    def __init__(self, num_input_features=128, norm_cfg=None, name='RCNNSpMiddleFHD', **kwargs):
        super(RCNNSpMiddleFHD, self).__init__()
        self.name = name
        self.dcn = None
        self.zero_init_residual = False
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d', eps=0.001, momentum=0.01)
        self.middle_conv = spconv.SparseSequential(SubMConv3d(num_input_features, 16, 3, bias=False, indice_key='subm0'), build_norm_layer(norm_cfg, 16)[1], nn.ReLU(), SubMConv3d(16, 16, 3, bias=False, indice_key='subm0'), build_norm_layer(norm_cfg, 16)[1], nn.ReLU(), SparseConv3d(16, 32, 3, 2, padding=1, bias=False), build_norm_layer(norm_cfg, 32)[1], nn.ReLU(), SubMConv3d(32, 32, 3, bias=False, indice_key='subm1'), build_norm_layer(norm_cfg, 32)[1], nn.ReLU(), SparseConv3d(32, 64, 3, 2, bias=False, padding=1), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, bias=False, indice_key='subm2'), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SparseConv3d(64, 64, 3, 2, bias=False, padding=[1, 1, 0]), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SubMConv3d(64, 64, 3, bias=False, indice_key='subm3'), build_norm_layer(norm_cfg, 64)[1], nn.ReLU(), SparseConv3d(64, 64, (1, 1, 3), (1, 1, 2), bias=False), build_norm_layer(norm_cfg, 64)[1], nn.ReLU())

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1]) + [0, 0, 1]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        ret = ret.permute(0, 1, 4, 2, 3).contiguous()
        N, C, W, D, H = ret.shape
        ret = ret.view(N, C * W, D, H)
        return ret


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


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


HEADS = Registry('head')


class Head(nn.Module):

    def __init__(self, num_input, num_pred, num_cls, use_dir=False, num_dir=0, header=True, name='', focal_loss_init=False, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.use_dir = use_dir
        self.conv_box = nn.Conv2d(num_input, num_pred, 1)
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)
        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)

    def forward(self, x):
        ret_list = []
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        ret_dict = {'box_preds': box_preds, 'cls_preds': cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_preds
        return ret_dict


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


class RegHead(nn.Module):

    def __init__(self, mode='z', in_channels=sum([128]), norm_cfg=None, tasks=None, name='rpn', logger=None, **kwargs):
        super(RegHead, self).__init__()
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=0.001, momentum=0.01)
        self._norm_cfg = norm_cfg
        num_preds = []
        for idx, num_class in enumerate(num_classes):
            num_preds.append(2)
        self.tasks = nn.ModuleList()
        num_input = in_channels
        for task_id, num_pred in enumerate(num_preds):
            conv_box = nn.Conv2d(num_input, num_pred, 1)
            self.tasks.append(conv_box)
        self.crop_cfg = kwargs.get('crop_cfg', None)
        self.z_mode = kwargs.get('z_type', 'top')
        self.iou_loss = kwargs.get('iou_loss', False)

    def forward(self, x):
        ret_dicts = []
        for task in self.tasks:
            out = task(x)
            out = F.max_pool2d(out, kernel_size=out.size()[2:])
            ret_dicts.append(out.permute(0, 2, 3, 1).contiguous())
        return ret_dicts

    def loss(self, example, preds, **kwargs):
        batch_size_dev = example['targets'].shape[0]
        rets = []
        for task_id, task_pred in enumerate(preds):
            zg = example['targets'][:, 2:3]
            hg = example['targets'][:, 3:4]
            gg = example['targets'][:, 4:5]
            gp = example['ground_plane'].view(-1, 1)
            zt = task_pred[:, :, :, 0:1].view(-1, 1)
            ht = task_pred[:, :, :, 1:2].view(-1, 1)
            height_loss = smooth_l1_loss(ht, hg, 3.0) / batch_size_dev
            height_a = self.crop_cfg.anchor.height
            z_center_a = self.crop_cfg.anchor.center
            if self.z_mode == 'top':
                z_top_a = z_center_a + height_a / 2
                gt = z_top_a + zt - (height_a + ht) - gp
                z_loss = smooth_l1_loss(zt, zg, 3.0) / batch_size_dev
                yg_top, yg_down = zg + z_top_a, zg + z_top_a - (hg + height_a)
                yp_top, yp_down = zt + z_top_a, zt + z_top_a - (ht + height_a)
            elif self.z_mode == 'center':
                gt = z_center_a + zt - (height_a + ht) / 2.0 - gp
                z_loss = smooth_l1_loss(zt, zg, 3.0) / batch_size_dev
                yg_top, yg_down = zg + z_center_a + (hg + height_a) / 2.0, zg + z_center_a - (hg + height_a) / 2.0
                yp_top, yp_down = zt + z_center_a + (ht + height_a) / 2.0, zt + z_center_a - (ht + height_a) / 2.0
            gp_loss = smooth_l1_loss(gt, gg, 3.0) / batch_size_dev
            h_intersect = torch.min(yp_top, yg_top) - torch.max(yp_down, yg_down)
            iou = h_intersect / (hg + height_a + ht + height_a - h_intersect)
            iou[iou < 0] = 0.0
            iou[iou > 1] = 1.0
            iou_loss = (1 - iou).sum() / batch_size_dev
            ret = dict(loss=z_loss + height_loss + gp_loss, z_loss=z_loss, height_loss=height_loss, gp_loss=gp_loss)
            if self.iou_loss:
                ret['loss'] += iou_loss
                ret.update(dict(iou_loss=iou_loss))
            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        return rets_merged

    def predict(self, example, preds, stage_one_outputs):
        rets = []
        for task_id, task_pred in enumerate(preds):
            if self.z_mode == 'top':
                height_a = self.crop_cfg.anchor.height
                z_top_a = self.crop_cfg.anchor.center + height_a / 2
                task_pred[:, :, :, 1] += height_a
                task_pred[:, :, :, 0] += z_top_a - task_pred[:, :, :, 1] / 2.0
            elif self.z_mode == 'center':
                height_a = self.crop_cfg.anchor.height
                z_center_a = self.crop_cfg.anchor.center
                task_pred[:, :, :, 0] += z_center_a
                task_pred[:, :, :, 1] += height_a
            rets.append(task_pred.view(-1, 2))
        num_tasks = len(rets)
        ret_list = []
        num_preds = len(rets)
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = torch.cat(rets)
            ret_list.append(ret)
        cnt = 0
        for idx, sample in enumerate(stage_one_outputs):
            for box in sample['box3d_lidar']:
                box[2] = ret_list[0][cnt, 0]
                box[5] = ret_list[0][cnt, 1]
                cnt += 1
        return stage_one_outputs


class LossNormType(Enum):
    NormByNumPositives = 'norm_by_num_positives'
    NormByNumExamples = 'norm_by_num_examples'
    NormByNumPosNeg = 'norm_by_num_pos_neg'
    DontNorm = 'dont_norm'


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


LOSSES = Registry('loss')


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if torchie.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg):
    return build(cfg, LOSSES)


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def create_loss(loc_loss_ftor, cls_loss_ftor, box_preds, cls_preds, cls_targets, cls_weights, reg_targets, reg_weights, num_class, encode_background_as_zeros=True, encode_rad_error_by_sin=True, bev_only=False, box_code_size=9):
    batch_size = int(box_preds.shape[0])
    if bev_only:
        box_preds = box_preds.view(batch_size, -1, box_code_size - 2)
    else:
        box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot_f(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)
    cls_losses = cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)
    return loc_losses, cls_losses


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (limit_period(rot_gt - dir_offset, 0.5, np.pi * 2) > 0).long()
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


class MultiGroupHead(nn.Module):

    def __init__(self, mode='3d', in_channels=[128], norm_cfg=None, tasks=[], weights=[], num_classes=[1], box_coder=None, with_cls=True, with_reg=True, reg_class_agnostic=False, encode_background_as_zeros=True, loss_norm=dict(type='NormByNumPositives', pos_class_weight=1.0, neg_class_weight=1.0), loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), use_sigmoid_score=True, loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0), encode_rad_error_by_sin=True, loss_aux=None, direction_offset=0.0, name='rpn', logger=None):
        super(MultiGroupHead, self).__init__()
        assert with_cls or with_reg
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.num_anchor_per_locs = [(2 * n) for n in num_classes]
        self.box_coder = box_coder
        box_code_sizes = [box_coder.code_size] * len(num_classes)
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.encode_rad_error_by_sin = encode_rad_error_by_sin
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_sigmoid_score = use_sigmoid_score
        self.box_n_dim = self.box_coder.code_size
        self.anchor_dim = self.box_coder.n_dim
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_bbox)
        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)
        self.loss_norm = loss_norm
        if not logger:
            logger = logging.getLogger('MultiGroupHead')
        self.logger = logger
        self.dcn = None
        self.zero_init_residual = False
        self.use_direction_classifier = loss_aux is not None
        if loss_aux:
            self.direction_offset = direction_offset
        self.bev_only = True if mode == 'bev' else False
        num_clss = []
        num_preds = []
        num_dirs = []
        for num_c, num_a, box_cs in zip(num_classes, self.num_anchor_per_locs, box_code_sizes):
            if self.encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)
            if self.bev_only:
                num_pred = num_a * (box_cs - 2)
            else:
                num_pred = num_a * box_cs
            num_preds.append(num_pred)
            if self.use_direction_classifier:
                num_dir = num_a * 2
                num_dirs.append(num_dir)
            else:
                num_dir = None
        logger.info(f'num_classes: {num_classes}, num_preds: {num_preds}, num_dirs: {num_dirs}')
        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(Head(in_channels, num_pred, num_cls, use_dir=self.use_direction_classifier, num_dir=num_dirs[task_id] if self.use_direction_classifier else None, header=False))
        logger.info('Finish MultiGroupHead Initialization')

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts

    def prepare_loss_weights(self, labels, loss_norm=dict(type='NormByNumPositives', pos_cls_weight=1.0, neg_cls_weight=1.0), dtype=torch.float32):
        loss_norm_type = getattr(LossNormType, loss_norm['type'])
        pos_cls_weight = loss_norm['pos_cls_weight']
        neg_cls_weight = loss_norm['neg_cls_weight']
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

    def loss(self, example, preds_dicts, **kwargs):
        voxels = example['voxels']
        num_points = example['num_points']
        coors = example['coordinates']
        batch_anchors = example['anchors']
        batch_size_device = batch_anchors[0].shape[0]
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            losses = dict()
            num_class = self.num_classes[task_id]
            box_preds = preds_dict['box_preds']
            cls_preds = preds_dict['cls_preds']
            labels = example['labels'][task_id]
            if kwargs.get('mode', False):
                reg_targets = example['reg_targets'][task_id][:, :, [0, 1, 3, 4, 6]]
                reg_targets_left = example['reg_targets'][task_id][:, :, [2, 5]]
            else:
                reg_targets = example['reg_targets'][task_id]
            cls_weights, reg_weights, cared = self.prepare_loss_weights(labels, loss_norm=self.loss_norm, dtype=torch.float32)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)
            loc_loss, cls_loss = create_loss(self.loss_reg, self.loss_cls, box_preds, cls_preds, cls_targets, cls_weights, reg_targets, reg_weights, num_class, self.encode_background_as_zeros, self.encode_rad_error_by_sin, bev_only=self.bev_only, box_code_size=self.box_n_dim)
            loc_loss_reduced = loc_loss.sum() / batch_size_device
            loc_loss_reduced *= self.loss_reg._loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self.loss_norm['pos_cls_weight']
            cls_neg_loss /= self.loss_norm['neg_cls_weight']
            cls_loss_reduced = cls_loss.sum() / batch_size_device
            cls_loss_reduced *= self.loss_cls._loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            if self.use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'][task_id], reg_targets, dir_offset=self.direction_offset)
                dir_logits = preds_dict['dir_cls_preds'].view(batch_size_device, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_device
                loss += dir_loss * self.loss_aux._loss_weight
            loc_loss_elem = [(loc_loss[:, :, i].sum() / batch_size_device) for i in range(loc_loss.shape[-1])]
            ret = {'loss': loss, 'cls_pos_loss': cls_pos_loss.detach().cpu(), 'cls_neg_loss': cls_neg_loss.detach().cpu(), 'dir_loss_reduced': dir_loss.detach().cpu() if self.use_direction_classifier else torch.tensor(0), 'cls_loss_reduced': cls_loss_reduced.detach().cpu().mean(), 'loc_loss_reduced': loc_loss_reduced.detach().cpu().mean(), 'loc_loss_elem': [elem.detach().cpu() for elem in loc_loss_elem], 'num_pos': (labels > 0)[0].sum(), 'num_neg': (labels == 0)[0].sum()}
            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        return rets_merged

    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx),
                    for nuscenes, sample_token is saved in it.
            }
        """
        voxels = example['voxels']
        num_points = example['num_points']
        coors = example['coordinates']
        batch_anchors = example['anchors']
        batch_size_device = batch_anchors[0].shape[0]
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = batch_anchors[task_id].shape[0]
            if 'metadata' not in example or len(example['metadata']) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example['metadata']
            batch_task_anchors = example['anchors'][task_id].view(batch_size, -1, self.anchor_dim)
            if 'anchors_mask' not in example:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = example['anchors_mask'][task_id].view(batch_size, -1)
            batch_box_preds = preds_dict['box_preds']
            batch_cls_preds = preds_dict['cls_preds']
            if self.bev_only:
                box_ndim = self.box_n_dim - 2
            else:
                box_ndim = self.box_n_dim
            if kwargs.get('mode', False):
                batch_box_preds_base = batch_box_preds.view(batch_size, -1, box_ndim)
                batch_box_preds = batch_task_anchors.clone()
                batch_box_preds[:, :, [0, 1, 3, 4, 6]] = batch_box_preds_base
            else:
                batch_box_preds = batch_box_preds.view(batch_size, -1, box_ndim)
            num_class_with_bg = self.num_classes[task_id]
            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1
            batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
            batch_reg_preds = self.box_coder.decode_torch(batch_box_preds[:, :, :self.box_coder.code_size], batch_task_anchors)
            if self.use_direction_classifier:
                batch_dir_preds = preds_dict['dir_cls_preds']
                batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
            else:
                batch_dir_preds = [None] * batch_size
            rets.append(self.get_task_detections(task_id, num_class_with_bg, test_cfg, batch_cls_preds, batch_reg_preds, batch_dir_preds, batch_anchors_mask, meta_list))
        num_tasks = len(rets)
        ret_list = []
        num_preds = len(rets)
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ['box3d_lidar', 'scores']:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ['label_preds']:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == 'metadata':
                    ret[k] = rets[0][i][k]
            ret_list.append(ret)
        return ret_list

    def get_task_detections(self, task_id, num_class_with_bg, test_cfg, batch_cls_preds, batch_reg_preds, batch_dir_preds=None, batch_anchors_mask=None, meta_list=None):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(post_center_range, dtype=batch_reg_preds.dtype, device=batch_reg_preds.device)
        for box_preds, cls_preds, dir_preds, a_mask, meta in zip(batch_reg_preds, batch_cls_preds, batch_dir_preds, batch_anchors_mask, meta_list):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self.encode_background_as_zeros:
                assert self.use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            elif self.use_sigmoid_score:
                total_scores = torch.sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            if test_cfg.nms.use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            feature_map_size_prod = batch_reg_preds.shape[1] // self.num_anchor_per_locs[task_id]
            if test_cfg.nms.use_multi_class_nms:
                assert self.encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                if not test_cfg.nms.use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = [test_cfg.score_threshold] * self.num_classes[task_id]
                pre_max_sizes = [test_cfg.nms.nms_pre_max_size] * self.num_classes[task_id]
                post_max_sizes = [test_cfg.nms.nms_post_max_size] * self.num_classes[task_id]
                iou_thresholds = [test_cfg.nms.nms_iou_threshold] * self.num_classes[task_id]
                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(range(self.num_classes[task_id]), score_threshs, pre_max_sizes, post_max_sizes, iou_thresholds):
                    self._nms_class_agnostic = False
                    if self._nms_class_agnostic:
                        class_scores = total_scores.view(feature_map_size_prod, -1, self.num_classes[task_id])[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        anchors_range = self.target_assigners[task_id].anchors_range
                        class_scores = total_scores.view(-1, self._num_classes[task_id])[anchors_range[0]:anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(-1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds.view(-1, box_preds.shape[-1])[anchors_range[0]:anchors_range[1], :]
                        class_boxes = class_boxes.contiguous().view(-1, box_preds.shape[-1])
                        if self.use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[anchors_range[0]:anchors_range[1]]
                            class_dir_labels = class_dir_labels.contiguous().view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[class_scores_keep]
                        keep = nms_func(class_boxes_nms, class_scores, pre_ms, post_ms, iou_th)
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]
                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(torch.full([class_boxes[selected].shape[0]], class_idx, dtype=torch.int64, device=box_preds.device))
                        if self.use_direction_classifier:
                            selected_dir_labels.append(class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self.use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
            else:
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(total_scores.shape[0], device=total_scores.device, dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)
                if test_cfg.score_threshold > 0.0:
                    thresh = torch.tensor([test_cfg.score_threshold], device=total_scores.device).type_as(total_scores)
                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if test_cfg.score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self.use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                    if not test_cfg.nms.use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
                    selected = nms_func(boxes_for_nms, top_scores, pre_max_size=test_cfg.nms.nms_pre_max_size, post_max_size=test_cfg.nms.nms_post_max_size, iou_threshold=test_cfg.nms.nms_iou_threshold)
                else:
                    selected = []
                selected_boxes = box_preds[selected]
                if self.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self.use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] - self.direction_offset > 0) ^ dir_labels.bool()
                    box_preds[..., -1] += torch.where(opp_labels, torch.tensor(np.pi).type_as(box_preds), torch.tensor(0.0).type_as(box_preds))
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = {'box3d_lidar': final_box_preds[mask], 'scores': final_scores[mask], 'label_preds': label_preds[mask], 'metadata': meta}
                else:
                    predictions_dict = {'box3d_lidar': final_box_preds, 'scores': final_scores, 'label_preds': label_preds, 'metadata': meta}
            else:
                dtype = batch_reg_preds.dtype
                device = batch_reg_preds.device
                predictions_dict = {'box3d_lidar': torch.zeros([0, self.anchor_dim], dtype=dtype, device=device), 'scores': torch.zeros([0], dtype=dtype, device=device), 'label_preds': torch.zeros([0], dtype=top_labels.dtype, device=device), 'metadata': meta}
            predictions_dicts.append(predictions_dict)
        return predictions_dicts


class BaseDetector(nn.Module):
    """Base class for detectors"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_reader(self):
        return hasattr(self, 'reader') and self.reader is not None

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, **kwargs):
        pass

    def forward(self, example, return_loss=True, **kwargs):
        pass


DETECTORS = Registry('detector')


class SingleStageDetector(BaseDetector):

    def __init__(self, reader, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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

    def extract_feat(self, data):
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, example):
        x = self.extract_feat(example)
        outs = self.bbox_head(x)
        return outs
    """
    def simple_test(self, example, example_meta, rescale=False):
        x = self.extract_feat(example)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (example_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]
    """

    def aug_test(self, example, rescale=False):
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        pass

    def predict(self, example, preds_dicts):
        pass


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
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    loss = F.cross_entropy(pred, label, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, reduction='mean')[None]


class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert use_sigmoid is False or use_mask is False
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls


class SigmoidFocalLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes, gamma, alpha)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss, num_classes, gamma, alpha)
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


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """
    assert mode in ['iou', 'iof']
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)
    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
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
        loss = self.loss_weight * iou_loss(pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss


class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, reduction='mean', code_weights=None, codewise=True, loss_weight=1.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self._sigma = sigma
        self._code_weights = None
        self._codewise = codewise
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(self, prediction_tensor, target_tensor, weights=None):
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
            diff = self._code_weights.view(1, 1, -1) * diff
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


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(gamma={}, alpha={})'.format(self.gamma, self.alpha)
        return tmpstr


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


class WeightedSoftmaxClassificationLoss(nn.Module):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0, loss_weight=1.0, name=''):
        """Constructor.

        Args:
        logit_scale: When this value is high, the prediction is "diffused" and
                    when this value is low, the prediction is made peakier.
                    (default 1.0)

        """
        super(WeightedSoftmaxClassificationLoss, self).__init__()
        self.name = name
        self._loss_weight = loss_weight
        self._logit_scale = logit_scale

    def forward(self, prediction_tensor, target_tensor, weights):
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


mse_loss = weighted_loss(F.mse_loss)


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


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.
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
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None, activation='relu', inplace=True, order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])
        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x


NECKS = Registry('neck')


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
    :Example:
        class MyModule1(nn.Module)
            # Convert x and y to fp16
            @auto_fp16()
            def forward(self, x, y):
                pass
        class MyModule2(nn.Module):
            # convert pred to fp16
            @auto_fp16(apply_to=('pred', ))
            def do_something(self, pred, others):
                pass
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


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, add_extra_convs=False, extra_convs_on_inputs=True, relu_before_extra_convs=False, no_norm_on_lateral=False, conv_cfg=None, norm_cfg=None, activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
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
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg if not self.no_norm_on_lateral else None, activation=self.activation, inplace=False)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=False)
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
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
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


class RPN(nn.Module):

    def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters, num_input_features, norm_cfg=None, name='rpn', logger=None, **kwargs):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features
        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=0.001, momentum=0.01)
        self._norm_cfg = norm_cfg
        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)
        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)
        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(self._upsample_strides[i] / np.prod(self._layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]
        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []
        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(in_filters[i], self._num_filters[i], layer_num, stride=self._layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = self._upsample_strides[i - self._upsample_start_idx]
                if stride > 1:
                    deblock = Sequential(nn.ConvTranspose2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False), build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx])[1], nn.ReLU())
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(nn.Conv2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False), build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx])[1], nn.ReLU())
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        logger.info('Finish RPN Initialization')

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(nn.ZeroPad2d(1), nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False), build_norm_layer(self._norm_cfg, planes)[1], nn.ReLU())
        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self._norm_cfg, planes)[1])
            block.add(nn.ReLU())
        return block, planes

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x


class PointModule(nn.Module):

    def __init__(self, num_input_features, layers=[1024, 128], norm_cfg=None, name='rpn', logger=None, **kwargs):
        super(PointModule, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=0.001, momentum=0.01)
        self._norm_cfg = norm_cfg
        blocks = [nn.Conv2d(num_input_features, layers[0], 1, bias=False), build_norm_layer(self._norm_cfg, layers[0])[1], nn.ReLU(), nn.Conv2d(layers[0], layers[1], 1, bias=False), build_norm_layer(self._norm_cfg, layers[1])[1], nn.ReLU()]
        self.pn = nn.ModuleList(blocks)
        self.out = nn.MaxPool1d(3, stride=1, padding=1)

    def forward(self, x):
        x = x.flatten(1, -1)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        for l in self.pn:
            x = l(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.out(x).view(x.shape[0], x.shape[2], 1, 1)
        return x


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


READERS = Registry('reader')


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

    def __init__(self, num_input_features=4, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1), norm_cfg=None):
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
        assert len(num_filters) > 0
        self.num_input = num_input_features
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
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer))
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


class PointPillarsScatter(nn.Module):

    def __init__(self, num_input_features=64, norm_cfg=None, name='PointPillarsScatter', **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.name = 'PointPillarsScatter'
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):
        self.nx = input_shape[0]
        self.ny = input_shape[1]
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

    def __init__(self, num_input_features=4, use_norm=True, num_filters=[32, 128], with_distance=False, voxel_size=(0.2, 0.2, 4), name='VoxelFeatureExtractor'):
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

    def __init__(self, num_input_features=4, use_norm=True, num_filters=[32, 128], with_distance=False, voxel_size=(0.2, 0.2, 4), name='VoxelFeatureExtractor'):
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


class VFEV3_ablation(nn.Module):

    def __init__(self, num_input_features=4, norm_cfg=None, name='VFEV3_ablation'):
        super(VFEV3_ablation, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, [0, 1, 3]].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        points_mean = torch.cat([points_mean, 1.0 / num_voxels.view(-1, 1)], dim=1)
        return points_mean.contiguous()


class VoxelFeatureExtractorV3(nn.Module):

    def __init__(self, num_input_features=4, norm_cfg=None, name='VoxelFeatureExtractorV3'):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class SimpleVoxel(nn.Module):
    """Simple voxel encoder. only keep r, z and reflection feature.
    """

    def __init__(self, num_input_features=4, norm_cfg=None, name='SimpleVoxel'):
        super(SimpleVoxel, self).__init__()
        self.num_input_features = num_input_features
        self.name = name

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        feature = torch.norm(points_mean[:, :2], p=2, dim=1, keepdim=True)
        res = torch.cat([feature, points_mean[:, 2:self.num_input_features]], dim=1)
        return res


class GroupNorm(torch.nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-05, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Aggregation(nn.Module):

    def __init__(self, num_channel, name=''):
        super(Aggregation, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, align_feature, feature):
        align_conv1 = self.conv1(align_feature)
        align_conv2 = self.conv2(align_conv1)
        align_conv3 = self.conv3(align_conv2)
        feature_conv1 = self.conv1(feature)
        feature_conv2 = self.conv2(feature_conv1)
        feature_conv3 = self.conv3(feature_conv2)
        weights = torch.cat([align_conv3, feature_conv3], dim=1)
        weights = torch.softmax(weights, dim=1)
        weights_slice = torch.split(weights, 1, dim=1)
        aggregation = weights_slice[0] * align_feature + weights_slice[1] * feature
        return aggregation


class AlignFeatureFunction(Function):

    def __init__(self, weight_width, weight_height):
        super(AlignFeatureFunction, self).__init__()
        self.weight_width = weight_width
        self.weight_height = weight_height

    def forward(self, data, weight):
        self.save_for_backward(data, weight)
        N = data.size(0)
        C = data.size(1)
        H = data.size(2)
        W = data.size(3)
        output = data.new_zeros(N, C, H, W)
        if data.is_cuda:
            align_feature_cuda.forward(data, weight, self.weight_height, self.weight_width, output)
        else:
            raise NotImplementedError
        return output

    def backward(self, grad_output):
        data, weight = self.saved_variables
        N, C, H, W = data.size()
        Weight_Size = weight.size(1)
        grad_data = data.new_zeros(N, C, H, W)
        grad_weight = weight.new_zeros(N, Weight_Size, H, W)
        if data.is_cuda:
            align_feature_cuda.backward(grad_output, data, weight, self.weight_height, self.weight_width, N, C, Weight_Size, H, W, grad_data, grad_weight)
        else:
            raise NotImplementedError
        return grad_data, grad_weight


def align_feature(data, weight, weight_height, weight_width):
    align_feature_function = AlignFeatureFunction(weight_height, weight_width)
    return align_feature_function(data, weight)


class AlignFeature(Module):

    def __init__(self, weight_height, weight_width):
        super(AlignFeature, self).__init__()
        self.weight_height = weight_height
        self.weight_width = weight_width

    def forward(self, data, weight):
        return align_feature(data, weight, self.weight_height, self.weight_width)


class CorrelationFunction(Function):

    def __init__(self, kernel_size, patch_size, stride, padding, dilation_patch):
        super(CorrelationFunction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation_patch = _pair(dilation_patch)

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride
        output = correlation_cuda.forward(input1, input2, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW)
        return output

    @once_differentiable
    def backward(self, grad_output):
        input1, input2 = self.saved_variables
        kH, kW = self.kernel_size
        patchH, patchW = self.patch_size
        padH, padW = self.padding
        dilation_patchH, dilation_patchW = self.dilation_patch
        dH, dW = self.stride
        grad_input1, grad_input2 = correlation_cuda.backward(input1, input2, grad_output, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW)
        return grad_input1, grad_input2


def correlation(input1, input2, kernel_size=1, patch_size=1, stride=1, padding=0, dilation_patch=1):
    correlation_function = CorrelationFunction(kernel_size, patch_size, stride, padding, dilation_patch)
    return correlation_function(input1, input2)


class Correlation(Module):

    def __init__(self, kernel_size=1, patch_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(Correlation, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return correlation(input1, input2, self.kernel_size, self.patch_size, self.stride, self.padding, self.dilation_patch)


class Align_Feature_and_Aggregation(nn.Module):

    def __init__(self, num_channel, neighbor=9, name=''):
        super(Align_Feature_and_Aggregation, self).__init__()
        self.num_channel = num_channel
        self.embed_keyframe_conv = nn.Conv2d(num_channel, 64, 1)
        self.embed_current_conv = nn.Conv2d(num_channel, 64, 1)
        self.align_feature = AlignFeature(neighbor, neighbor)
        self.correlation = Correlation(kernel_size=1, patch_size=neighbor, stride=1, padding=0, dilation=1, dilation_patch=1)
        self.aggregation = Aggregation(num_channel, name='Aggregation_Module')

    def forward(self, feature_select, feature_current):
        embed_feature_select = self.embed_keyframe_conv(feature_select)
        embed_feature_current = self.embed_current_conv(feature_current)
        weights = self.correlation(embed_feature_current, embed_feature_select)
        weights = weights.reshape([weights.shape[0], -1, weights.shape[3], weights.shape[4]])
        weights = torch.softmax(weights, dim=1)
        align_feature = self.align_feature(feature_select, weights)
        aggregation = self.aggregation(align_feature, feature_current)
        return aggregation


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None) ->(torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, sample_uniformly: bool=False):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz)


class PointnetSAModuleVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pooling: str='max', sigma: float=None, normalize_xyz: bool=False, sample_uniformly: bool=False, ret_unique_cnt: bool=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)
        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, inds: torch.Tensor=None) ->(torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)
        new_features = self.mlp_module(grouped_features)
        if self.pooling == 'max':
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'rbf':
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1, keepdim=False) / self.sigma ** 2 / 2)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample)
        new_features = new_features.squeeze(-1)
        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


class PointnetSAModuleMSGVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(self, *, mlps: List[List[int]], npoint: int, radii: List[float], nsamples: List[int], bn: bool=True, use_xyz: bool=True, sample_uniformly: bool=False):
        super().__init__()
        assert len(mlps) == len(nsamples) == len(radii)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, inds: torch.Tensor=None) ->(torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, C) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), inds


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool=True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) ->torch.Tensor:
        """
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class PointnetLFPModuleMSG(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    learnable feature propagation layer."""

    def __init__(self, *, mlps: List[List[int]], radii: List[float], nsamples: List[int], post_mlp: List[int], bn: bool=True, use_xyz: bool=True, sample_uniformly: bool=False):
        super().__init__()
        assert len(mlps) == len(nsamples) == len(radii)
        self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz2: torch.Tensor, xyz1: torch.Tensor, features2: torch.Tensor, features1: torch.Tensor) ->torch.Tensor:
        """ Propagate features from xyz1 to xyz2.
        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), N1) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz1, xyz2, features1)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            if features2 is not None:
                new_features = torch.cat([new_features, features2], dim=1)
            new_features = new_features.unsqueeze(-1)
            new_features = self.post_mlp(new_features)
            new_features_list.append(new_features)
        return torch.cat(new_features_list, dim=1).squeeze(-1)


class RandomDropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        """

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features, idx):
        """

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = idx, N
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
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
        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name=''):
        super().__init__()
        bias = bias and not bn
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv2d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str=''):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name)


class SharedMLP(nn.Sequential):

    def __init__(self, args: List[int], *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str=''):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact))


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str=''):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name)


class Conv3d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int, int]=(1, 1, 1), stride: Tuple[int, int, int]=(1, 1, 1), padding: Tuple[int, int, int]=(0, 0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str=''):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv3d, batch_norm=BatchNorm3d, bias=bias, preact=preact, name=name)


class FC(nn.Sequential):

    def __init__(self, in_size: int, out_size: int, *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'fc', fc)
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)


class RoIFunction(Function):

    @staticmethod
    def forward(ctx, inputs, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio):
        ctx.save_for_backward(rois)
        ctx.output_size = _pair((pooled_height, pooled_width))
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = inputs.size()
        output = RRoI.forward(inputs, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        batch_size, channels, height, width = ctx.input_shape
        grad_input = RRoI.backward(grad_output, rois, spatial_scale, output_size[0], output_size[1], batch_size, channels, height, width, sampling_ratio)
        return grad_input, None, None, None, None, None


class RotateRoIAlign(nn.Module):

    def __init__(self, output_size, scale, ratio):
        super(RotateRoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = scale
        self.sampling_ratio = ratio

    def forward(self, inputs, rois):
        return RoIFunction.apply(inputs, rois, self.output_size[0], self.output_size[1], self.spatial_scale, self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio' + str(self.sampling_ratio)
        tmpstr += ')'
        return tmpstr


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(args[0].__class__.__name__, func.__name__, args[0].datatype))
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
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

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

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


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
                synchronize_stream(output[i * chunk_size + j], [devices[i]], [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.device(devices[0]):
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


class MegDataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class MegDistributedDataParallel(nn.Module):

    def __init__(self, module, dim=0, broadcast_buffers=True, bucket_cap_mb=25):
        super(MegDistributedDataParallel, self).__init__()
        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers
        self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024
        self._sync_params()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states, self.broadcast_bucket_size)
        if self.broadcast_buffers:
            if torch.__version__ < '1.0':
                buffers = [b.data for b in self.module._all_buffers()]
            else:
                buffers = [b.data for b in self.module.buffers()]
            if len(buffers) > 0:
                self._dist_broadcast_coalesced(buffers, self.broadcast_bucket_size)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Aggregation,
     lambda: ([], {'num_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BalancedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BaseDetector,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BatchNorm1d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm3d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvModule,
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
    (DistributedSyncBN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Empty,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupNorm,
     lambda: ([], {'num_channels': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Head,
     lambda: ([], {'num_input': 4, 'num_pred': 4, 'num_cls': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (L2Norm,
     lambda: ([], {'n_dims': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PointModule,
     lambda: ([], {'num_input_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {'depth': 18}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SimpleVoxel,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (VFELayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (VFEV3_ablation,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (VoxelFeatureExtractorV3,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (WeightedSmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_poodarchu_Det3D(_paritybench_base):
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

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

