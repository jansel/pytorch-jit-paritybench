import sys
_module = sys.modules[__name__]
del sys
setup = _module
timesformer = _module
config = _module
defaults = _module
datasets = _module
build = _module
cv2_transform = _module
decoder = _module
kinetics = _module
loader = _module
multigrid_helper = _module
ssv2 = _module
transform = _module
utils = _module
video_container = _module
models = _module
batchnorm_helper = _module
build = _module
conv2d_same = _module
custom_video_model_builder = _module
features = _module
head_helper = _module
helpers = _module
linear = _module
losses = _module
nonlocal_helper = _module
operators = _module
optimizer = _module
resnet_helper = _module
stem_helper = _module
video_model_builder = _module
vit = _module
vit_utils = _module
ava_eval_helper = _module
ava_evaluation = _module
label_map_util = _module
metrics = _module
np_box_list = _module
np_box_list_ops = _module
np_box_mask_list = _module
np_box_mask_list_ops = _module
np_box_ops = _module
np_mask_ops = _module
object_detection_evaluation = _module
per_image_evaluation = _module
standard_fields = _module
benchmark = _module
bn_helper = _module
c2_model_loading = _module
checkpoint = _module
distributed = _module
env = _module
logging = _module
lr_policy = _module
meters = _module
metrics = _module
misc = _module
multigrid = _module
multiprocessing = _module
parser = _module
weight_init_helper = _module
visualization = _module
tensorboard_vis = _module
utils = _module
run_net = _module
submit = _module
test_net = _module
train_net = _module
visualization = _module

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


import math


import numpy as np


import random


import torch


import torchvision.io as io


import torch.utils.data


import itertools


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import Sampler


from itertools import chain as chain


import logging


import time


from collections import defaultdict


from functools import partial


import torch.distributed as dist


import torch.nn as nn


from torch.autograd.function import Function


import torch.nn.functional as F


from typing import Tuple


from typing import Optional


from typing import List


from collections import OrderedDict


from copy import deepcopy


from typing import Dict


from typing import Callable


import torch.utils.model_zoo as model_zoo


from torch import nn as nn


from torch import einsum


from torch.nn.modules.module import Module


from torch.nn.modules.activation import MultiheadAttention


from torch.nn import ReplicationPad3d


import copy


import warnings


from itertools import repeat


import functools


from collections import deque


from sklearn.metrics import average_precision_score


from matplotlib import pyplot as plt


from torch import nn


import logging as log


import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter


from torchvision.utils import make_grid


from sklearn.metrics import confusion_matrix


import scipy.io


class SubBatchNorm3d(nn.Module):
    """
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm3d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): other arguments.
        """
        super(SubBatchNorm3d, self).__init__()
        self.num_splits = num_splits
        num_features = args['num_features']
        if args.get('affine', True):
            self.affine = True
            args['affine'] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm3d(**args)
        args['num_features'] = num_features * num_splits
        self.split_bn = nn.BatchNorm3d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = stds.view(n, -1).sum(0) / n + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var. Call this before eval.
        """
        if self.split_bn.track_running_stats:
            self.bn.running_mean.data, self.bn.running_var.data = self._get_aggregated_mean_std(self.split_bn.running_mean, self.split_bn.running_var, self.num_splits)

    def forward(self, x):
        if self.training:
            n, c, t, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, t, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, t, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x


class GroupGather(Function):
    """
    GroupGather performs all gather on each of the local process/ GPU groups.
    """

    @staticmethod
    def forward(ctx, input, num_sync_devices, num_groups):
        """
        Perform forwarding, gathering the stats across different process/ GPU
        group.
        """
        ctx.num_sync_devices = num_sync_devices
        ctx.num_groups = num_groups
        input_list = [torch.zeros_like(input) for k in range(du.get_local_size())]
        dist.all_gather(input_list, input, async_op=False, group=du._LOCAL_PROCESS_GROUP)
        inputs = torch.stack(input_list, dim=0)
        if num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // num_sync_devices
            inputs = inputs[group_idx * num_sync_devices:(group_idx + 1) * num_sync_devices]
        inputs = torch.sum(inputs, dim=0)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Perform backwarding, gathering the gradients across different process/ GPU
        group.
        """
        grad_output_list = [torch.zeros_like(grad_output) for k in range(du.get_local_size())]
        dist.all_gather(grad_output_list, grad_output, async_op=False, group=du._LOCAL_PROCESS_GROUP)
        grads = torch.stack(grad_output_list, dim=0)
        if ctx.num_groups > 1:
            rank = du.get_local_rank()
            group_idx = rank // ctx.num_sync_devices
            grads = grads[group_idx * ctx.num_sync_devices:(group_idx + 1) * ctx.num_sync_devices]
        grads = torch.sum(grads, dim=0)
        return grads, None, None


class NaiveSyncBatchNorm3d(nn.BatchNorm3d):

    def __init__(self, num_sync_devices, **args):
        """
        Naive version of Synchronized 3D BatchNorm.
        Args:
            num_sync_devices (int): number of device to sync.
            args (list): other arguments.
        """
        self.num_sync_devices = num_sync_devices
        if self.num_sync_devices > 0:
            assert du.get_local_size() % self.num_sync_devices == 0, (du.get_local_size(), self.num_sync_devices)
            self.num_groups = du.get_local_size() // self.num_sync_devices
        else:
            self.num_sync_devices = du.get_local_size()
            self.num_groups = 1
        super(NaiveSyncBatchNorm3d, self).__init__(**args)

    def forward(self, input):
        if du.get_local_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBatchNorm does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3, 4])
        meansqr = torch.mean(input * input, dim=[0, 2, 3, 4])
        vec = torch.cat([mean, meansqr], dim=0)
        vec = GroupGather.apply(vec, self.num_sync_devices, self.num_groups) * (1.0 / self.num_sync_devices)
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return input * scale + bias


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((int(math.ceil(x // s)) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k, s, d=(1, 1), value=0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def conv2d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor]=None, stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), dilation: Tuple[int, int]=(1, 1), groups: int=1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class FeatureInfo:

    def __init__(self, feature_info: List[Dict], out_indices: Tuple[int]):
        prev_reduction = 1
        for fi in feature_info:
            assert 'num_chs' in fi and fi['num_chs'] > 0
            assert 'reduction' in fi and fi['reduction'] >= prev_reduction
            prev_reduction = fi['reduction']
            assert 'module' in fi
        self.out_indices = out_indices
        self.info = feature_info

    def from_other(self, out_indices: Tuple[int]):
        return FeatureInfo(deepcopy(self.info), out_indices)

    def get(self, key, idx=None):
        """ Get value by key at specified index (indices)
        if idx == None, returns value for key at each output index
        if idx is an integer, return value for that feature module index (ignoring output indices)
        if idx is a list/tupple, return value for each module index (ignoring output indices)
        """
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [self.info[i][key] for i in idx]
        else:
            return self.info[idx][key]

    def get_dicts(self, keys=None, idx=None):
        """ return info dicts for specified keys (or all if None) at specified indices (or out_indices if None)
        """
        if idx is None:
            if keys is None:
                return [self.info[i] for i in self.out_indices]
            else:
                return [{k: self.info[i][k] for k in keys} for i in self.out_indices]
        if isinstance(idx, (tuple, list)):
            return [(self.info[i] if keys is None else {k: self.info[i][k] for k in keys}) for i in idx]
        else:
            return self.info[idx] if keys is None else {k: self.info[idx][k] for k in keys}

    def channels(self, idx=None):
        """ feature channels accessor
        """
        return self.get('num_chs', idx)

    def reduction(self, idx=None):
        """ feature reduction (output stride) accessor
        """
        return self.get('reduction', idx)

    def module_name(self, idx=None):
        """ feature module name accessor
        """
        return self.get('module', idx)

    def __getitem__(self, item):
        return self.info[item]

    def __len__(self):
        return len(self.info)


def _get_feature_info(net, out_indices):
    feature_info = getattr(net, 'feature_info')
    if isinstance(feature_info, FeatureInfo):
        return feature_info.from_other(out_indices)
    elif isinstance(feature_info, (list, tuple)):
        return FeatureInfo(net.feature_info, out_indices)
    else:
        assert False, 'Provided feature_info is not valid'


def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i] if out_map is not None else feature_info.out_indices[i]
    return return_layers


def _module_list(module, flatten_sequential=False):
    ml = []
    for name, module in module.named_children():
        if flatten_sequential and isinstance(module, nn.Sequential):
            for child_name, child_module in module.named_children():
                combined = [name, child_name]
                ml.append(('_'.join(combined), '.'.join(combined), child_module))
        else:
            ml.append((name, name, module))
    return ml


class FeatureDictNet(nn.ModuleDict):
    """ Feature extractor with OrderedDict return
    Wrap a model and extract features as specified by the out indices, the network is
    partially re-built from contained modules.
    There is a strong assumption that the modules have been registered into the model in the same
    order as they are used. There should be no reuse of the same nn.Module more than once, including
    trivial modules like `self.relu = nn.ReLU`.
    Only submodules that are directly assigned to the model class (`model.feature1`) or at most
    one Sequential container deep (`model.features.1`, with flatten_sequent=True) can be captured.
    All Sequential containers that are directly assigned to the original model will have their
    modules assigned to this module with the name `model.features.1` being changed to `model.features_1`
    Arguments:
        model (nn.Module): model from which we will extract the features
        out_indices (tuple[int]): model output indices to extract features for
        out_map (sequence): list or tuple specifying desired return id for each out index,
            otherwise str(index) is used
        feature_concat (bool): whether to concatenate intermediate features that are lists or tuples
            vs select element [0]
        flatten_sequential (bool): whether to flatten sequential modules assigned to model
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.concat = feature_concat
        self.return_layers = {}
        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers), f'Return layers ({remaining}) are not present in model'
        self.update(layers)

    def _collect(self, x) ->Dict[str, torch.Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_id = self.return_layers[name]
                if isinstance(x, (tuple, list)):
                    out[out_id] = torch.cat(x, 1) if self.concat else x[0]
                else:
                    out[out_id] = x
        return out

    def forward(self, x) ->Dict[str, torch.Tensor]:
        return self._collect(x)


class FeatureListNet(FeatureDictNet):
    """ Feature extractor with list return
    See docstring for FeatureDictNet above, this class exists only to appease Torchscript typing constraints.
    In eager Python we could have returned List[Tensor] vs Dict[id, Tensor] based on a member bool.
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, feature_concat=False, flatten_sequential=False):
        super(FeatureListNet, self).__init__(model, out_indices=out_indices, out_map=out_map, feature_concat=feature_concat, flatten_sequential=flatten_sequential)

    def forward(self, x) ->List[torch.Tensor]:
        return list(self._collect(x).values())


class FeatureHooks:
    """ Feature Hook Helper
    This module helps with the setup and extraction of hooks for extracting features from
    internal nodes in a model by node name. This works quite well in eager Python but needs
    redesign for torcscript.
    """

    def __init__(self, hooks, named_modules, out_map=None, default_hook_type='forward'):
        modules = {k: v for k, v in named_modules}
        for i, h in enumerate(hooks):
            hook_name = h['module']
            m = modules[hook_name]
            hook_id = out_map[i] if out_map else hook_name
            hook_fn = partial(self._collect_output_hook, hook_id)
            hook_type = h['hook_type'] if 'hook_type' in h else default_hook_type
            if hook_type == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif hook_type == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, 'Unsupported hook type'
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, hook_id, *args):
        x = args[-1]
        if isinstance(x, tuple):
            x = x[0]
        self._feature_outputs[x.device][hook_id] = x

    def get_output(self, device) ->Dict[str, torch.tensor]:
        output = self._feature_outputs[device]
        self._feature_outputs[device] = OrderedDict()
        return output


class FeatureHookNet(nn.ModuleDict):
    """ FeatureHookNet
    Wrap a model and extract features specified by the out indices using forward/forward-pre hooks.
    If `no_rewrite` is True, features are extracted via hooks without modifying the underlying
    network in any way.
    If `no_rewrite` is False, the model will be re-written as in the
    FeatureList/FeatureDict case by folding first to second (Sequential only) level modules into this one.
    FIXME this does not currently work with Torchscript, see FeatureHooks class
    """

    def __init__(self, model, out_indices=(0, 1, 2, 3, 4), out_map=None, out_as_dict=False, no_rewrite=False, feature_concat=False, flatten_sequential=False, default_hook_type='forward'):
        super(FeatureHookNet, self).__init__()
        assert not torch.jit.is_scripting()
        self.feature_info = _get_feature_info(model, out_indices)
        self.out_as_dict = out_as_dict
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential)
            remaining = {f['module']: (f['hook_type'] if 'hook_type' in f else default_hook_type) for f in self.feature_info.get_dicts()}
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_modules(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'Return layers ({remaining}) are not present in model'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_modules(), out_map=out_map)

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        out = self.hooks.get_output(x.device)
        return out if self.out_as_dict else list(out.values())


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(self, dim_in, num_classes, pool_size, dropout_rate=0.0, act_func='softmax'):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, 'pathway dimensions are not consistent.'
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module('pathway{}_avgpool'.format(pathway), avg_pool)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        if act_func == 'softmax':
            self.act = nn.Softmax(dim=4)
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(act_func))

    def forward(self, inputs):
        assert len(inputs) == self.num_pathways, 'Input tensor does not contain {} pathway'.format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, 'pathway{}_avgpool'.format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        return x


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(self, dim_in, dim_inner, dim_out, num_classes, pool_size, dropout_rate=0.0, act_func='softmax', inplace_relu=True, eps=1e-05, bn_mmt=0.1, norm_module=nn.BatchNorm3d, bn_lin5_on=False):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):
        self.conv_5 = nn.Conv3d(dim_in, dim_inner, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.conv_5_bn = norm_module(num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt)
        self.conv_5_relu = nn.ReLU(self.inplace_relu)
        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)
        self.lin_5 = nn.Conv3d(dim_inner, dim_out, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.lin_5_relu = nn.ReLU(self.inplace_relu)
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)
        if self.act_func == 'softmax':
            self.act = nn.Softmax(dim=4)
        elif self.act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activationfunction.'.format(self.act_func))

    def forward(self, inputs):
        assert len(inputs) == 1, 'Input tensor does not contain 1 pathway'
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)
        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)
        x = x.permute((0, 2, 3, 4, 1))
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        return x


class Linear(nn.Linear):

    def forward(self, input: torch.Tensor) ->torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias if self.bias is not None else None
            return F.linear(input, self.weight, bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


class Nonlocal(nn.Module):
    """
    Builds Non-local Neural Networks as a generic family of building
    blocks for capturing long-range dependencies. Non-local Network
    computes the response at a position as a weighted sum of the
    features at all positions. This building block can be plugged into
    many computer vision architectures.
    More details in the paper: https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, dim, dim_inner, pool_size=None, instantiation='softmax', zero_init_final_conv=False, zero_init_final_norm=True, norm_eps=1e-05, norm_momentum=0.1, norm_module=nn.BatchNorm3d):
        """
        Args:
            dim (int): number of dimension for the input.
            dim_inner (int): number of dimension inside of the Non-local block.
            pool_size (list): the kernel size of spatial temporal pooling,
                temporal pool kernel size, spatial pool kernel size, spatial
                pool kernel size in order. By default pool_size is None,
                then there would be no pooling used.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            zero_init_final_conv (bool): If true, zero initializing the final
                convolution of the Non-local block.
            zero_init_final_norm (bool):
                If true, zero initializing the final batch norm of the Non-local
                block.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = False if pool_size is None else any(size > 1 for size in pool_size)
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(zero_init_final_conv, zero_init_final_norm, norm_module)

    def _construct_nonlocal(self, zero_init_final_conv, zero_init_final_norm, norm_module):
        self.conv_theta = nn.Conv3d(self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_phi = nn.Conv3d(self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_g = nn.Conv3d(self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv3d(self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv_out.zero_init = zero_init_final_conv
        self.bn = norm_module(num_features=self.dim, eps=self.norm_eps, momentum=self.norm_momentum)
        self.bn.transform_final_bn = zero_init_final_norm
        if self.use_pool:
            self.pool = nn.MaxPool3d(kernel_size=self.pool_size, stride=self.pool_size, padding=[0, 0, 0])

    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()
        theta = self.conv_theta(x)
        if self.use_pool:
            x = self.pool(x)
        phi = self.conv_phi(x)
        g = self.conv_g(x)
        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)
        theta_phi = torch.einsum('nct,ncp->ntp', (theta, phi))
        if self.instantiation == 'softmax':
            theta_phi = theta_phi * self.dim_inner ** -0.5
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == 'dot_product':
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError('Unknown norm type {}'.format(self.instantiation))
        theta_phi_g = torch.einsum('ntg,ncg->nct', (theta_phi, g))
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)
        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        return x_identity + p


class SwishEfficient(torch.autograd.Function):
    """Swish activation function: x * sigmoid(x)."""

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
            divisor (int): the new width should be dividable by divisor.
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)
        self.fc2_sig = nn.Sigmoid()

    def forward(self, x):
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x


class BasicTransform(nn.Module):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, dim_inner=None, num_groups=1, stride_1x1=None, inplace_relu=True, eps=1e-05, bn_mmt=0.1, norm_module=nn.BatchNorm3d, block_idx=0):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the basic block.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, norm_module)

    def _construct(self, dim_in, dim_out, stride, norm_module):
        self.a = nn.Conv3d(dim_in, dim_out, kernel_size=[self.temp_kernel_size, 3, 3], stride=[1, stride, stride], padding=[int(self.temp_kernel_size // 2), 1, 1], bias=False)
        self.a_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        self.b = nn.Conv3d(dim_out, dim_out, kernel_size=[1, 3, 3], stride=[1, 1, 1], padding=[0, 1, 1], bias=False)
        self.b_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.b_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x


class X3DTransform(nn.Module):
    """
    X3D transformation: 1x1x1, Tx3x3 (channelwise, num_groups=dim_in), 1x1x1,
        augmented with (optional) SE (squeeze-excitation) on the 3x3x3 output.
        T is the temporal kernel size (defaulting to 3)
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups, stride_1x1=False, inplace_relu=True, eps=1e-05, bn_mmt=0.1, dilation=1, norm_module=nn.BatchNorm3d, se_ratio=0.0625, swish_inner=True, block_idx=0):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            se_ratio (float): if > 0, apply SE to the Tx3x3 conv, with the SE
                channel dimensionality being se_ratio times the Tx3x3 conv dim.
            swish_inner (bool): if True, apply swish to the Tx3x3 conv, otherwise
                apply ReLU to the Tx3x3 conv.
        """
        super(X3DTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._se_ratio = se_ratio
        self._swish_inner = swish_inner
        self._stride_1x1 = stride_1x1
        self._block_idx = block_idx
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups, dilation, norm_module)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups, dilation, norm_module):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)
        self.a = nn.Conv3d(dim_in, dim_inner, kernel_size=[1, 1, 1], stride=[1, str1x1, str1x1], padding=[0, 0, 0], bias=False)
        self.a_bn = norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        self.b = nn.Conv3d(dim_inner, dim_inner, [self.temp_kernel_size, 3, 3], stride=[1, str3x3, str3x3], padding=[int(self.temp_kernel_size // 2), dilation, dilation], groups=num_groups, bias=False, dilation=[1, dilation, dilation])
        self.b_bn = norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        use_se = True if (self._block_idx + 1) % 2 else False
        if self._se_ratio > 0.0 and use_se:
            self.se = SE(dim_inner, self._se_ratio)
        if self._swish_inner:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=self._inplace_relu)
        self.c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
        self.c_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups, stride_1x1=False, inplace_relu=True, eps=1e-05, bn_mmt=0.1, dilation=1, norm_module=nn.BatchNorm3d, block_idx=0):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups, dilation, norm_module)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups, dilation, norm_module):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)
        self.a = nn.Conv3d(dim_in, dim_inner, kernel_size=[self.temp_kernel_size, 1, 1], stride=[1, str1x1, str1x1], padding=[int(self.temp_kernel_size // 2), 0, 0], bias=False)
        self.a_bn = norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        self.b = nn.Conv3d(dim_inner, dim_inner, [1, 3, 3], stride=[1, str3x3, str3x3], padding=[0, dilation, dilation], groups=num_groups, bias=False, dilation=[1, dilation, dilation])
        self.b_bn = norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)
        self.c = nn.Conv3d(dim_inner, dim_out, kernel_size=[1, 1, 1], stride=[1, 1, 1], padding=[0, 0, 0], bias=False)
        self.c_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(self, dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, num_groups=1, stride_1x1=False, inplace_relu=True, eps=1e-05, bn_mmt=0.1, dilation=1, norm_module=nn.BatchNorm3d, block_idx=0, drop_connect_rate=0.0):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, num_groups, stride_1x1, inplace_relu, dilation, norm_module, block_idx)

    def _construct(self, dim_in, dim_out, temp_kernel_size, stride, trans_func, dim_inner, num_groups, stride_1x1, inplace_relu, dilation, norm_module, block_idx):
        if dim_in != dim_out or stride != 1:
            self.branch1 = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=[1, stride, stride], padding=0, bias=False, dilation=1)
            self.branch1_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.branch2 = trans_func(dim_in, dim_out, temp_kernel_size, stride, dim_inner, num_groups, stride_1x1=stride_1x1, inplace_relu=inplace_relu, dilation=dilation, norm_module=norm_module, block_idx=block_idx)
        self.relu = nn.ReLU(self._inplace_relu)

    def _drop_connect(self, x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], 1, 1, 1, 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, 'branch1'):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {'bottleneck_transform': BottleneckTransform, 'basic_transform': BasicTransform, 'x3d_transform': X3DTransform}
    assert name in trans_funcs.keys(), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class ResStage(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, dim_in, dim_out, stride, temp_kernel_sizes, num_blocks, dim_inner, num_groups, num_block_temp_kernel, nonlocal_inds, nonlocal_group, nonlocal_pool, dilation, instantiation='softmax', trans_func_name='bottleneck_transform', stride_1x1=False, inplace_relu=True, norm_module=nn.BatchNorm3d, drop_connect_rate=0.0):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage, self).__init__()
        assert all(num_block_temp_kernel[i] <= num_blocks[i] for i in range(len(temp_kernel_sizes)))
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [((temp_kernel_sizes[i] * num_blocks[i])[:num_block_temp_kernel[i]] + [1] * (num_blocks[i] - num_block_temp_kernel[i])) for i in range(len(temp_kernel_sizes))]
        assert len({len(dim_in), len(dim_out), len(temp_kernel_sizes), len(stride), len(num_blocks), len(dim_inner), len(num_groups), len(num_block_temp_kernel), len(nonlocal_inds), len(nonlocal_group)}) == 1
        self.num_pathways = len(self.num_blocks)
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups, trans_func_name, stride_1x1, inplace_relu, nonlocal_inds, nonlocal_pool, instantiation, dilation, norm_module)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups, trans_func_name, stride_1x1, inplace_relu, nonlocal_inds, nonlocal_pool, instantiation, dilation, norm_module):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                trans_func = get_trans_func(trans_func_name)
                res_block = ResBlock(dim_in[pathway] if i == 0 else dim_out[pathway], dim_out[pathway], self.temp_kernel_sizes[pathway][i], stride[pathway] if i == 0 else 1, trans_func, dim_inner[pathway], num_groups[pathway], stride_1x1=stride_1x1, inplace_relu=inplace_relu, dilation=dilation[pathway], norm_module=norm_module, block_idx=i, drop_connect_rate=self._drop_connect_rate)
                self.add_module('pathway{}_res{}'.format(pathway, i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(dim_out[pathway], dim_out[pathway] // 2, nonlocal_pool[pathway], instantiation=instantiation, norm_module=norm_module)
                    self.add_module('pathway{}_nonlocal{}'.format(pathway, i), nln)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, 'pathway{}_res{}'.format(pathway, i))
                x = m(x)
                if hasattr(self, 'pathway{}_nonlocal{}'.format(pathway, i)):
                    nln = getattr(self, 'pathway{}_nonlocal{}'.format(pathway, i))
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[pathway] > 1:
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b * self.nonlocal_group[pathway], t // self.nonlocal_group[pathway], c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[pathway] > 1:
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)
        return output


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, eps=1e-05, bn_mmt=0.1, norm_module=nn.BatchNorm3d):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(dim_in, dim_out, self.kernel, stride=self.stride, padding=self.padding, bias=False)
        self.bn = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool_layer(x)
        return x


class X3DStem(nn.Module):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, eps=1e-05, bn_mmt=0.1, norm_module=nn.BatchNorm3d):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(X3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv_xy = nn.Conv3d(dim_in, dim_out, kernel_size=(1, self.kernel[1], self.kernel[2]), stride=(1, self.stride[1], self.stride[2]), padding=(0, self.padding[1], self.padding[2]), bias=False)
        self.conv = nn.Conv3d(dim_out, dim_out, kernel_size=(self.kernel[0], 1, 1), stride=(self.stride[0], 1, 1), padding=(self.padding[0], 0, 0), bias=False, groups=dim_out)
        self.bn = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def get_stem_func(name):
    """
    Retrieves the stem module by name.
    """
    trans_funcs = {'x3d_stem': X3DStem, 'basic_stem': ResNetBasicStem}
    assert name in trans_funcs.keys(), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(self, dim_in, dim_out, kernel, stride, padding, inplace_relu=True, eps=1e-05, bn_mmt=0.1, norm_module=nn.BatchNorm3d, stem_func_name='basic_stem'):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            stem_func_name (string): name of the the stem function applied on
                input to the network.
        """
        super(VideoModelStem, self).__init__()
        assert len({len(dim_in), len(dim_out), len(kernel), len(stride), len(padding)}) == 1, 'Input pathway dimensions are not consistent.'
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self._construct_stem(dim_in, dim_out, norm_module, stem_func_name)

    def _construct_stem(self, dim_in, dim_out, norm_module, stem_func_name):
        trans_func = get_stem_func(stem_func_name)
        for pathway in range(len(dim_in)):
            stem = trans_func(dim_in[pathway], dim_out[pathway], self.kernel[pathway], self.stride[pathway], self.padding[pathway], self.inplace_relu, self.eps, self.bn_mmt, norm_module)
            self.add_module('pathway{}_stem'.format(pathway), stem)

    def forward(self, x):
        assert len(x) == self.num_pathways, 'Input tensor does not contain {} pathway'.format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, 'pathway{}_stem'.format(pathway))
            x[pathway] = m(x[pathway])
        return x


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(self, dim_in, fusion_conv_channel_ratio, fusion_kernel, alpha, eps=1e-05, bn_mmt=0.1, inplace_relu=True, norm_module=nn.BatchNorm3d):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(dim_in, dim_in * fusion_conv_channel_ratio, kernel_size=[fusion_kernel, 1, 1], stride=[alpha, 1, 1], padding=[fusion_kernel // 2, 0, 0], bias=False)
        self.bn = norm_module(num_features=dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


_MODEL_STAGE_DEPTH = {(50): (3, 4, 6, 3), (101): (3, 4, 23, 3)}


_POOL1 = {'c2d': [[2, 1, 1]], 'c2d_nopool': [[1, 1, 1]], 'i3d': [[2, 1, 1]], 'i3d_nopool': [[1, 1, 1]], 'slow': [[1, 1, 1]], 'slowfast': [[1, 1, 1], [1, 1, 1]], 'x3d': [[1, 1, 1]]}


_TEMPORAL_KERNEL_BASIS = {'c2d': [[[1]], [[1]], [[1]], [[1]], [[1]]], 'c2d_nopool': [[[1]], [[1]], [[1]], [[1]], [[1]]], 'i3d': [[[5]], [[3]], [[3, 1]], [[3, 1]], [[1, 3]]], 'i3d_nopool': [[[5]], [[3]], [[3, 1]], [[3, 1]], [[1, 3]]], 'slow': [[[1]], [[1]], [[1]], [[3]], [[3]]], 'slowfast': [[[1], [5]], [[1], [3]], [[1], [3]], [[3], [3]], [[3], [3]]], 'x3d': [[[5]], [[3]], [[3]], [[3]], [[3]]]}


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    if cfg.BN.NORM_TYPE == 'batchnorm':
        return nn.BatchNorm3d
    elif cfg.BN.NORM_TYPE == 'sub_batchnorm':
        return partial(SubBatchNorm3d, num_splits=cfg.BN.NUM_SPLITS)
    elif cfg.BN.NORM_TYPE == 'sync_batchnorm':
        return partial(NaiveSyncBatchNorm3d, num_sync_devices=cfg.BN.NUM_SYNC_DEVICES)
    else:
        raise NotImplementedError('Norm type {} is not supported'.format(cfg.BN.NORM_TYPE))


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN)

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        d2, d3, d4, d5 = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        self.s1 = stem_helper.VideoModelStem(dim_in=cfg.DATA.INPUT_CHANNEL_NUM, dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV], kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]], stride=[[1, 2, 2]] * 2, padding=[[temp_kernel[0][0][0] // 2, 3, 3], [temp_kernel[0][1][0] // 2, 3, 3]], norm_module=self.norm_module)
        self.s1_fuse = FuseFastToSlow(width_per_group // cfg.SLOWFAST.BETA_INV, cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO, cfg.SLOWFAST.FUSION_KERNEL_SZ, cfg.SLOWFAST.ALPHA, norm_module=self.norm_module)
        self.s2 = resnet_helper.ResStage(dim_in=[width_per_group + width_per_group // out_dim_ratio, width_per_group // cfg.SLOWFAST.BETA_INV], dim_out=[width_per_group * 4, width_per_group * 4 // cfg.SLOWFAST.BETA_INV], dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV], temp_kernel_sizes=temp_kernel[1], stride=cfg.RESNET.SPATIAL_STRIDES[0], num_blocks=[d2] * 2, num_groups=[num_groups] * 2, num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0], nonlocal_inds=cfg.NONLOCAL.LOCATION[0], nonlocal_group=cfg.NONLOCAL.GROUP[0], nonlocal_pool=cfg.NONLOCAL.POOL[0], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, dilation=cfg.RESNET.SPATIAL_DILATIONS[0], norm_module=self.norm_module)
        self.s2_fuse = FuseFastToSlow(width_per_group * 4 // cfg.SLOWFAST.BETA_INV, cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO, cfg.SLOWFAST.FUSION_KERNEL_SZ, cfg.SLOWFAST.ALPHA, norm_module=self.norm_module)
        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(kernel_size=pool_size[pathway], stride=pool_size[pathway], padding=[0, 0, 0])
            self.add_module('pathway{}_pool'.format(pathway), pool)
        self.s3 = resnet_helper.ResStage(dim_in=[width_per_group * 4 + width_per_group * 4 // out_dim_ratio, width_per_group * 4 // cfg.SLOWFAST.BETA_INV], dim_out=[width_per_group * 8, width_per_group * 8 // cfg.SLOWFAST.BETA_INV], dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV], temp_kernel_sizes=temp_kernel[2], stride=cfg.RESNET.SPATIAL_STRIDES[1], num_blocks=[d3] * 2, num_groups=[num_groups] * 2, num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1], nonlocal_inds=cfg.NONLOCAL.LOCATION[1], nonlocal_group=cfg.NONLOCAL.GROUP[1], nonlocal_pool=cfg.NONLOCAL.POOL[1], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, dilation=cfg.RESNET.SPATIAL_DILATIONS[1], norm_module=self.norm_module)
        self.s3_fuse = FuseFastToSlow(width_per_group * 8 // cfg.SLOWFAST.BETA_INV, cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO, cfg.SLOWFAST.FUSION_KERNEL_SZ, cfg.SLOWFAST.ALPHA, norm_module=self.norm_module)
        self.s4 = resnet_helper.ResStage(dim_in=[width_per_group * 8 + width_per_group * 8 // out_dim_ratio, width_per_group * 8 // cfg.SLOWFAST.BETA_INV], dim_out=[width_per_group * 16, width_per_group * 16 // cfg.SLOWFAST.BETA_INV], dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV], temp_kernel_sizes=temp_kernel[3], stride=cfg.RESNET.SPATIAL_STRIDES[2], num_blocks=[d4] * 2, num_groups=[num_groups] * 2, num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2], nonlocal_inds=cfg.NONLOCAL.LOCATION[2], nonlocal_group=cfg.NONLOCAL.GROUP[2], nonlocal_pool=cfg.NONLOCAL.POOL[2], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, dilation=cfg.RESNET.SPATIAL_DILATIONS[2], norm_module=self.norm_module)
        self.s4_fuse = FuseFastToSlow(width_per_group * 16 // cfg.SLOWFAST.BETA_INV, cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO, cfg.SLOWFAST.FUSION_KERNEL_SZ, cfg.SLOWFAST.ALPHA, norm_module=self.norm_module)
        self.s5 = resnet_helper.ResStage(dim_in=[width_per_group * 16 + width_per_group * 16 // out_dim_ratio, width_per_group * 16 // cfg.SLOWFAST.BETA_INV], dim_out=[width_per_group * 32, width_per_group * 32 // cfg.SLOWFAST.BETA_INV], dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV], temp_kernel_sizes=temp_kernel[4], stride=cfg.RESNET.SPATIAL_STRIDES[3], num_blocks=[d5] * 2, num_groups=[num_groups] * 2, num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3], nonlocal_inds=cfg.NONLOCAL.LOCATION[3], nonlocal_group=cfg.NONLOCAL.GROUP[3], nonlocal_pool=cfg.NONLOCAL.POOL[3], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, dilation=cfg.RESNET.SPATIAL_DILATIONS[3], norm_module=self.norm_module)
        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(dim_in=[width_per_group * 32, width_per_group * 32 // cfg.SLOWFAST.BETA_INV], num_classes=cfg.MODEL.NUM_CLASSES, pool_size=[[cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0], 1, 1], [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1]], resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2, scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2, dropout_rate=cfg.MODEL.DROPOUT_RATE, act_func=cfg.MODEL.HEAD_ACT, aligned=cfg.DETECTION.ALIGNED)
        else:
            head = head_helper.ResNetBasicHead(dim_in=[width_per_group * 32, width_per_group * 32 // cfg.SLOWFAST.BETA_INV], num_classes=cfg.MODEL.NUM_CLASSES, pool_size=[None, None] if cfg.MULTIGRID.SHORT_CYCLE else [[cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0], cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1], cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2]], [cfg.DATA.NUM_FRAMES // pool_size[1][0], cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1], cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2]]], dropout_rate=cfg.MODEL.DROPOUT_RATE, act_func=cfg.MODEL.HEAD_ACT)
            self.head_name = 'head{}'.format(cfg.TASK)
            self.add_module(self.head_name, head)

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, 'pathway{}_pool'.format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        head = getattr(self, self.head_name)
        if self.enable_detection:
            x = head(x, bboxes)
        else:
            x = head(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        d2, d3, d4, d5 = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        self.s1 = stem_helper.VideoModelStem(dim_in=cfg.DATA.INPUT_CHANNEL_NUM, dim_out=[width_per_group], kernel=[temp_kernel[0][0] + [7, 7]], stride=[[1, 2, 2]], padding=[[temp_kernel[0][0][0] // 2, 3, 3]], norm_module=self.norm_module)
        self.s2 = resnet_helper.ResStage(dim_in=[width_per_group], dim_out=[width_per_group * 4], dim_inner=[dim_inner], temp_kernel_sizes=temp_kernel[1], stride=cfg.RESNET.SPATIAL_STRIDES[0], num_blocks=[d2], num_groups=[num_groups], num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0], nonlocal_inds=cfg.NONLOCAL.LOCATION[0], nonlocal_group=cfg.NONLOCAL.GROUP[0], nonlocal_pool=cfg.NONLOCAL.POOL[0], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, stride_1x1=cfg.RESNET.STRIDE_1X1, inplace_relu=cfg.RESNET.INPLACE_RELU, dilation=cfg.RESNET.SPATIAL_DILATIONS[0], norm_module=self.norm_module)
        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(kernel_size=pool_size[pathway], stride=pool_size[pathway], padding=[0, 0, 0])
            self.add_module('pathway{}_pool'.format(pathway), pool)
        self.s3 = resnet_helper.ResStage(dim_in=[width_per_group * 4], dim_out=[width_per_group * 8], dim_inner=[dim_inner * 2], temp_kernel_sizes=temp_kernel[2], stride=cfg.RESNET.SPATIAL_STRIDES[1], num_blocks=[d3], num_groups=[num_groups], num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1], nonlocal_inds=cfg.NONLOCAL.LOCATION[1], nonlocal_group=cfg.NONLOCAL.GROUP[1], nonlocal_pool=cfg.NONLOCAL.POOL[1], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, stride_1x1=cfg.RESNET.STRIDE_1X1, inplace_relu=cfg.RESNET.INPLACE_RELU, dilation=cfg.RESNET.SPATIAL_DILATIONS[1], norm_module=self.norm_module)
        self.s4 = resnet_helper.ResStage(dim_in=[width_per_group * 8], dim_out=[width_per_group * 16], dim_inner=[dim_inner * 4], temp_kernel_sizes=temp_kernel[3], stride=cfg.RESNET.SPATIAL_STRIDES[2], num_blocks=[d4], num_groups=[num_groups], num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2], nonlocal_inds=cfg.NONLOCAL.LOCATION[2], nonlocal_group=cfg.NONLOCAL.GROUP[2], nonlocal_pool=cfg.NONLOCAL.POOL[2], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, stride_1x1=cfg.RESNET.STRIDE_1X1, inplace_relu=cfg.RESNET.INPLACE_RELU, dilation=cfg.RESNET.SPATIAL_DILATIONS[2], norm_module=self.norm_module)
        self.s5 = resnet_helper.ResStage(dim_in=[width_per_group * 16], dim_out=[width_per_group * 32], dim_inner=[dim_inner * 8], temp_kernel_sizes=temp_kernel[4], stride=cfg.RESNET.SPATIAL_STRIDES[3], num_blocks=[d5], num_groups=[num_groups], num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3], nonlocal_inds=cfg.NONLOCAL.LOCATION[3], nonlocal_group=cfg.NONLOCAL.GROUP[3], nonlocal_pool=cfg.NONLOCAL.POOL[3], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, stride_1x1=cfg.RESNET.STRIDE_1X1, inplace_relu=cfg.RESNET.INPLACE_RELU, dilation=cfg.RESNET.SPATIAL_DILATIONS[3], norm_module=self.norm_module)
        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(dim_in=[width_per_group * 32], num_classes=cfg.MODEL.NUM_CLASSES, pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]], resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2], scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR], dropout_rate=cfg.MODEL.DROPOUT_RATE, act_func=cfg.MODEL.HEAD_ACT, aligned=cfg.DETECTION.ALIGNED)
        else:
            head = head_helper.ResNetBasicHead(dim_in=[width_per_group * 32], num_classes=cfg.MODEL.NUM_CLASSES, pool_size=[None, None] if cfg.MULTIGRID.SHORT_CYCLE else [[cfg.DATA.NUM_FRAMES // pool_size[0][0], cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1], cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2]]], dropout_rate=cfg.MODEL.DROPOUT_RATE, act_func=cfg.MODEL.HEAD_ACT)
            self.head_name = 'head{}'.format(cfg.TASK)
            self.add_module(self.head_name, head)

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, 'pathway{}_pool'.format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        head = getattr(self, self.head_name)
        if self.enable_detection:
            x = head(x, bboxes)
        else:
            x = head(x)
        return x


class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1
        self.dim_res2 = self._round_width(self.dim_c1, exp_stage, divisor=8) if cfg.X3D.SCALE_RES2 else self.dim_c1
        self.dim_res3 = self._round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = self._round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = self._round_width(self.dim_res4, exp_stage, divisor=8)
        self.block_basis = [[1, self.dim_res2, 2], [2, self.dim_res3, 2], [5, self.dim_res4, 2], [3, self.dim_res5, 2]]
        self._construct_network(cfg)
        init_helper.init_weights(self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN)

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width
        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        d2, d3, d4, d5 = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = self._round_width(self.dim_c1, w_mul)
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        self.s1 = stem_helper.VideoModelStem(dim_in=cfg.DATA.INPUT_CHANNEL_NUM, dim_out=[dim_res1], kernel=[temp_kernel[0][0] + [3, 3]], stride=[[1, 2, 2]], padding=[[temp_kernel[0][0][0] // 2, 1, 1]], norm_module=self.norm_module, stem_func_name='x3d_stem')
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = self._round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)
            n_rep = self._round_repeats(block[0], d_mul)
            prefix = 's{}'.format(stage + 2)
            s = resnet_helper.ResStage(dim_in=[dim_in], dim_out=[dim_out], dim_inner=[dim_inner], temp_kernel_sizes=temp_kernel[1], stride=[block[2]], num_blocks=[n_rep], num_groups=[dim_inner] if cfg.X3D.CHANNELWISE_3x3x3 else [num_groups], num_block_temp_kernel=[n_rep], nonlocal_inds=cfg.NONLOCAL.LOCATION[0], nonlocal_group=cfg.NONLOCAL.GROUP[0], nonlocal_pool=cfg.NONLOCAL.POOL[0], instantiation=cfg.NONLOCAL.INSTANTIATION, trans_func_name=cfg.RESNET.TRANS_FUNC, stride_1x1=cfg.RESNET.STRIDE_1X1, norm_module=self.norm_module, dilation=cfg.RESNET.SPATIAL_DILATIONS[stage], drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE * (stage + 2) / (len(self.block_basis) + 1))
            dim_in = dim_out
            self.add_module(prefix, s)
        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(dim_in=dim_out, dim_inner=dim_inner, dim_out=cfg.X3D.DIM_C5, num_classes=cfg.MODEL.NUM_CLASSES, pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz], dropout_rate=cfg.MODEL.DROPOUT_RATE, act_func=cfg.MODEL.HEAD_ACT, bn_lin5_on=cfg.X3D.BN_LIN5)

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float=0.0, training: bool=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert attention_type in ['divided_space_time', 'space_only', 'joint_space_time']
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W
        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            xt = x[:, 1:, :]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:, 1:, :] + res_temporal
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            cls_token = torch.mean(cls_token, 1, True)
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.0):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type) for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=T, mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x, B, T, W)
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
            x = torch.mean(x, 1)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


IMAGENET_DEFAULT_MEAN = 0.485, 0.456, 0.406


IMAGENET_DEFAULT_STD = 0.229, 0.224, 0.225


def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'first_conv': 'patch_embed.proj', 'classifier': 'head', **kwargs}


default_cfgs = {'vit_base_patch16_224': _cfg(url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))}


_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, img_size=224, num_frames=8, num_patches=196, attention_type='divided_space_time', pretrained_model='', strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning('Pretrained model URL is invalid, using random initialization.')
        return
    if len(pretrained_model) == 0:
        state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    else:
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= 3 / float(in_chans)
            conv1_weight = conv1_weight
            state_dict[conv1_name + '.weight'] = conv1_weight
    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=num_patches, mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=num_frames, mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)
    if attention_type == 'divided_space_time':
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if 'blocks' in key and 'attn' in key:
                new_key = key.replace('attn', 'temporal_attn')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if 'blocks' in key and 'norm1' in key:
                new_key = key.replace('norm1', 'temporal_norm1')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)


class vit_base_patch16_224(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained = True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)
        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = cfg.DATA.TRAIN_CROP_SIZE // patch_size * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x


class TimeSformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time', pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained = True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06), drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)
        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch' + str(patch_size) + '_224']
        self.num_patches = img_size // patch_size * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicTransform,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temp_kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (BottleneckTransform,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temp_kernel_size': 4, 'stride': 1, 'dim_inner': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv2dSame,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Nonlocal,
     lambda: ([], {'dim': 4, 'dim_inner': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (ResNetBasicStem,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'kernel': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (SE,
     lambda: ([], {'dim_in': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (X3DTransform,
     lambda: ([], {'dim_in': 4, 'dim_out': 4, 'temp_kernel_size': 4, 'stride': 1, 'dim_inner': 4, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
]

class Test_facebookresearch_TimeSformer(_paritybench_base):
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

