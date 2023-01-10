import sys
_module = sys.modules[__name__]
del sys
demo = _module
plot_roc_with_pickle = _module
predictor = _module
visualize_result = _module
conf = _module
fastreid = _module
config = _module
defaults = _module
data = _module
build = _module
common = _module
data_utils = _module
AirportALERT = _module
datasets = _module
bases = _module
caviara = _module
cuhk03 = _module
cuhk_sysu = _module
dukemtmcreid = _module
grid = _module
iLIDS = _module
lpw = _module
market1501 = _module
msmt17 = _module
pes3d = _module
pku = _module
prai = _module
prid = _module
saivt = _module
sensereid = _module
shinpuhkan = _module
sysu_mm = _module
thermalworld = _module
vehicleid = _module
veri = _module
veriwild = _module
viper = _module
wildtracker = _module
samplers = _module
data_sampler = _module
imbalance_sampler = _module
triplet_sampler = _module
transforms = _module
autoaugment = _module
functional = _module
transforms = _module
engine = _module
defaults = _module
hooks = _module
launch = _module
train_loop = _module
evaluation = _module
clas_evaluator = _module
evaluator = _module
query_expansion = _module
rank = _module
rank_cylib = _module
setup = _module
test_cython = _module
reid_evaluation = _module
rerank = _module
roc = _module
testing = _module
layers = _module
activation = _module
any_softmax = _module
batch_norm = _module
context_block = _module
drop = _module
frn = _module
gather_layer = _module
helpers = _module
non_local = _module
pooling = _module
se_layer = _module
splat = _module
weight_init = _module
modeling = _module
backbones = _module
mobilenet = _module
mobilenetv3 = _module
osnet = _module
regnet = _module
config = _module
effnet = _module
regnet = _module
repvgg = _module
resnest = _module
resnet = _module
resnext = _module
shufflenet = _module
vision_transformer = _module
heads = _module
clas_head = _module
embedding_head = _module
losses = _module
circle_loss = _module
cross_entroy_loss = _module
focal_loss = _module
triplet_loss = _module
utils = _module
meta_arch = _module
baseline = _module
build = _module
distiller = _module
mgn = _module
moco = _module
solver = _module
build = _module
lr_scheduler = _module
optim = _module
lamb = _module
radam = _module
swa = _module
checkpoint = _module
collect_env = _module
comm = _module
compute_dist = _module
env = _module
events = _module
faiss_utils = _module
file_io = _module
history_buffer = _module
logger = _module
params = _module
precision_bn = _module
registry = _module
summary = _module
timer = _module
visualizer = _module
fastattr = _module
attr_dataset = _module
attr_evaluation = _module
dukemtmcattr = _module
market1501attr = _module
pa100k = _module
attr_baseline = _module
attr_head = _module
bce_loss = _module
train_net = _module
fastclas = _module
bee_ant = _module
dataset = _module
trainer = _module
fastdistill = _module
overhaul = _module
resnet_distill = _module
fastface = _module
ms1mv2 = _module
test_dataset = _module
face_data = _module
face_evaluator = _module
face_baseline = _module
face_head = _module
iresnet = _module
partial_fc = _module
pfc_checkpointer = _module
trainer = _module
utils_amp = _module
verification = _module
market_benchmark = _module
test = _module
gen_wts = _module
fastretri = _module
retri_evaluator = _module
autotuner = _module
tune_hooks = _module
tune_net = _module
naic = _module
naic_dataset = _module
naic_evaluator = _module
partialreid = _module
dsr_distance = _module
dsr_evaluation = _module
dsr_head = _module
partial_dataset = _module
partialbaseline = _module
tests = _module
dataset_test = _module
feature_align = _module
interp_test = _module
lr_scheduler_test = _module
model_test = _module
sampler_test = _module
test_repvgg = _module
Caffe = _module
caffe_lmdb = _module
caffe_net = _module
caffe_pb2 = _module
layer_param = _module
net = _module
caffe_export = _module
caffe_inference = _module
onnx_export = _module
onnx_inference = _module
pytorch_to_caffe = _module
trt_calibrator = _module
trt_export = _module
trt_inference = _module
plain_train_net = _module

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


import torch.nn.functional as F


import numpy as np


from torch.backends import cudnn


from collections import deque


import torch


import torch.multiprocessing as mp


import logging


from typing import Dict


from typing import List


from typing import Tuple


from torch._six import string_classes


from collections import Mapping


from torch.utils.data import Dataset


import queue


from torch.utils.data import DataLoader


import itertools


from typing import Optional


from torch.utils.data import Sampler


from typing import Callable


from torch.utils.data.sampler import Sampler


import copy


from collections import defaultdict


import math


import random


from collections import OrderedDict


from torch.nn.parallel import DistributedDataParallel


import time


from collections import Counter


from torch import nn


import torch.distributed as dist


from torch.nn.parallel import DataParallel


from sklearn import metrics


import torch.nn as nn


from torch.nn.modules.batchnorm import BatchNorm2d


from torch.nn import ReLU


from torch.nn import LeakyReLU


from torch.nn.parameter import Parameter


from torch.nn import Conv2d


from torch.nn.modules.utils import _pair


import warnings


from torch import Tensor


from torch.nn.init import _calculate_fan_in_and_fan_out


from functools import partial


from typing import Any


from typing import Sequence


from torch.nn import functional as F


import re


from enum import Enum


from typing import Iterable


from typing import Set


from typing import Type


from typing import Union


from torch.optim.lr_scheduler import *


from torch.optim import *


import collections


from torch.optim.optimizer import Optimizer


from torch.utils.tensorboard import SummaryWriter


from typing import NamedTuple


import torchvision


import functools


from typing import IO


from typing import MutableMapping


from torch.autograd import Variable


import matplotlib.pyplot as plt


from scipy.stats import norm


import numbers


from torch.nn.utils import clip_grad_norm_


from torch.cuda.amp import GradScaler


import torchvision.transforms as T


from torch.onnx import OperatorExportTypes


from torch.nn.modules.utils import _list_with_default


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Linear(nn.Module):

    def __init__(self, num_classes, scale, margin):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits, targets):
        return logits.mul_(self.s)

    def extra_repr(self):
        return f'num_classes={self.num_classes}, scale={self.s}, margin={self.m}'


class CosSoftmax(Linear):
    """Implement of large margin cosine distance:
    """

    def forward(self, logits, targets):
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], self.m)
        logits[index] -= m_hot
        logits.mul_(self.s)
        return logits


class ArcSoftmax(Linear):

    def forward(self, logits, targets):
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], self.m)
        logits.acos_()
        logits[index] += m_hot
        logits.cos_().mul_(self.s)
        return logits


class CircleSoftmax(Linear):

    def forward(self, logits, targets):
        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.0)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m
        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], 1)
        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)
        logits[index] = logits_p[index] * m_hot + logits_n[index] * (1 - m_hot)
        neg_index = torch.where(targets == -1)[0]
        logits[neg_index] = logits_n[neg_index]
        logits.mul_(self.s)
        return logits


class BatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            nn.init.constant_(self.weight, weight_init)
        if bias_init is not None:
            nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class SyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            nn.init.constant_(self.weight, weight_init)
        if bias_init is not None:
            nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class FrozenBatchNorm(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """
    _version = 3

    def __init__(self, num_features, eps=1e-05, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = torch.zeros_like(self.running_mean)
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = torch.ones_like(self.running_var)
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info('FrozenBatchNorm {} is upgraded to version 3.'.format(prefix.rstrip('.')))
            state_dict[prefix + 'running_var'] -= self.eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = bn_module.BatchNorm2d, bn_module.SyncBatchNorm
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class GhostBatchNorm(BatchNorm):

    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var, self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits), True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm, 'syncBN': SyncBatchNorm, 'GhostBN': GhostBatchNorm, 'FrozenBN': FrozenBatchNorm, 'GN': lambda channels, **args: nn.GroupNorm(32, channels)}[norm]
    return norm(out_channels, **kwargs)


class IBN(nn.Module):

    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        if hasattr(m[-1], 'bias') and m[-1].bias is not None:
            nn.init.constant_(m[-1].bias, 0)
    else:
        nn.init.constant_(m.weight, val=0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([(f in valid_fusion_types) for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, a=0, mode='fan_in', nonlinearity='relu')
            if hasattr(self.conv_mask, 'bias') and self.conv_mask.bias is not None:
                nn.init.constant_(self.conv_mask.bias, 0)
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


def drop_block_2d(x, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    w_i, h_i = torch.meshgrid(torch.arange(W), torch.arange(H))
    valid_block = (w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2) & ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W))
    if batchwise:
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = 2 - gamma - valid_block + uniform_noise >= 1
    block_mask = -F.max_pool2d(-block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor, drop_prob: float=0.1, block_size: int=7, gamma_scale: float=1.0, with_noise: bool=False, inplace: bool=False, batchwise: bool=False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / ((W - block_size + 1) * (H - block_size + 1))
    if batchwise:
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(block_mask, kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)
    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-07)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self, drop_prob=0.1, block_size=7, gamma_scale=1.0, with_noise=False, inplace=False, batchwise=False, fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


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


class TLU(nn.Module):

    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau.view(1, self.num_features, 1, 1))


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-06, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()
        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        if is_eps_leanable:
            self.eps = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = self.weight.view(1, self.num_features, 1, 1) * x + self.bias.view(1, self.num_features, 1, 1)
        return x


class Non_local(nn.Module):

    def __init__(self, in_channels, bn_norm, reduc_ratio=2):
        super(Non_local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), get_norm(bn_norm, self.in_channels))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalAvgPool(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)


class GlobalMaxPool(nn.AdaptiveMaxPool2d):

    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)


class GeneralizedMeanPooling(nn.Module):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-06, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.p) + ', ' + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-06, *args, **kwargs):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class FastGlobalAvgPool(nn.Module):

    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class AdaptiveAvgMaxPool(nn.Module):

    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__()
        self.gap = FastGlobalAvgPool()
        self.gmp = GlobalMaxPool(output_size)

    def forward(self, x):
        avg_feat = self.gap(x)
        max_feat = self.gmp(x)
        feat = avg_feat + max_feat
        return feat


class ClipGlobalAvgPool(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.avgpool = FastGlobalAvgPool()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, int(channel / reduction), bias=False), nn.ReLU(inplace=True), nn.Linear(int(channel / reduction), channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DropBlock2D(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = get_norm(norm_layer, channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm_layer, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class ConvBNActivation(nn.Sequential):

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int=3, stride: int=1, groups: int=1, bn_norm=None, activation_layer: Optional[Callable[..., nn.Module]]=None, dilation: int=1) ->None:
        padding = (kernel_size - 1) // 2 * dilation
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False), get_norm(bn_norm, out_planes), activation_layer(inplace=True))
        self.out_channels = out_planes


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool, activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == 'HS'
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int=4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) ->Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) ->Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, bn_norm, se_layer: Callable[..., nn.Module]=SqueezeExcitation):
        super().__init__()
        if not 1 <= cnf.stride <= 2:
            raise ValueError('illegal stride value')
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1, bn_norm=bn_norm, activation_layer=activation_layer))
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel, stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels, bn_norm=bn_norm, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, bn_norm=bn_norm, activation_layer=nn.Identity))
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) ->Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


def conv_1x1_bn(inp, oup, bn_norm):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), get_norm(bn_norm, oup), nn.ReLU6(inplace=True))


def conv_3x3_bn(inp, oup, stride, bn_norm):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), get_norm(bn_norm, oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, bn_norm, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, bn_norm)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, bn_norm, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, bn_norm)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
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


class MobileNetV3(nn.Module):

    def __init__(self, bn_norm, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, block: Optional[Callable[..., nn.Module]]=None) ->None:
        """
        MobileNet V3 main class
        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
        """
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, Sequence) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual
        layers: List[nn.Module] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, bn_norm=bn_norm, activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, bn_norm))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, bn_norm=bn_norm, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(lastconv_output_channels, last_channel, bn_norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) ->Tensor:
        x = self.features(x)
        x = self.conv(x)
        return x

    def forward(self, x: Tensor) ->Tensor:
        return self._forward_impl(x)


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, bn_norm, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = get_norm(bn_norm, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels, bn_norm):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = nn.Identity()
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, bn_norm, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels, bn_norm)
        self.conv2a = LightConv3x3(mid_channels, mid_channels, bn_norm)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.conv2c = nn.Sequential(LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.conv2d = nn.Sequential(LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm), LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn_norm)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels, bn_norm)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return self.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, blocks, layers, channels, bn_norm, IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.conv1 = ConvLayer(3, channels[0], 7, bn_norm, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], bn_norm, reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], bn_norm, reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], bn_norm, reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3], bn_norm)
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, bn_norm, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, bn_norm, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, bn_norm, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels, bn_norm), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class EffHead(nn.Module):
    """EfficientNet head: 1x1, BN, Swish, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, bn_norm):
        super(EffHead, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.conv_bn = get_norm(bn_norm, w_out)
        self.conv_swish = Swish()

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.construct(w_in, w_se)

    def construct(self, w_in, w_se):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, kernel_size=1, bias=True), nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE), nn.Conv2d(w_se, w_in, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, bn_norm):
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1, padding=0, bias=False)
            self.exp_bn = get_norm(bn_norm, w_exp)
            self.exp_swish = Swish()
        dwise_args = {'groups': w_exp, 'padding': (kernel - 1) // 2, 'bias': False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel, stride=stride, **dwise_args)
        self.dwise_bn = get_norm(bn_norm, w_exp)
        self.dwise_swish = Swish()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = nn.Conv2d(w_exp, w_out, 1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = get_norm(bn_norm, w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and effnet_cfg.EN.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, effnet_cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d, bn_norm):
        super(EffStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = 'b{}'.format(i + 1)
            self.add_module(name, MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out, bn_norm))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet: 3x3, BN, Swish."""

    def __init__(self, w_in, w_out, bn_norm):
        super(StemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, 'final_bn') and m.final_bn and regnet_cfg.BN.ZERO_INIT_FINAL_GAMMA
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class EffNet(nn.Module):
    """EfficientNet model."""

    @staticmethod
    def get_args():
        return {'stem_w': effnet_cfg.EN.STEM_W, 'ds': effnet_cfg.EN.DEPTHS, 'ws': effnet_cfg.EN.WIDTHS, 'exp_rs': effnet_cfg.EN.EXP_RATIOS, 'se_r': effnet_cfg.EN.SE_R, 'ss': effnet_cfg.EN.STRIDES, 'ks': effnet_cfg.EN.KERNELS, 'head_w': effnet_cfg.EN.HEAD_W}

    def __init__(self, last_stride, bn_norm, **kwargs):
        super(EffNet, self).__init__()
        kwargs = self.get_args() if not kwargs else kwargs
        self._construct(**kwargs, last_stride=last_stride, bn_norm=bn_norm)
        self.apply(init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, last_stride, bn_norm):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, stem_w, bn_norm)
        prev_w = stem_w
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            name = 's{}'.format(i + 1)
            if i == 5:
                stride = last_stride
            self.add_module(name, EffStage(prev_w, exp_r, kernel, stride, se_r, w, d, bn_norm))
            prev_w = w
        self.head = EffHead(prev_w, head_w, bn_norm)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=None, gw=None, se_r=None):
        assert bm is None and gw is None and se_r is None, 'Vanilla block does not support bm, gw, and se_r options'
        super(VanillaBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def construct(self, w_in, w_out, stride, bn_norm):
        self.a = nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a_bn = get_norm(bn_norm, w_out)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(bn_norm, w_out)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bn_norm):
        super(BasicTransform, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def construct(self, w_in, w_out, stride, bn_norm):
        self.a = nn.Conv2d(w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.a_bn = get_norm(bn_norm, w_out)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(bn_norm, w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=None, gw=None, se_r=None):
        assert bm is None and gw is None and se_r is None, 'Basic transform does not support bm, gw, and se_r options'
        super(ResBasicBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def _add_skip_proj(self, w_in, w_out, stride, bn_norm):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = get_norm(bn_norm, w_out)

    def construct(self, w_in, w_out, stride, bn_norm):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_norm)
        self.f = BasicTransform(w_in, w_out, stride, bn_norm)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        w_b = int(round(w_out * bm))
        num_gs = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = get_norm(bn_norm, w_b)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        self.b = nn.Conv2d(w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False)
        self.b_bn = get_norm(bn_norm, w_b)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = get_norm(bn_norm, w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride, bn_norm):
        self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = get_norm(bn_norm, w_out)

    def construct(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_norm)
        self.f = BottleneckTransform(w_in, w_out, stride, bn_norm, bm, gw, se_r)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out, bn_norm):
        super(ResStemCifar, self).__init__()
        self.construct(w_in, w_out, bn_norm)

    def construct(self, w_in, w_out, bn_norm):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out, bn_norm):
        super(ResStemIN, self).__init__()
        self.construct(w_in, w_out, bn_norm)

    def construct(self, w_in, w_out, bn_norm):
        self.conv = nn.Conv2d(w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w, bn_norm):
        super(SimpleStemIN, self).__init__()
        self.construct(in_w, out_w, bn_norm)

    def construct(self, in_w, out_w, bn_norm):
        self.conv = nn.Conv2d(in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = get_norm(bn_norm, out_w)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            self.add_module('b{}'.format(i + 1), block_fun(b_w_in, w_out, b_stride, bn_norm, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {'vanilla_block': VanillaBlock, 'res_basic_block': ResBasicBlock, 'res_bottleneck_block': ResBottleneckBlock}
    assert block_type in block_funs.keys(), "Block type '{}' not supported".format(block_type)
    return block_funs[block_type]


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {'res_stem_cifar': ResStemCifar, 'res_stem_in': ResStemIN, 'simple_stem_in': SimpleStemIN}
    assert stem_type in stem_funs.keys(), "Stem type '{}' not supported".format(stem_type)
    return stem_funs[stem_type]


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self.construct(stem_type=kwargs['stem_type'], stem_w=kwargs['stem_w'], block_type=kwargs['block_type'], ds=kwargs['ds'], ws=kwargs['ws'], ss=kwargs['ss'], bn_norm=kwargs['bn_norm'], bms=kwargs['bms'], gws=kwargs['gws'], se_r=kwargs['se_r'])
        else:
            self.construct(stem_type=regnet_cfg.ANYNET.STEM_TYPE, stem_w=regnet_cfg.ANYNET.STEM_W, block_type=regnet_cfg.ANYNET.BLOCK_TYPE, ds=regnet_cfg.ANYNET.DEPTHS, ws=regnet_cfg.ANYNET.WIDTHS, ss=regnet_cfg.ANYNET.STRIDES, bn_norm=regnet_cfg.ANYNET.BN_NORM, bms=regnet_cfg.ANYNET.BOT_MULS, gws=regnet_cfg.ANYNET.GROUP_WS, se_r=regnet_cfg.ANYNET.SE_R if regnet_cfg.ANYNET.SE_ON else None)
        self.apply(init_weights)

    def construct(self, stem_type, stem_w, block_type, ds, ws, ss, bn_norm, bms, gws, se_r):
        bms = bms if bms else [(1.0) for _d in ds]
        gws = gws if gws else [(1) for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w, bn_norm)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module('s{}'.format(i + 1), AnyStage(prev_w, w, s, bn_norm, d, block_fun, bm, gw, se_r))
            prev_w = w
        self.in_planes = prev_w

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [(w != wp or r != rp) for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, last_stride, bn_norm):
        b_ws, num_s, _, _ = generate_regnet(regnet_cfg.REGNET.WA, regnet_cfg.REGNET.W0, regnet_cfg.REGNET.WM, regnet_cfg.REGNET.DEPTH)
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        gws = [regnet_cfg.REGNET.GROUP_W for _ in range(num_s)]
        bms = [regnet_cfg.REGNET.BOT_MUL for _ in range(num_s)]
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        ss = [regnet_cfg.REGNET.STRIDE for _ in range(num_s)]
        ss[-1] = last_stride
        se_r = regnet_cfg.REGNET.SE_R if regnet_cfg.REGNET.SE_ON else None
        kwargs = {'stem_type': regnet_cfg.REGNET.STEM_TYPE, 'stem_w': regnet_cfg.REGNET.STEM_W, 'block_type': regnet_cfg.REGNET.BLOCK_TYPE, 'ss': ss, 'ds': ds, 'ws': ws, 'bn_norm': bn_norm, 'bms': bms, 'gws': gws, 'se_r': se_r}
        super(RegNet, self).__init__(**kwargs)


def conv_bn(norm_type, in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', get_norm(norm_type, out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_type, kernel_size, stride=1, padding=0, groups=1):
        super(RepVGGBlock, self).__init__()
        self.deploying = False
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        self.in_channels = in_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.register_parameter('fused_weight', None)
        self.register_parameter('fused_bias', None)
        self.rbr_identity = get_norm(norm_type, in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(norm_type, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(norm_type, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if self.deploying:
            assert self.fused_weight is not None and self.fused_bias is not None, 'Make deploy mode=True to generate fused weight and fused bias first'
            fused_out = self.nonlinearity(torch.nn.functional.conv2d(inputs, self.fused_weight, self.fused_bias, self.stride, self.padding, 1, self.groups))
            return fused_out
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out = self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        return out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert branch.__class__.__name__.find('BatchNorm') != -1
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def deploy(self, mode=False):
        self.deploying = mode
        if mode:
            fused_weight, fused_bias = self.get_equivalent_kernel_bias()
            self.register_parameter('fused_weight', nn.Parameter(fused_weight))
            self.register_parameter('fused_bias', nn.Parameter(fused_bias))
            del self.rbr_identity, self.rbr_1x1, self.rbr_dense


class RepVGG(nn.Module):

    def __init__(self, last_stride, norm_type, num_blocks, width_multiplier=None, override_groups_map=None):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploying = False
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, norm_type=norm_type, kernel_size=3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), norm_type, num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), norm_type, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), norm_type, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), norm_type, num_blocks[3], stride=last_stride)

    def _make_stage(self, planes, norm_type, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, norm_type=norm_type, kernel_size=3, stride=stride, padding=1, groups=cur_groups))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def deploy(self, mode=False):
        self.deploying = mode
        for module in self.children():
            if hasattr(module, 'deploying'):
                module.deploy(mode)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.relu(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ResNeSt(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, last_stride, block, layers, radix=1, groups=1, bottleneck_width=64, dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down=False, rectified_conv=False, rectify_avg=False, avd=False, avd_first=False, final_drop=0.0, dropblock_prob=0, last_gamma=False, norm_layer='BN'):
        if last_stride == 1:
            dilation = 2
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super().__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs), get_norm(norm_layer, stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs), get_norm(norm_layer, stem_width), nn.ReLU(inplace=True), conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False, **conv_kwargs)
        self.bn1 = get_norm(norm_layer, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, dropblock_prob=dropblock_prob)
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(get_norm(norm_layer, planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=1, is_first=is_first, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=2, is_first=is_first, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix, cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, dilation=dilation, rectified_conv=self.rectified_conv, rectify_avg=self.rectify_avg, norm_layer=norm_layer, dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.relu(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


logger = logging.getLogger('fastreid')


class ResNet(nn.Module):

    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.channel_nums = []
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)
        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm='BN', with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), get_norm(bn_norm, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))
        self.channel_nums.append(self.inplanes)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(x, inplace=True)
        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            logger.info('ResNet unknown block error!')
        return [bn1, bn2, bn3, bn4]

    def extract_feature(self, x, preReLU=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)
        return [feat1, feat2, feat3, feat4], F.relu(feat4)

    def get_channel_nums(self):
        return self.channel_nums


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, last_stride, bn_norm, with_ibn, with_nl, block, layers, non_layers, baseWidth=4, cardinality=32):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64
        self.output_size = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn=with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn=with_ibn)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn=with_ibn)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_ibn=with_ibn)
        self.random_init()
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm='BN', with_ibn=False):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), get_norm(bn_norm, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, self.baseWidth, self.cardinality, 1, None))
        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList([Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([(layers[0] - (i + 1)) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList([Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([(layers[1] - (i + 1)) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList([Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([(layers[2] - (i + 1)) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList([Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([(layers[3] - (i + 1)) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        return x

    def random_init(self):
        self.conv1.weight.data.normal_(0, math.sqrt(2.0 / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ShuffleV2Block(nn.Module):
    """
    Reference:
        https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
    """

    def __init__(self, bn_norm, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        branch_main = [nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False), get_norm(bn_norm, mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False), get_norm(bn_norm, mid_channels), nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False), get_norm(bn_norm, outputs), nn.ReLU(inplace=True)]
        self.branch_main = nn.Sequential(*branch_main)
        if stride == 2:
            branch_proj = [nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False), get_norm(bn_norm, inp), nn.Conv2d(inp, inp, 1, 1, 0, bias=False), get_norm(bn_norm, inp), nn.ReLU(inplace=True)]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    """
    Reference:
        https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
    """

    def __init__(self, bn_norm, model_size='1.5x'):
        super(ShuffleNetV2, self).__init__()
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), get_norm(bn_norm, input_channel), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(bn_norm, input_channel, output_channel, mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(bn_norm, input_channel // 2, output_channel, mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self.conv_last = nn.Sequential(nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False), get_norm(bn_norm, self.stage_out_channels[-1]), nn.ReLU(inplace=True))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
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
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


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
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, camera=0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-06), sie_xishu=1.0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

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
        return {'pos_embed', 'cls_token'}

    def forward(self, x, camera_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0].reshape(x.shape[0], -1, 1, 1)


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) ->None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) ->None:
        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj: object=None) ->Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:

            def deco(func_or_class: object) ->object:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) ->object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret


REID_HEADS_REGISTRY = Registry('HEADS')


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    if len(args) and isinstance(args[0], _CfgNode):
        return True
    if isinstance(kwargs.pop('cfg', None), _CfgNode):
        return True
    return False


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != 'cfg':
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f'{from_config_func.__self__}.from_config'
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD] for param in signature.parameters.values())
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.
    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    def check_docstring(func):
        if func.__module__.startswith('fastreid.'):
            assert func.__doc__ is not None and 'experimental' in func.__doc__.lower(), f'configurable {func} should be marked experimental'
    if init_func is not None:
        assert inspect.isfunction(init_func) and from_config is None and init_func.__name__ == '__init__', 'Incorrect use of @configurable. Check API documentation for examples.'
        check_docstring(init_func)

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError("Class with @configurable must have a 'from_config' classmethod.") from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")
            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)
        return wrapped
    else:
        if from_config is None:
            return configurable
        assert inspect.isfunction(from_config), 'from_config argument of configurable must be a function!'

        def wrapper(orig_func):
            check_docstring(orig_func)

            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            return wrapped
        return wrapper


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class EmbeddingHead(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(self, *, feat_dim, embedding_dim, num_classes, neck_feat, pool_type, cls_type, scale, margin, with_bnneck, norm_type):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()
        assert hasattr(pooling, pool_type), 'Expected pool types are {}, but got {}'.format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()
        self.neck_feat = neck_feat
        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim
        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        self.bottleneck = nn.Sequential(*neck)
        assert hasattr(any_softmax, cls_type), 'Expected cls types are {}, but got {}'.format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        scale = cfg.MODEL.HEADS.SCALE
        margin = cfg.MODEL.HEADS.MARGIN
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        return {'feat_dim': feat_dim, 'embedding_dim': embedding_dim, 'num_classes': num_classes, 'neck_feat': neck_feat, 'pool_type': pool_type, 'cls_type': cls_type, 'scale': scale, 'margin': margin, 'with_bnneck': with_bnneck, 'norm_type': norm_type}

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]
        if not self.training:
            return neck_feat
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))
        cls_outputs = self.cls_layer(logits.clone(), targets)
        if self.neck_feat == 'before':
            feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after':
            feat = neck_feat
        else:
            raise KeyError(f'{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT')
        return {'cls_outputs': cls_outputs, 'pred_class_logits': logits.mul(self.cls_layer.s), 'features': feat}


META_ARCH_REGISTRY = Registry('META_ARCH')


BACKBONE_REGISTRY = Registry('BACKBONE')


def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone


def build_heads(cfg):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg)


def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    num_classes = pred_class_outputs.size(1)
    if eps >= 0:
        smooth_param = eps
    else:
        soft_label = F.softmax(pred_class_outputs, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)
    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), 1 - smooth_param)
    loss = (-targets * log_probs).sum(dim=1)
    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    loss = loss.sum() / non_zero_cnt
    return loss


_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(_CURRENT_STORAGE_STACK), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.size(0)
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))
    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1.0 / bsz))
    storage = get_event_storage()
    storage.put_scalar('cls_accuracy', ret[0])


def pairwise_circleloss(embedding: torch.Tensor, targets: torch.Tensor, margin: float, gamma: float) ->torch.Tensor:
    embedding = F.normalize(embedding, dim=1)
    dist_mat = torch.matmul(embedding, embedding.t())
    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg
    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.0)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.0)
    delta_p = 1 - margin
    delta_n = margin
    logit_p = -gamma * alpha_p * (s_p - delta_p) + -99999999.0 * (1 - is_pos)
    logit_n = gamma * alpha_n * (s_n - delta_n) + -99999999.0 * (1 - is_neg)
    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    return loss


def pairwise_cosface(embedding: torch.Tensor, targets: torch.Tensor, margin: float, gamma: float) ->torch.Tensor:
    embedding = F.normalize(embedding, dim=1)
    dist_mat = torch.matmul(embedding, embedding.t())
    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg
    logit_p = -gamma * s_p + -99999999.0 * (1 - is_pos)
    logit_n = gamma * (s_n + margin) + -99999999.0 * (1 - is_neg)
    loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
    return loss


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1000000000.0, dim=1)
    return dist_ap, dist_an


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-06
    W = torch.exp(diff) * mask / Z
    return W


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2
    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg
    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)
    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)
    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)
    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
    return loss


class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(self, *, backbone, heads, pixel_mean, pixel_std, loss_kwargs=None):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.loss_kwargs = loss_kwargs
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {'backbone': backbone, 'heads': heads, 'pixel_mean': cfg.MODEL.PIXEL_MEAN, 'pixel_std': cfg.MODEL.PIXEL_STD, 'loss_kwargs': {'loss_names': cfg.MODEL.LOSSES.NAME, 'ce': {'eps': cfg.MODEL.LOSSES.CE.EPSILON, 'alpha': cfg.MODEL.LOSSES.CE.ALPHA, 'scale': cfg.MODEL.LOSSES.CE.SCALE}, 'tri': {'margin': cfg.MODEL.LOSSES.TRI.MARGIN, 'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT, 'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING, 'scale': cfg.MODEL.LOSSES.TRI.SCALE}, 'circle': {'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN, 'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA, 'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE}, 'cosface': {'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN, 'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA, 'scale': cfg.MODEL.LOSSES.COSFACE.SCALE}}}

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        if self.training:
            assert 'targets' in batched_inputs, 'Person ID annotation are missing in training!'
            targets = batched_inputs['targets']
            if targets.sum() < 0:
                targets.zero_()
            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError('batched_inputs must be dict or torch.Tensor, but get {}'.format(type(batched_inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        log_accuracy(pred_class_logits, gt_labels)
        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']
        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(cls_outputs, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale')
        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(pred_features, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale')
        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(pred_features, gt_labels, circle_kwargs.get('margin'), circle_kwargs.get('gamma')) * circle_kwargs.get('scale')
        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(pred_features, gt_labels, cosface_kwargs.get('margin'), cosface_kwargs.get('gamma')) * cosface_kwargs.get('scale')
        return loss_dict


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """
    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]) ->None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.
        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError('Unused arguments: {}'.format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning('[PathManager] {}={} argument ignored'.format(k, v))

    def _get_supported_prefixes(self) ->List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str, **kwargs: Any) ->str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, this function is meant to be
        used with read-only resources.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _open(self, path: str, mode: str='r', buffering: int=-1, **kwargs: Any) ->Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.
        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.
        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def _copy(self, src_path: str, dst_path: str, overwrite: bool=False, **kwargs: Any) ->bool:
        """
        Copies a source path to a destination path.
        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) ->bool:
        """
        Checks if there is a resource at the given URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) ->bool:
        """
        Checks if the resource at the given URI is a file.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str, **kwargs: Any) ->bool:
        """
        Checks if the resource at the given URI is a directory.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def _ls(self, path: str, **kwargs: Any) ->List[str]:
        """
        List the contents of the directory at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str, **kwargs: Any) ->None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.
        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _rm(self, path: str, **kwargs: Any) ->None:
        """
        Remove the file (not directory) at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_local_path(self, path: str, **kwargs: Any) ->str:
        self._check_kwargs(kwargs)
        return path

    def _open(self, path: str, mode: str='r', buffering: int=-1, encoding: Optional[str]=None, errors: Optional[str]=None, newline: Optional[str]=None, closefd: bool=True, opener: Optional[Callable]=None, **kwargs: Any) ->Union[IO[str], IO[bytes]]:
        "\n        Open a path.\n        Args:\n            path (str): A URI supported by this PathHandler\n            mode (str): Specifies the mode in which the file is opened. It defaults\n                to 'r'.\n            buffering (int): An optional integer used to set the buffering policy.\n                Pass 0 to switch buffering off and an integer >= 1 to indicate the\n                size in bytes of a fixed-size chunk buffer. When no buffering\n                argument is given, the default buffering policy works as follows:\n                    * Binary files are buffered in fixed-size chunks; the size of\n                    the buffer is chosen using a heuristic trying to determine the\n                    underlying device’s “block size” and falling back on\n                    io.DEFAULT_BUFFER_SIZE. On many systems, the buffer will\n                    typically be 4096 or 8192 bytes long.\n            encoding (Optional[str]): the name of the encoding used to decode or\n                encode the file. This should only be used in text mode.\n            errors (Optional[str]): an optional string that specifies how encoding\n                and decoding errors are to be handled. This cannot be used in binary\n                mode.\n            newline (Optional[str]): controls how universal newlines mode works\n                (it only applies to text mode). It can be None, '', '\n', '\r',\n                and '\r\n'.\n            closefd (bool): If closefd is False and a file descriptor rather than\n                a filename was given, the underlying file descriptor will be kept\n                open when the file is closed. If a filename is given closefd must\n                be True (the default) otherwise an error will be raised.\n            opener (Optional[Callable]): A custom opener can be used by passing\n                a callable as opener. The underlying file descriptor for the file\n                object is then obtained by calling opener with (file, flags).\n                opener must return an open file descriptor (passing os.open as opener\n                results in functionality similar to passing None).\n            See https://docs.python.org/3/library/functions.html#open for details.\n        Returns:\n            file: a file-like object.\n        "
        self._check_kwargs(kwargs)
        return open(path, mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener)

    def _copy(self, src_path: str, dst_path: str, overwrite: bool=False, **kwargs: Any) ->bool:
        """
        Copies a source path to a destination path.
        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)
        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error('Destination file {} already exists.'.format(dst_path))
            return False
        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error('Error in file copy - {}'.format(str(e)))
            return False

    def _exists(self, path: str, **kwargs: Any) ->bool:
        self._check_kwargs(kwargs)
        return os.path.exists(path)

    def _isfile(self, path: str, **kwargs: Any) ->bool:
        self._check_kwargs(kwargs)
        return os.path.isfile(path)

    def _isdir(self, path: str, **kwargs: Any) ->bool:
        self._check_kwargs(kwargs)
        return os.path.isdir(path)

    def _ls(self, path: str, **kwargs: Any) ->List[str]:
        self._check_kwargs(kwargs)
        return os.listdir(path)

    def _mkdirs(self, path: str, **kwargs: Any) ->None:
        self._check_kwargs(kwargs)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str, **kwargs: Any) ->None:
        self._check_kwargs(kwargs)
        os.remove(path)


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """
    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativePathHandler()

    @staticmethod
    def __get_path_handler(path: str) ->PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.
        Args:
            path (str): URI path to resource
        Returns:
            handler (PathHandler)
        """
        for p in PathManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return PathManager._PATH_HANDLERS[p]
        return PathManager._NATIVE_PATH_HANDLER

    @staticmethod
    def open(path: str, mode: str='r', buffering: int=-1, **kwargs: Any) ->Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.
        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.
        Returns:
            file: a file-like object.
        """
        return PathManager.__get_path_handler(path)._open(path, mode, buffering=buffering, **kwargs)

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool=False, **kwargs: Any) ->bool:
        """
        Copies a source path to a destination path.
        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
        Returns:
            status (bool): True on success
        """
        assert PathManager.__get_path_handler(src_path) == PathManager.__get_path_handler(dst_path)
        return PathManager.__get_path_handler(src_path)._copy(src_path, dst_path, overwrite, **kwargs)

    @staticmethod
    def get_local_path(path: str, **kwargs: Any) ->str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        return PathManager.__get_path_handler(path)._get_local_path(path, **kwargs)

    @staticmethod
    def exists(path: str, **kwargs: Any) ->bool:
        """
        Checks if there is a resource at the given URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path exists
        """
        return PathManager.__get_path_handler(path)._exists(path, **kwargs)

    @staticmethod
    def isfile(path: str, **kwargs: Any) ->bool:
        """
        Checks if there the resource at the given URI is a file.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a file
        """
        return PathManager.__get_path_handler(path)._isfile(path, **kwargs)

    @staticmethod
    def isdir(path: str, **kwargs: Any) ->bool:
        """
        Checks if the resource at the given URI is a directory.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a directory
        """
        return PathManager.__get_path_handler(path)._isdir(path, **kwargs)

    @staticmethod
    def ls(path: str, **kwargs: Any) ->List[str]:
        """
        List the contents of the directory at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            List[str]: list of contents in given path
        """
        return PathManager.__get_path_handler(path)._ls(path, **kwargs)

    @staticmethod
    def mkdirs(path: str, **kwargs: Any) ->None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.
        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._mkdirs(path, **kwargs)

    @staticmethod
    def rm(path: str, **kwargs: Any) ->None:
        """
        Remove the file (not directory) at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._rm(path, **kwargs)

    @staticmethod
    def register_handler(handler: PathHandler) ->None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.
        Args:
            handler (PathHandler)
        """
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            assert prefix not in PathManager._PATH_HANDLERS
            PathManager._PATH_HANDLERS[prefix] = handler
        PathManager._PATH_HANDLERS = OrderedDict(sorted(PathManager._PATH_HANDLERS.items(), key=lambda t: t[0], reverse=True))

    @staticmethod
    def set_strict_kwargs_checking(enable: bool) ->None:
        """
        Toggles strict kwargs checking. If enabled, a ValueError is thrown if any
        unused parameters are passed to a PathHandler function. If disabled, only
        a warning is given.
        With a centralized file API, there's a tradeoff of convenience and
        correctness delegating arguments to the proper I/O layers. An underlying
        `PathHandler` may support custom arguments which should not be statically
        exposed on the `PathManager` function. For example, a custom `HTTPURLHandler`
        may want to expose a `cache_timeout` argument for `open()` which specifies
        how old a locally cached resource can be before it's refetched from the
        remote server. This argument would not make sense for a `NativePathHandler`.
        If strict kwargs checking is disabled, `cache_timeout` can be passed to
        `PathManager.open` which will forward the arguments to the underlying
        handler. By default, checking is enabled since it is innately unsafe:
        multiple `PathHandler`s could reuse arguments with different semantic
        meanings or types.
        Args:
            enable (bool)
        """
        PathManager._NATIVE_PATH_HANDLER._strict_kwargs_check = enable
        for handler in PathManager._PATH_HANDLERS.values():
            handler._strict_kwargs_check = enable


class _IncompatibleKeys(NamedTuple('IncompatibleKeys', [('missing_keys', List[str]), ('unexpected_keys', List[str]), ('incorrect_shapes', List[Tuple])])):
    pass


def _named_modules_with_dup(model: nn.Module, prefix: str='') ->Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        yield from _named_modules_with_dup(module, submodule_prefix)


def _filter_reused_missing_keys(model: nn.Module, keys: List[str]) ->List[str]:
    """
    Filter "missing keys" to not include keys that have been loaded with another name.
    """
    keyset = set(keys)
    param_to_names = defaultdict(set)
    for module_prefix, module in _named_modules_with_dup(model):
        for name, param in (list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))):
            full_name = (module_prefix + '.' if module_prefix else '') + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        if any(n in keyset for n in names) and not all(n in keyset for n in names):
            [keyset.remove(n) for n in names if n in keyset]
    return list(keyset)


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) ->None:
    """
    Strip the prefix in metadata, if any.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return
    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)
    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) ->Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.

    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind('.')
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) ->str:
    """
    Format a group of parameter name suffixes into a loggable string.

    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ''
    if len(group) == 1:
        return '.' + group[0]
    return '.{' + ', '.join(group) + '}'


def get_missing_parameters_message(keys: List[str]) ->str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.

    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = 'Some model parameters or buffers are not found in the checkpoint:\n'
    msg += '\n'.join('  ' + colored(k + _group_to_str(v), 'blue') for k, v in groups.items())
    return msg


def get_unexpected_parameters_message(keys: List[str]) ->str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.

    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = 'The checkpoint state_dict contains keys that are not used by the model:\n'
    msg += '\n'.join('  ' + colored(k + _group_to_str(v), 'magenta') for k, v in groups.items())
    return msg


class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """

    def __init__(self, model: nn.Module, save_dir: str='', *, save_to_disk: bool=True, **checkpointables: object):
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the `state_dict()` and `load_state_dict()` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.path_manager = PathManager

    def save(self, name: str, **kwargs: Dict[str, str]):
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return
        data = {}
        data['model'] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)
        basename = '{}.pth'.format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info('Saving checkpoint to {}'.format(save_file))
        with PathManager.open(save_file, 'wb') as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str, checkpointables: Optional[List[str]]=None) ->object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            self.logger.info('No checkpoint found. Training model from scratch')
            return {}
        self.logger.info('Loading checkpoint from {}'.format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), 'Checkpoint {} not found!'.format(path)
        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if incompatible is not None:
            self._log_incompatible_keys(incompatible)
        for key in (self.checkpointables if checkpointables is None else checkpointables):
            if key in checkpoint:
                self.logger.info('Loading {} from {}'.format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))
        return checkpoint

    def has_checkpoint(self):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        return PathManager.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        try:
            with PathManager.open(save_file, 'r') as f:
                last_saved = f.read().strip()
        except IOError:
            return ''
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        all_model_checkpoints = [os.path.join(self.save_dir, file) for file in PathManager.ls(self.save_dir) if PathManager.isfile(os.path.join(self.save_dir, file)) and file.endswith('.pth')]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool=True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.
        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basename: str):
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        with PathManager.open(save_file, 'w') as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str):
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device('cpu'))

    def _load_model(self, checkpoint: Any):
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.
        """
        checkpoint_state_dict = checkpoint.pop('model')
        self._convert_ndarray_to_tensor(checkpoint_state_dict)
        _strip_prefix_if_present(checkpoint_state_dict, 'module.')
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(missing_keys=incompatible.missing_keys, unexpected_keys=incompatible.unexpected_keys, incorrect_shapes=incorrect_shapes)

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) ->None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning("Skip loading parameter '{}' to the model due to incompatible shapes: {} in the checkpoint but {} in the model! You might want to double check if this is expected.".format(k, shape_checkpoint, shape_model))
        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(self.model, incompatible.missing_keys)
            if missing_keys:
                self.logger.info(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            self.logger.info(get_unexpected_parameters_message(incompatible.unexpected_keys))

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.

        Args:
            state_dict (dict): a state-dict to be loaded to the model.
        """
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError('Unsupported type found in checkpoint! {}: {}'.format(k, type(v)))
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model
    return model


BASE_KEY = '_BASE_'


GPU_ID = 0


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            return
        self.iter.exit_event.set()
        for _ in self.iter:
            pass
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k]

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def shutdown(self):
        self._shutdown_background_thread()


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, 'rb') as f:
        image = Image.open(f)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        if format is not None:
            conversion_format = format
            if format == 'BGR':
                conversion_format = 'RGB'
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == 'L':
            image = np.expand_dims(image, -1)
        elif format == 'BGR':
            image = image[:, :, ::-1]
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        image = Image.fromarray(image)
        return image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])
        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {'images': img, 'targets': pid, 'camids': camid, 'img_paths': img_path}

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


DATASET_REGISTRY = Registry('DATASET')


def autocontrast(pil_img, *args):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, *args):
    return ImageOps.equalize(pil_img)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def posterize(pil_img, level, *args):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, *args):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def solarize(pil_img, level, *args):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


augmentations = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y]


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        self.prob = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.augmentations = augmentations

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.

        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            return np.asarray(image).copy()
        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))
        mix = np.zeros([image.size[1], image.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)
        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)


_FILL = 128, 128, 128


_HPARAMS_DEFAULT = dict(translate_const=57, img_mean=_FILL)


_MAX_LEVEL = 10.0


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _enhance_increasing_level_to_arg(level, _hparams):
    level = level / _MAX_LEVEL * 0.9
    level = 1.0 + _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    return level / _MAX_LEVEL * 1.8 + 0.1,


def _posterize_level_to_arg(level, _hparams):
    return int(level / _MAX_LEVEL * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    return int(level / _MAX_LEVEL * 4) + 4,


def _rotate_level_to_arg(level, _hparams):
    level = level / _MAX_LEVEL * 30.0
    level = _randomly_negate(level)
    return level,


def _shear_level_to_arg(level, _hparams):
    level = level / _MAX_LEVEL * 0.3
    level = _randomly_negate(level)
    return level,


def _solarize_add_level_to_arg(level, _hparams):
    return int(level / _MAX_LEVEL * 110),


def _solarize_level_to_arg(level, _hparams):
    return int(level / _MAX_LEVEL * 256),


def _solarize_increasing_level_to_arg(level, _hparams):
    return 256 - _solarize_level_to_arg(level, _hparams)[0],


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = level / _MAX_LEVEL * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, hparams):
    translate_pct = hparams.get('translate_pct', 0.45)
    level = level / _MAX_LEVEL * translate_pct
    level = _randomly_negate(level)
    return level,


LEVEL_TO_ARG = {'AutoContrast': None, 'Equalize': None, 'Invert': None, 'Rotate': _rotate_level_to_arg, 'Posterize': _posterize_level_to_arg, 'PosterizeIncreasing': _posterize_increasing_level_to_arg, 'PosterizeOriginal': _posterize_original_level_to_arg, 'Solarize': _solarize_level_to_arg, 'SolarizeIncreasing': _solarize_increasing_level_to_arg, 'SolarizeAdd': _solarize_add_level_to_arg, 'Color': _enhance_level_to_arg, 'ColorIncreasing': _enhance_increasing_level_to_arg, 'Contrast': _enhance_level_to_arg, 'ContrastIncreasing': _enhance_increasing_level_to_arg, 'Brightness': _enhance_level_to_arg, 'BrightnessIncreasing': _enhance_increasing_level_to_arg, 'Sharpness': _enhance_level_to_arg, 'SharpnessIncreasing': _enhance_increasing_level_to_arg, 'ShearX': _shear_level_to_arg, 'ShearY': _shear_level_to_arg, 'TranslateX': _translate_abs_level_to_arg, 'TranslateY': _translate_abs_level_to_arg, 'TranslateXRel': _translate_rel_level_to_arg, 'TranslateYRel': _translate_rel_level_to_arg}


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def brightness(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def color(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def invert(img, **__):
    return ImageOps.invert(img)


def sharpness(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ('L', 'RGB'):
        if img.mode == 'RGB' and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


NAME_TO_OP = {'AutoContrast': auto_contrast, 'Equalize': equalize, 'Invert': invert, 'Rotate': rotate, 'Posterize': posterize, 'PosterizeIncreasing': posterize, 'PosterizeOriginal': posterize, 'Solarize': solarize, 'SolarizeIncreasing': solarize, 'SolarizeAdd': solarize_add, 'Color': color, 'ColorIncreasing': color, 'Contrast': contrast, 'ContrastIncreasing': contrast, 'Brightness': brightness, 'BrightnessIncreasing': brightness, 'Sharpness': sharpness, 'SharpnessIncreasing': sharpness, 'ShearX': shear_x, 'ShearY': shear_y, 'TranslateX': translate_x_abs, 'TranslateY': translate_y_abs, 'TranslateXRel': translate_x_rel, 'TranslateYRel': translate_y_rel}


class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL, resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION)
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


def auto_augment_policy_original(hparams):
    policy = [[('PosterizeOriginal', 0.4, 8), ('Rotate', 0.6, 9)], [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)], [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)], [('PosterizeOriginal', 0.6, 7), ('PosterizeOriginal', 0.6, 6)], [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)], [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)], [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)], [('PosterizeOriginal', 0.8, 5), ('Equalize', 1.0, 2)], [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)], [('Equalize', 0.6, 8), ('PosterizeOriginal', 0.4, 6)], [('Rotate', 0.8, 8), ('Color', 0.4, 0)], [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)], [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)], [('Invert', 0.6, 4), ('Equalize', 1.0, 8)], [('Color', 0.6, 4), ('Contrast', 1.0, 8)], [('Rotate', 0.8, 8), ('Color', 1.0, 2)], [('Color', 0.8, 8), ('Solarize', 0.8, 7)], [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)], [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)], [('Color', 0.4, 0), ('Equalize', 0.6, 3)], [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)], [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)], [('Invert', 0.6, 4), ('Equalize', 1.0, 8)], [('Color', 0.6, 4), ('Contrast', 1.0, 8)], [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)]]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_originalr(hparams):
    policy = [[('PosterizeIncreasing', 0.4, 8), ('Rotate', 0.6, 9)], [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)], [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)], [('PosterizeIncreasing', 0.6, 7), ('PosterizeIncreasing', 0.6, 6)], [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)], [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)], [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)], [('PosterizeIncreasing', 0.8, 5), ('Equalize', 1.0, 2)], [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)], [('Equalize', 0.6, 8), ('PosterizeIncreasing', 0.4, 6)], [('Rotate', 0.8, 8), ('Color', 0.4, 0)], [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)], [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)], [('Invert', 0.6, 4), ('Equalize', 1.0, 8)], [('Color', 0.6, 4), ('Contrast', 1.0, 8)], [('Rotate', 0.8, 8), ('Color', 1.0, 2)], [('Color', 0.8, 8), ('Solarize', 0.8, 7)], [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)], [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)], [('Color', 0.4, 0), ('Equalize', 0.6, 3)], [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)], [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)], [('Invert', 0.6, 4), ('Equalize', 1.0, 8)], [('Color', 0.6, 4), ('Contrast', 1.0, 8)], [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)]]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_v0(hparams):
    policy = [[('Equalize', 0.8, 1), ('ShearY', 0.8, 4)], [('Color', 0.4, 9), ('Equalize', 0.6, 3)], [('Color', 0.4, 1), ('Rotate', 0.6, 8)], [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)], [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)], [('Color', 0.2, 0), ('Equalize', 0.8, 8)], [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)], [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)], [('Color', 0.6, 1), ('Equalize', 1.0, 2)], [('Invert', 0.4, 9), ('Rotate', 0.6, 0)], [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)], [('Color', 0.4, 7), ('Equalize', 0.6, 0)], [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)], [('Solarize', 0.6, 8), ('Color', 0.6, 9)], [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)], [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)], [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)], [('ShearY', 0.8, 0), ('Color', 0.6, 4)], [('Color', 1.0, 0), ('Rotate', 0.6, 2)], [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)], [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)], [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)], [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)], [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)], [('Color', 0.8, 6), ('Rotate', 0.4, 5)]]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_v0r(hparams):
    policy = [[('Equalize', 0.8, 1), ('ShearY', 0.8, 4)], [('Color', 0.4, 9), ('Equalize', 0.6, 3)], [('Color', 0.4, 1), ('Rotate', 0.6, 8)], [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)], [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)], [('Color', 0.2, 0), ('Equalize', 0.8, 8)], [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)], [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)], [('Color', 0.6, 1), ('Equalize', 1.0, 2)], [('Invert', 0.4, 9), ('Rotate', 0.6, 0)], [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)], [('Color', 0.4, 7), ('Equalize', 0.6, 0)], [('PosterizeIncreasing', 0.4, 6), ('AutoContrast', 0.4, 7)], [('Solarize', 0.6, 8), ('Color', 0.6, 9)], [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)], [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)], [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)], [('ShearY', 0.8, 0), ('Color', 0.6, 4)], [('Color', 1.0, 0), ('Rotate', 0.6, 2)], [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)], [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)], [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)], [('PosterizeIncreasing', 0.8, 2), ('Solarize', 0.6, 10)], [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)], [('Color', 0.8, 6), ('Rotate', 0.4, 5)]]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name='original'):
    hparams = _HPARAMS_DEFAULT
    if name == 'original':
        return auto_augment_policy_original(hparams)
    elif name == 'originalr':
        return auto_augment_policy_originalr(hparams)
    elif name == 'v0':
        return auto_augment_policy_v0(hparams)
    elif name == 'v0r':
        return auto_augment_policy_v0r(hparams)
    else:
        assert False, 'Unknown AA policy (%s)' % name


class AutoAugment:

    def __init__(self):
        self.policy = auto_augment_policy()

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100, patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1, prob_flip_leftright=0.5):
        self.prob_happen = prob_happen
        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio
        self.prob_flip_leftright = prob_flip_leftright
        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1.0 / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)
        if len(self.patchpool) < self.min_sample_size:
            return img
        if random.uniform(0, 1) > self.prob_happen:
            return img
        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch
        return img


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, np.ndarray):
        assert len(pic.shape) in (2, 3)
        if pic.ndim == 2:
            pic = pic[:, :, None]
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_transforms(cfg, is_train=True):
    res = []
    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE
        crop_scale = cfg.INPUT.CROP.SCALE
        crop_ratio = cfg.INPUT.CROP.RATIO
        do_augmix = cfg.INPUT.AUGMIX.ENABLED
        augmix_prob = cfg.INPUT.AUGMIX.PROB
        do_autoaug = cfg.INPUT.AUTOAUG.ENABLED
        autoaug_prob = cfg.INPUT.AUTOAUG.PROB
        do_flip = cfg.INPUT.FLIP.ENABLED
        flip_prob = cfg.INPUT.FLIP.PROB
        do_pad = cfg.INPUT.PADDING.ENABLED
        padding_size = cfg.INPUT.PADDING.SIZE
        padding_mode = cfg.INPUT.PADDING.MODE
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE
        do_affine = cfg.INPUT.AFFINE.ENABLED
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_value = cfg.INPUT.REA.VALUE
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB
        if do_autoaug:
            res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))
        if size_train[0] > 0:
            res.append(T.Resize(size_train[0] if len(size_train) == 1 else size_train, interpolation=3))
        if do_crop:
            res.append(T.RandomResizedCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size, interpolation=3, scale=crop_scale, ratio=crop_ratio))
        if do_pad:
            res.extend([T.Pad(padding_size, padding_mode=padding_mode), T.RandomCrop(size_train[0] if len(size_train) == 1 else size_train)])
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_affine:
            res.append(T.RandomAffine(degrees=10, translate=None, scale=[0.9, 1.1], shear=0.1, resample=False, fillcolor=0))
        if do_augmix:
            res.append(AugMix(prob=augmix_prob))
        res.append(ToTensor())
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE
        if size_test[0] > 0:
            res.append(T.Resize(size_test[0] if len(size_test) == 1 else size_test, interpolation=3))
        if do_crop:
            res.append(T.CenterCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size))
        res.append(ToTensor())
    return T.Compose(res)


def _test_loader_from_config(cfg, *, dataset_name=None, test_set=None, num_query=0, transforms=None, **kwargs):
    if transforms is None:
        transforms = build_transforms(cfg, is_train=False)
    if test_set is None:
        assert dataset_name is not None, 'dataset_name must be explicitly passed in when test_set is not provided'
        data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
        if comm.is_main_process():
            data.show_test()
        test_items = data.query + data.gallery
        test_set = CommDataset(test_items, transforms, relabel=False)
        num_query = len(data.query)
    return {'test_set': test_set, 'test_batch_size': cfg.TEST.IMS_PER_BATCH, 'num_query': num_query}


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out
    elif isinstance(elem, Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}
    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs


@configurable(from_config=_test_loader_from_config)
def build_reid_test_loader(test_set, test_batch_size, num_query, num_workers=4):
    """
    Similar to `build_reid_train_loader`. This sampler coordinates all workers to produce
    the exact set of all samples
    This interface is experimental.

    Args:
        test_set:
        test_batch_size:
        num_query:
        num_workers:

    Returns:
        DataLoader: a torch DataLoader, that loads the given reid dataset, with
        the test-time transformation.

    Examples:
    ::
        data_loader = build_reid_test_loader(test_set, test_batch_size, num_query)
        # or, instantiate with a CfgNode:
        data_loader = build_reid_test_loader(cfg, "my_test")
    """
    mini_batch_size = test_batch_size // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoaderX(comm.get_local_rank(), dataset=test_set, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=fast_batch_collator, pin_memory=True)
    return test_loader, num_query


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        None
    indices = np.argsort(distmat, axis=1)
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [(x / (i + 1.0)) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    return all_cmc, all_AP, all_INP


def map(wrapper):
    model = wrapper
    cfg = get_cfg()
    test_loader, num_query = build_reid_test_loader(cfg, 'Market1501', T.Compose([]))
    feats = []
    pids = []
    camids = []
    for batch in test_loader:
        for image_path in batch['img_paths']:
            t = torch.Tensor(np.array([model.infer(cv2.imread(image_path))]))
            t
            feats.append(t)
        pids.extend(batch['targets'].numpy())
        camids.extend(batch['camids'].numpy())
    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()
    all_cmc, all_AP, all_INP = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, 5)
    mAP = np.mean(all_AP)
    None


class Distiller(Baseline):

    def __init__(self, cfg):
        super().__init__(cfg)
        model_ts = []
        for i in range(len(cfg.KD.MODEL_CONFIG)):
            cfg_t = get_cfg()
            cfg_t.merge_from_file(cfg.KD.MODEL_CONFIG[i])
            cfg_t.defrost()
            cfg_t.MODEL.META_ARCHITECTURE = 'Baseline'
            if cfg_t.MODEL.BACKBONE.NORM == 'syncBN':
                cfg_t.MODEL.BACKBONE.NORM = 'BN'
            if cfg_t.MODEL.HEADS.NORM == 'syncBN':
                cfg_t.MODEL.HEADS.NORM = 'BN'
            model_t = build_model(cfg_t)
            for param in model_t.parameters():
                param.requires_grad_(False)
            logger.info('Loading teacher model weights ...')
            Checkpointer(model_t).load(cfg.KD.MODEL_WEIGHTS[i])
            model_ts.append(model_t)
        self.ema_enabled = cfg.KD.EMA.ENABLED
        self.ema_momentum = cfg.KD.EMA.MOMENTUM
        if self.ema_enabled:
            cfg_self = cfg.clone()
            cfg_self.defrost()
            cfg_self.MODEL.META_ARCHITECTURE = 'Baseline'
            if cfg_self.MODEL.BACKBONE.NORM == 'syncBN':
                cfg_self.MODEL.BACKBONE.NORM = 'BN'
            if cfg_self.MODEL.HEADS.NORM == 'syncBN':
                cfg_self.MODEL.HEADS.NORM = 'BN'
            model_self = build_model(cfg_self)
            for param in model_self.parameters():
                param.requires_grad_(False)
            if cfg_self.MODEL.WEIGHTS != '':
                logger.info('Loading self distillation model weights ...')
                Checkpointer(model_self).load(cfg_self.MODEL.WEIGHTS)
            else:
                for param_q, param_k in zip(self.parameters(), model_self.parameters()):
                    param_k.data.copy_(param_q.data)
            model_ts.insert(0, model_self)
        self.model_ts = model_ts

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m=0.999):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.parameters(), self.model_ts[0].parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    def forward(self, batched_inputs):
        if self.training:
            images = self.preprocess_image(batched_inputs)
            s_feat = self.backbone(images)
            assert 'targets' in batched_inputs, 'Labels are missing in training!'
            targets = batched_inputs['targets']
            if targets.sum() < 0:
                targets.zero_()
            s_outputs = self.heads(s_feat, targets)
            t_outputs = []
            with torch.no_grad():
                if self.ema_enabled:
                    self._momentum_update_key_encoder(self.ema_momentum)
                for model_t in self.model_ts:
                    t_feat = model_t.backbone(images)
                    t_output = model_t.heads(t_feat, targets)
                    t_outputs.append(t_output)
            losses = self.losses(s_outputs, t_outputs, targets)
            return losses
        else:
            return super().forward(batched_inputs)

    def losses(self, s_outputs, t_outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(s_outputs, gt_labels)
        s_logits = s_outputs['pred_class_logits']
        loss_jsdiv = 0.0
        for t_output in t_outputs:
            t_logits = t_output['pred_class_logits'].detach()
            loss_jsdiv += self.jsdiv_loss(s_logits, t_logits)
        loss_dict['loss_jsdiv'] = loss_jsdiv / len(t_outputs)
        return loss_dict

    @staticmethod
    def _kldiv(y_s, y_t, t):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * t ** 2 / y_s.shape[0]
        return loss

    def jsdiv_loss(self, y_s, y_t, t=16):
        loss = (self._kldiv(y_s, y_t, t) + self._kldiv(y_t, y_s, t)) / 2
        return loss


class MGN(nn.Module):
    """
    Multiple Granularities Network architecture, which contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Multi-branch feature aggregation
    """

    @configurable
    def __init__(self, *, backbone, neck1, neck2, neck3, b1_head, b2_head, b21_head, b22_head, b3_head, b31_head, b32_head, b33_head, pixel_mean, pixel_std, loss_kwargs=None):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            neck1:
            neck2:
            neck3:
            b1_head:
            b2_head:
            b21_head:
            b22_head:
            b3_head:
            b31_head:
            b32_head:
            b33_head:
            pixel_mean:
            pixel_std:
            loss_kwargs:
        """
        super().__init__()
        self.backbone = backbone
        self.b1 = neck1
        self.b1_head = b1_head
        self.b2 = neck2
        self.b2_head = b2_head
        self.b21_head = b21_head
        self.b22_head = b22_head
        self.b3 = neck3
        self.b3_head = b3_head
        self.b31_head = b31_head
        self.b32_head = b32_head
        self.b33_head = b33_head
        self.loss_kwargs = loss_kwargs
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        bn_norm = cfg.MODEL.BACKBONE.NORM
        with_se = cfg.MODEL.BACKBONE.WITH_SE
        all_blocks = build_backbone(cfg)
        backbone = nn.Sequential(all_blocks.conv1, all_blocks.bn1, all_blocks.relu, all_blocks.maxpool, all_blocks.layer1, all_blocks.layer2, all_blocks.layer3[0])
        res_conv4 = nn.Sequential(*all_blocks.layer3[1:])
        res_g_conv5 = all_blocks.layer4
        res_p_conv5 = nn.Sequential(Bottleneck(1024, 512, bn_norm, False, with_se, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048))), Bottleneck(2048, 512, bn_norm, False, with_se), Bottleneck(2048, 512, bn_norm, False, with_se))
        res_p_conv5.load_state_dict(all_blocks.layer4.state_dict())
        neck1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        b1_head = build_heads(cfg)
        neck2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        b2_head = build_heads(cfg)
        b21_head = build_heads(cfg)
        b22_head = build_heads(cfg)
        neck3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        b3_head = build_heads(cfg)
        b31_head = build_heads(cfg)
        b32_head = build_heads(cfg)
        b33_head = build_heads(cfg)
        return {'backbone': backbone, 'neck1': neck1, 'neck2': neck2, 'neck3': neck3, 'b1_head': b1_head, 'b2_head': b2_head, 'b21_head': b21_head, 'b22_head': b22_head, 'b3_head': b3_head, 'b31_head': b31_head, 'b32_head': b32_head, 'b33_head': b33_head, 'pixel_mean': cfg.MODEL.PIXEL_MEAN, 'pixel_std': cfg.MODEL.PIXEL_STD, 'loss_kwargs': {'loss_names': cfg.MODEL.LOSSES.NAME, 'ce': {'eps': cfg.MODEL.LOSSES.CE.EPSILON, 'alpha': cfg.MODEL.LOSSES.CE.ALPHA, 'scale': cfg.MODEL.LOSSES.CE.SCALE}, 'tri': {'margin': cfg.MODEL.LOSSES.TRI.MARGIN, 'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT, 'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING, 'scale': cfg.MODEL.LOSSES.TRI.SCALE}, 'circle': {'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN, 'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA, 'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE}, 'cosface': {'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN, 'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA, 'scale': cfg.MODEL.LOSSES.COSFACE.SCALE}}}

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        b1_feat = self.b1(features)
        b2_feat = self.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
        b3_feat = self.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)
        if self.training:
            assert 'targets' in batched_inputs, 'Person ID annotation are missing in training!'
            targets = batched_inputs['targets']
            if targets.sum() < 0:
                targets.zero_()
            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b21_outputs = self.b21_head(b21_feat, targets)
            b22_outputs = self.b22_head(b22_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)
            b31_outputs = self.b31_head(b31_feat, targets)
            b32_outputs = self.b32_head(b32_feat, targets)
            b33_outputs = self.b33_head(b33_feat, targets)
            losses = self.losses(b1_outputs, b2_outputs, b21_outputs, b22_outputs, b3_outputs, b31_outputs, b32_outputs, b33_outputs, targets)
            return losses
        else:
            b1_pool_feat = self.b1_head(b1_feat)
            b2_pool_feat = self.b2_head(b2_feat)
            b21_pool_feat = self.b21_head(b21_feat)
            b22_pool_feat = self.b22_head(b22_feat)
            b3_pool_feat = self.b3_head(b3_feat)
            b31_pool_feat = self.b31_head(b31_feat)
            b32_pool_feat = self.b32_head(b32_feat)
            b33_pool_feat = self.b33_head(b33_feat)
            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat, b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
            return pred_feat

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError('batched_inputs must be dict or torch.Tensor, but get {}'.format(type(batched_inputs)))
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, b1_outputs, b2_outputs, b21_outputs, b22_outputs, b3_outputs, b31_outputs, b32_outputs, b33_outputs, gt_labels):
        pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits = b1_outputs['cls_outputs']
        b2_logits = b2_outputs['cls_outputs']
        b21_logits = b21_outputs['cls_outputs']
        b22_logits = b22_outputs['cls_outputs']
        b3_logits = b3_outputs['cls_outputs']
        b31_logits = b31_outputs['cls_outputs']
        b32_logits = b32_outputs['cls_outputs']
        b33_logits = b33_outputs['cls_outputs']
        b1_pool_feat = b1_outputs['features']
        b2_pool_feat = b2_outputs['features']
        b3_pool_feat = b3_outputs['features']
        b21_pool_feat = b21_outputs['features']
        b22_pool_feat = b22_outputs['features']
        b31_pool_feat = b31_outputs['features']
        b32_pool_feat = b32_outputs['features']
        b33_pool_feat = b33_outputs['features']
        log_accuracy(pred_class_logits, gt_labels)
        b22_pool_feat = torch.cat((b21_pool_feat, b22_pool_feat), dim=1)
        b33_pool_feat = torch.cat((b31_pool_feat, b32_pool_feat, b33_pool_feat), dim=1)
        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']
        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_b1'] = cross_entropy_loss(b1_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b2'] = cross_entropy_loss(b2_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b21'] = cross_entropy_loss(b21_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b22'] = cross_entropy_loss(b22_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b3'] = cross_entropy_loss(b3_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b31'] = cross_entropy_loss(b31_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b32'] = cross_entropy_loss(b32_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
            loss_dict['loss_cls_b33'] = cross_entropy_loss(b33_logits, gt_labels, ce_kwargs.get('eps'), ce_kwargs.get('alpha')) * ce_kwargs.get('scale') * 0.125
        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_b1'] = triplet_loss(b1_pool_feat, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale') * 0.2
            loss_dict['loss_triplet_b2'] = triplet_loss(b2_pool_feat, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale') * 0.2
            loss_dict['loss_triplet_b3'] = triplet_loss(b3_pool_feat, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale') * 0.2
            loss_dict['loss_triplet_b22'] = triplet_loss(b22_pool_feat, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale') * 0.2
            loss_dict['loss_triplet_b33'] = triplet_loss(b33_pool_feat, gt_labels, tri_kwargs.get('margin'), tri_kwargs.get('norm_feat'), tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale') * 0.2
        return loss_dict


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class Memory(nn.Module):
    """
    Build a MoCo memory with a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=512, K=65536):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        """
        super().__init__()
        self.K = K
        self.margin = 0.25
        self.gamma = 32
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_label', torch.zeros((1, K), dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):
        if comm.get_world_size() > 1:
            keys = concat_all_gather(keys)
            targets = concat_all_gather(targets)
        else:
            keys = keys.detach()
            targets = targets.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[:, ptr:ptr + batch_size] = targets
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, feat_q, targets):
        """
        Memory bank enqueue and compute metric loss
        Args:
            feat_q: model features
            targets: gt labels

        Returns:
        """
        feat_q = F.normalize(feat_q, p=2, dim=1)
        self._dequeue_and_enqueue(feat_q.detach(), targets)
        loss = self._pairwise_cosface(feat_q, targets)
        return loss

    def _pairwise_cosface(self, feat_q, targets):
        dist_mat = torch.matmul(feat_q, self.queue)
        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()
        is_neg = targets.view(N, 1).expand(N, M).ne(self.queue_label.expand(N, M)).float()
        same_indx = torch.eye(N, N, device=is_pos.device)
        other_indx = torch.zeros(N, M - N, device=is_pos.device)
        same_indx = torch.cat((same_indx, other_indx), dim=1)
        is_pos = is_pos - same_indx
        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg
        logit_p = -self.gamma * s_p + -99999999.0 * (1 - is_pos)
        logit_n = self.gamma * (s_n + self.margin) + -99999999.0 * (1 - is_neg)
        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss


class MoCo(Baseline):

    def __init__(self, cfg):
        super().__init__(cfg)
        dim = cfg.MODEL.HEADS.EMBEDDING_DIM if cfg.MODEL.HEADS.EMBEDDING_DIM else cfg.MODEL.BACKBONE.FEAT_DIM
        size = cfg.MODEL.QUEUE_SIZE
        self.memory = Memory(dim, size)

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(outputs, gt_labels)
        pred_features = outputs['features']
        loss_mb = self.memory(pred_features, gt_labels)
        loss_dict['loss_mb'] = loss_mb
        return loss_dict


class AttrHead(EmbeddingHead):

    def __init__(self, cfg):
        super().__init__(cfg)
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.bnneck = nn.BatchNorm1d(num_classes)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)
        logits = F.linear(neck_feat, self.weight)
        logits = self.bnneck(logits)
        if not self.training:
            cls_outptus = torch.sigmoid(logits)
            return cls_outptus
        return {'cls_outputs': logits}


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(t_channel)]
    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return nn.Sequential(*C)


def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = F.mse_loss(source, target, reduction='none')
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for s, m in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(-s * math.exp(-(m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)
    return torch.tensor(margin, dtype=torch.float32, device=mean.device)


class DistillerOverhaul(Distiller):

    def __init__(self, cfg):
        super().__init__(cfg)
        s_channels = self.backbone.get_channel_nums()
        for i in range(len(self.model_ts)):
            t_channels = self.model_ts[i].backbone.get_channel_nums()
            setattr(self, 'connectors_{}'.format(i), nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)]))
            teacher_bns = self.model_ts[i].backbone.get_bn_before_relu()
            margins = [get_margin_from_BN(bn) for bn in teacher_bns]
            for j, margin in enumerate(margins):
                self.register_buffer('margin{}_{}'.format(i, j + 1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

    def forward(self, batched_inputs):
        if self.training:
            images = self.preprocess_image(batched_inputs)
            s_feats, s_feat = self.backbone.extract_feature(images, preReLU=True)
            assert 'targets' in batched_inputs, 'Labels are missing in training!'
            targets = batched_inputs['targets']
            if targets.sum() < 0:
                targets.zero_()
            s_outputs = self.heads(s_feat, targets)
            t_feats_list = []
            t_outputs = []
            with torch.no_grad():
                if self.ema_enabled:
                    self._momentum_update_key_encoder(self.ema_momentum)
                for model_t in self.model_ts:
                    t_feats, t_feat = model_t.backbone.extract_feature(images, preReLU=True)
                    t_output = model_t.heads(t_feat, targets)
                    t_feats_list.append(t_feats)
                    t_outputs.append(t_output)
            losses = self.losses(s_outputs, s_feats, t_outputs, t_feats_list, targets)
            return losses
        else:
            outputs = super(DistillerOverhaul, self).forward(batched_inputs)
            return outputs

    def losses(self, s_outputs, s_feats, t_outputs, t_feats_list, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(s_outputs, t_outputs, gt_labels)
        feat_num = len(s_feats)
        loss_distill = 0
        for i in range(len(t_feats_list)):
            for j in range(feat_num):
                s_feats_connect = getattr(self, 'connectors_{}'.format(i))[j](s_feats[j])
                loss_distill += distillation_loss(s_feats_connect, t_feats_list[i][j].detach(), getattr(self, 'margin{}_{}'.format(i, j + 1))) / 2 ** (feat_num - j - 1)
        loss_dict['loss_overhaul'] = loss_distill / len(t_feats_list) / len(gt_labels) / 10000
        return loss_dict


class FaceBaseline(Baseline):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pfc_enabled = cfg.MODEL.HEADS.PFC.ENABLED
        self.amp_enabled = cfg.SOLVER.AMP.ENABLED

    def forward(self, batched_inputs):
        if not self.pfc_enabled:
            return super().forward(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        with torch.amp.autocast(self.amp_enabled):
            features = self.backbone(images)
        features = features.float() if self.amp_enabled else features
        if self.training:
            assert 'targets' in batched_inputs, 'Person ID annotation are missing in training!'
            targets = batched_inputs['targets']
            if targets.sum() < 0:
                targets.zero_()
            outputs = self.heads(features, targets)
            return outputs, targets
        else:
            outputs = self.heads(features)
            return outputs


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.bn1 = get_norm(bn_norm, inplanes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = get_norm(bn_norm, planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = get_norm(bn_norm, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, bn_norm, dropout=0, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.fp16 = fp16
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = get_norm(bn_norm, self.inplanes)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_norm, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_norm, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], bn_norm, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], bn_norm, stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = get_norm(bn_norm, 512 * block.expansion)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif m.__class__.__name__.find('Norm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, bn_norm, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), get_norm(bn_norm, planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        return x


class PartialFC(nn.Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    def __init__(self, embedding_size, num_classes, sample_rate, cls_type, scale, margin):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
        self.local_rank = comm.get_local_rank()
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.num_local: int = self.num_classes // self.world_size + int(self.rank < self.num_classes % self.world_size)
        self.class_start: int = self.num_classes // self.world_size * self.rank + min(self.rank, self.num_classes % self.world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)
        self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
        self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
        logger.info('softmax weight init successfully!')
        logger.info('softmax weight mom init successfully!')
        self.stream: torch.Stream = torch.Stream(self.local_rank)
        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda : 0
            self.sub_weight = nn.Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = nn.Parameter(torch.empty((0, 0), device=self.device))

    def forward(self, total_features):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(total_features, self.sub_weight)
        else:
            logits = F.linear(F.normalize(total_features), F.normalize(self.sub_weight))
        return logits

    def forward_backward(self, features, targets, optimizer):
        """
        Partial FC forward, which will sample positive weights and part of negative weights,
        then compute logits and get the grad of features.
        """
        total_targets = self.prepare(targets, optimizer)
        if self.world_size > 1:
            total_features = concat_all_gather(features)
        else:
            total_features = features.detach()
        total_features.requires_grad_(True)
        logits = self.forward(total_features)
        logits = self.cls_layer(logits, total_targets)
        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            if self.world_size > 1:
                dist.all_reduce(max_fc, dist.ReduceOp.MAX)
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdim=True)
            if self.world_size > 1:
                dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)
            logits_exp.div_(logits_sum_exp)
            grad = logits_exp
            index = torch.where(total_targets != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_targets[index, None], 1)
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_targets[index, None])
            if self.world_size > 1:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * -1
            grad[index] -= one_hot
            grad.div_(logits.size(0))
        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features)
        if self.world_size > 1:
            dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        else:
            x_grad = total_features.grad
        x_grad = x_grad * self.world_size
        return x_grad, loss_v

    @torch.no_grad()
    def sample(self, total_targets):
        """
        Get sub_weights according to total targets gathered from all GPUs, due to each weights in different
        GPU contains different class centers.
        """
        index_positive = (self.class_start <= total_targets) & (total_targets < self.class_start + self.num_local)
        total_targets[~index_positive] = -1
        total_targets[index_positive] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_targets[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.weight.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_targets[index_positive] = torch.searchsorted(index, total_targets[index_positive])
            self.sub_weight = nn.Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, targets, optimizer):
        with torch.cuda.stream(self.stream):
            if self.world_size > 1:
                total_targets = concat_all_gather(targets)
            else:
                total_targets = targets
            self.sample(total_targets)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            return total_targets


class OcclusionUnit(nn.Module):

    def __init__(self, in_planes=2048):
        super(OcclusionUnit, self).__init__()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.mask_layer = nn.Linear(in_planes, 1, bias=True)

    def forward(self, x):
        SpaFeat1 = self.MaxPool1(x)
        SpaFeat2 = self.MaxPool2(x)
        SpaFeat3 = self.MaxPool3(x)
        SpaFeat4 = self.MaxPool4(x)
        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), 2)
        SpatialFeatAll = SpatialFeatAll.transpose(1, 2)
        y = self.mask_layer(SpatialFeatAll)
        mask_weight = torch.sigmoid(y[:, :, 0])
        feat_dim = SpaFeat1.size(2) * SpaFeat1.size(3)
        mask_score = F.normalize(mask_weight[:, :feat_dim], p=1, dim=1)
        mask_weight_norm = F.normalize(mask_weight, p=1, dim=1)
        mask_score = mask_score.unsqueeze(1)
        SpaFeat1 = SpaFeat1.transpose(1, 2)
        SpaFeat1 = SpaFeat1.transpose(2, 3)
        SpaFeat1 = SpaFeat1.view((SpaFeat1.size(0), SpaFeat1.size(1) * SpaFeat1.size(2), -1))
        global_feats = mask_score.matmul(SpaFeat1).view(SpaFeat1.shape[0], -1, 1, 1)
        return global_feats, mask_weight, mask_weight_norm


class DSRHead(EmbeddingHead):

    def __init__(self, cfg):
        super().__init__(cfg)
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        self.occ_unit = OcclusionUnit(in_planes=feat_dim)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        occ_neck = []
        if embedding_dim > 0:
            occ_neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim
        if with_bnneck:
            occ_neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        self.bnneck_occ = nn.Sequential(*occ_neck)
        self.bnneck_occ.apply(weights_init_kaiming)
        self.weight_occ = nn.Parameter(torch.normal(0, 0.01, (num_classes, feat_dim)))

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        SpaFeat1 = self.MaxPool1(features)
        SpaFeat2 = self.MaxPool2(features)
        SpaFeat3 = self.MaxPool3(features)
        SpaFeat4 = self.MaxPool4(features)
        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), dim=2)
        foreground_feat, mask_weight, mask_weight_norm = self.occ_unit(features)
        bn_foreground_feat = self.bnneck_occ(foreground_feat)
        bn_foreground_feat = bn_foreground_feat[..., 0, 0]
        if not self.training:
            return bn_foreground_feat, SpatialFeatAll, mask_weight_norm
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        if self.cls_layer.__class__.__name__ == 'Linear':
            pred_class_logits = F.linear(bn_feat, self.weight)
            fore_pred_class_logits = F.linear(bn_foreground_feat, self.weight_occ)
        else:
            pred_class_logits = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
            fore_pred_class_logits = F.linear(F.normalize(bn_foreground_feat), F.normalize(self.weight_occ))
        cls_outputs = self.cls_layer(pred_class_logits, targets)
        fore_cls_outputs = self.cls_layer(fore_pred_class_logits, targets)
        return {'cls_outputs': cls_outputs, 'fore_cls_outputs': fore_cls_outputs, 'pred_class_logits': pred_class_logits * self.cls_layer.s, 'features': global_feat[..., 0, 0], 'foreground_features': foreground_feat[..., 0, 0]}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveAvgMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AnyHead,
     lambda: ([], {'w_in': 4, 'nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AnyStage,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'bn_norm': _mock_layer, 'd': 4, 'block_fun': _mock_layer, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ArcSoftmax,
     lambda: ([], {'num_classes': 4, 'scale': 1.0, 'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CircleSoftmax,
     lambda: ([], {'num_classes': 4, 'scale': 1.0, 'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (ClipGlobalAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextBlock,
     lambda: ([], {'inplanes': 4, 'ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1x1Linear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CosSoftmax,
     lambda: ([], {'num_classes': 4, 'scale': 1.0, 'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (DropBlock2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EffHead,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FRN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastGlobalAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FrozenBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedMeanPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeneralizedMeanPoolingP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBN,
     lambda: ([], {'planes': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LightConv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'num_classes': 4, 'scale': 1.0, 'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNetV2,
     lambda: ([], {'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SplAtConv2d,
     lambda: ([], {'in_channels': 4, 'channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SqueezeExcitation,
     lambda: ([], {'input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StemIN,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'bn_norm': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SyncBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TLU,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (rSoftMax,
     lambda: ([], {'radix': 4, 'cardinality': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_JDAI_CV_fast_reid(_paritybench_base):
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

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

