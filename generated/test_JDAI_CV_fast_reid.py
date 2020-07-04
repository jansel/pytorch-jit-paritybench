import sys
_module = sys.modules[__name__]
del sys
demo = _module
plot_roc_with_pickle = _module
predictor = _module
visualize_result = _module
fastreid = _module
config = _module
defaults = _module
data = _module
build = _module
common = _module
data_utils = _module
datasets = _module
bases = _module
cuhk03 = _module
dukemtmcreid = _module
market1501 = _module
msmt17 = _module
vehicleid = _module
veri = _module
veriwild = _module
samplers = _module
data_sampler = _module
triplet_sampler = _module
transforms = _module
autoaugment = _module
functional = _module
engine = _module
defaults = _module
hooks = _module
train_loop = _module
evaluation = _module
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
export = _module
tensorflow_export = _module
tf_modeling = _module
layers = _module
activation = _module
arcface = _module
batch_drop = _module
batch_norm = _module
circle = _module
context_block = _module
frn = _module
non_local = _module
pooling = _module
se_layer = _module
splat = _module
sync_bn = _module
batchnorm = _module
batchnorm_reimpl = _module
comm = _module
replicate = _module
unittest = _module
modeling = _module
backbones = _module
osnet = _module
resnest = _module
resnet = _module
resnext = _module
heads = _module
bnneck_head = _module
linear_head = _module
reduction_head = _module
losses = _module
build_losses = _module
center_loss = _module
cross_entroy_loss = _module
focal_loss = _module
metric_loss = _module
meta_arch = _module
baseline = _module
build = _module
mgn = _module
solver = _module
lr_scheduler = _module
optim = _module
lamb = _module
lookahead = _module
novograd = _module
over9000 = _module
radam = _module
ralamb = _module
ranger = _module
swa = _module
utils = _module
checkpoint = _module
events = _module
file_io = _module
history_buffer = _module
logger = _module
one_hot = _module
precision_bn = _module
registry = _module
summary = _module
timer = _module
visualizer = _module
weight_init = _module
partialreid = _module
dsr_distance = _module
dsr_evaluation = _module
dsr_head = _module
partial_dataset = _module
partialbaseline = _module
train_net = _module
tests = _module
dataset_test = _module
interp_test = _module
lr_scheduler_test = _module
model_test = _module
sampler_test = _module
Caffe = _module
caffe_lmdb = _module
caffe_net = _module
caffe_pb2 = _module
layer_param = _module
net = _module
caffe_export = _module
caffe_inference = _module
export2tf = _module
pytorch_to_caffe = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import logging


from collections import OrderedDict


import torch


import torch.nn.functional as F


from torch.nn import DataParallel


import itertools


import warnings


import time


from collections import Counter


from torch import nn


import numpy as np


import copy


import torchvision.transforms as transforms


from torch.backends import cudnn


import math


import torch.nn as nn


from torch.nn import Parameter


import random


from torch.nn.modules.batchnorm import BatchNorm2d


from torch.nn import ReLU


from torch.nn import LeakyReLU


from torch.nn.parameter import Parameter


from torch.nn import Conv2d


from torch.nn.modules.utils import _pair


import collections


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as init


import functools


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn import functional as F


from torch.utils import model_zoo


from torch.nn import init


from collections import defaultdict


from typing import Any


from torch.utils.data import Dataset


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from torch.autograd import Variable


from torch.nn.modules.utils import _list_with_default


def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
    return backbone


def build_reid_heads(cfg, in_feat, num_classes, pool_layer):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg, in_feat, num_classes, pool_layer)


class TfMetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)

    def forward(self, x):
        global_feat = self.backbone(x)
        pred_features = self.heads(global_feat)
        return pred_features


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):

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
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class Arcface(nn.Module):

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN
        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))

    def forward(self, features, targets):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-07, 1.0 - 1e-07))
        phi = torch.cos(theta + self._m)
        targets = one_hot(targets, self._num_classes)
        pred_class_logits = targets * phi + (1.0 - targets) * cosine
        pred_class_logits *= self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m)


class BatchDrop(nn.Module):
    """ref: https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
    batch drop mask
    """

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class BatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze
        =False, bias_freeze=False, weight_init=1.0, bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            self.weight.data.fill_(weight_init)
        if bias_init is not None:
            self.bias.data.fill_(bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class FrozenBatchNorm(BatchNorm):
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

    def __init__(self, num_features, eps=1e-05):
        super().__init__(num_features, weight_freeze=True, bias_freeze=True)
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            return F.batch_norm(x, self.running_mean, self.running_var,
                self.weight, self.bias, training=False, eps=self.eps)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = torch.zeros_like(self
                    .running_mean)
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = torch.ones_like(self.
                    running_var)
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info('FrozenBatchNorm {} is upgraded to version 3.'.
                format(prefix.rstrip('.')))
            state_dict[prefix + 'running_var'] -= self.eps
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(self.
            num_features, self.eps)

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
            outputs = F.batch_norm(input.view(-1, C * self.num_splits, H, W
                ), self.running_mean, self.running_var, self.weight.repeat(
                self.num_splits), self.bias.repeat(self.num_splits), True,
                self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.
                num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.
                num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


def get_norm(norm, out_channels, num_splits=1, **kwargs):
    """
    Args:
        norm (str or callable):
    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm(out_channels, **kwargs), 'GhostBN':
            GhostBatchNorm(out_channels, num_splits, **kwargs), 'FrozenBN':
            FrozenBatchNorm(out_channels), 'GN': nn.GroupNorm(32,
            out_channels), 'syncBN': SyncBatchNorm(out_channels, **kwargs)}[
            norm]
    return norm


class IBN(nn.Module):

    def __init__(self, planes, bn_norm, num_splits):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, num_splits)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Circle(nn.Module):

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN
        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m
        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)
        targets = one_hot(targets, self._num_classes)
        pred_class_logits = targets * s_p + (1.0 - targets) * s_n
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m)


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

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=(
        'channel_add',)):
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
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes,
                self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 
                1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.
                inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            nn.init.kaiming_normal_(self.conv_mask.weight, a=0, mode=
                'fan_in', nonlinearity='relu')
            if hasattr(self.conv_mask, 'bias'
                ) and self.conv_mask.bias is not None:
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
        return 'num_features={num_features}, eps={init_eps}'.format(**self.
            __dict__)

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
        x = self.weight.view(1, self.num_features, 1, 1) * x + self.bias.view(
            1, self.num_features, 1, 1)
        return x


class Non_local(nn.Module):

    def __init__(self, in_channels, bn_norm, num_splits, reduc_ratio=2):
        super(Non_local, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.
            inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels,
            out_channels=self.in_channels, kernel_size=1, stride=1, padding
            =0), get_norm(bn_norm, self.in_channels, num_splits))
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=
            self.inter_channels, kernel_size=1, stride=1, padding=0)

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


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


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

    def __init__(self, norm, output_size=1, eps=1e-06):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size
            ).pow(1.0 / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.p
            ) + ', ' + 'output_size=' + str(self.output_size) + ')'


class AdaptiveAvgMaxPool2d(nn.Module):

    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


class FastGlobalAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0),
                x.size(1), 1, 1)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, int(channel / reduction),
            bias=False), nn.ReLU(inplace=True), nn.Linear(int(channel /
            reduction), channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
        padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2,
        reduction_factor=4, rectify=False, rectify_avg=False, norm_layer=
        None, num_splits=1, dropblock_prob=0.0, **kwargs):
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
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size,
                stride, padding, dilation, groups=groups * radix, bias=bias,
                **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = get_norm(norm_layer, channels * radix, num_splits)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm_layer, inter_channels, num_splits)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.
            cardinality)
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
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([(att * split) for att, split in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


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


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


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

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
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
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
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


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze
        =False, bias_freeze=False, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(
                input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(
                input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std *
                self.weight) + _unsqueeze_ft(self.bias)
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
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum
                    ) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum
                    ) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum
                ) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum
                ) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1
            ) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0,
            2, 3).contiguous()


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
        modules = super(DataParallelWithCallback, self).replicate(module,
            device_ids)
        execute_replication_callbacks(modules)
        return modules


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
            padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
            padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
            padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
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

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1,
            padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
            padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
        gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction,
            kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates,
            kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = nn.Identity()
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(
                gate_activation))

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

    def __init__(self, in_channels, out_channels, IN=False,
        bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels
            ), LightConv3x3(mid_channels, mid_channels))
        self.conv2c = nn.Sequential(LightConv3x3(mid_channels, mid_channels
            ), LightConv3x3(mid_channels, mid_channels), LightConv3x3(
            mid_channels, mid_channels))
        self.conv2d = nn.Sequential(LightConv3x3(mid_channels, mid_channels
            ), LightConv3x3(mid_channels, mid_channels), LightConv3x3(
            mid_channels, mid_channels), LightConv3x3(mid_channels,
            mid_channels))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
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

    def __init__(self, blocks, layers, channels, IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0],
            channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1],
            channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2],
            channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels,
        reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels),
                nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
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


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=
        False, stride=1, downsample=None, radix=1, cardinality=1,
        bottleneck_width=64, avd=False, avd_first=False, dilation=1,
        is_first=False, rectified_conv=False, rectify_avg=False,
        dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False
            )
        if with_ibn:
            self.bn1 = IBN(group_width, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, group_width, num_splits)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        if radix > 1:
            self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=
                3, stride=stride, padding=dilation, dilation=dilation,
                groups=cardinality, bias=False, radix=radix, rectify=
                rectified_conv, rectify_avg=rectify_avg, norm_layer=bn_norm,
                num_splits=num_splits, dropblock_prob=dropblock_prob)
        elif rectified_conv:
            self.conv2 = RFConv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False, average_mode=rectify_avg)
            self.bn2 = get_norm(bn_norm, group_width, num_splits)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation, dilation=dilation, groups=
                cardinality, bias=False)
            self.bn2 = get_norm(bn_norm, group_width, num_splits)
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias
            =False)
        self.bn3 = get_norm(bn_norm, planes * 4, num_splits)
        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNest(nn.Module):
    """ResNet Variants ResNest
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

    def __init__(self, last_stride, bn_norm, num_splits, with_ibn, with_nl,
        block, layers, non_layers, radix=1, groups=1, bottleneck_width=64,
        dilated=False, dilation=1, deep_stem=False, stem_width=64, avg_down
        =False, rectified_conv=False, rectify_avg=False, avd=False,
        avd_first=False, final_drop=0.0, dropblock_prob=0, last_gamma=False):
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
            self.conv1 = nn.Sequential(conv_layer(3, stem_width,
                kernel_size=3, stride=2, padding=1, bias=False, **
                conv_kwargs), get_norm(bn_norm, stem_width, num_splits), nn
                .ReLU(inplace=True), conv_layer(stem_width, stem_width,
                kernel_size=3, stride=1, padding=1, bias=False, **
                conv_kwargs), get_norm(bn_norm, stem_width, num_splits), nn
                .ReLU(inplace=True), conv_layer(stem_width, stem_width * 2,
                kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs))
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding
                =3, bias=False, **conv_kwargs)
        self.bn1 = get_norm(bn_norm, self.inplanes, num_splits)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm,
            num_splits, with_ibn=with_ibn, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm,
            num_splits, with_ibn=with_ibn)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], 1,
                bn_norm, num_splits, with_ibn=with_ibn, dilation=2,
                dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], 1,
                bn_norm, num_splits, with_ibn=with_ibn, dilation=4,
                dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], 2,
                bn_norm, num_splits, with_ibn=with_ibn, dilation=1,
                dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], 1,
                bn_norm, num_splits, with_ibn=with_ibn, dilation=2,
                dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], 2,
                bn_norm, num_splits, with_ibn=with_ibn, dropblock_prob=
                dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3],
                last_stride, bn_norm, num_splits, with_ibn=with_ibn,
                dropblock_prob=dropblock_prob)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm='BN',
        num_splits=1, with_ibn=False, dilation=1, dropblock_prob=0.0,
        is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride,
                        stride=stride, ceil_mode=True, count_include_pad=False)
                        )
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                        ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.
                    expansion, kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.
                    expansion, kernel_size=1, stride=stride, bias=False))
            down_layers.append(get_norm(bn_norm, planes * block.expansion,
                num_splits))
            downsample = nn.Sequential(*down_layers)
        layers = []
        if planes == 512:
            with_ibn = False
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, bn_norm, num_splits,
                with_ibn, stride, downsample=downsample, radix=self.radix,
                cardinality=self.cardinality, bottleneck_width=self.
                bottleneck_width, avd=self.avd, avd_first=self.avd_first,
                dilation=1, is_first=is_first, rectified_conv=self.
                rectified_conv, rectify_avg=self.rectify_avg,
                dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, bn_norm, num_splits,
                with_ibn, stride, downsample=downsample, radix=self.radix,
                cardinality=self.cardinality, bottleneck_width=self.
                bottleneck_width, avd=self.avd, avd_first=self.avd_first,
                dilation=2, is_first=is_first, rectified_conv=self.
                rectified_conv, rectify_avg=self.rectify_avg,
                dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, num_splits,
                with_ibn, radix=self.radix, cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width, avd=self.avd,
                avd_first=self.avd_first, dilation=dilation, rectified_conv
                =self.rectified_conv, rectify_avg=self.rectify_avg,
                dropblock_prob=dropblock_prob, last_gamma=self.last_gamma))
        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, num_splits):
        self.NL_1 = nn.ModuleList([Non_local(256, bn_norm, num_splits) for
            _ in range(non_layers[0])])
        self.NL_1_idx = sorted([(layers[0] - (i + 1)) for i in range(
            non_layers[0])])
        self.NL_2 = nn.ModuleList([Non_local(512, bn_norm, num_splits) for
            _ in range(non_layers[1])])
        self.NL_2_idx = sorted([(layers[1] - (i + 1)) for i in range(
            non_layers[1])])
        self.NL_3 = nn.ModuleList([Non_local(1024, bn_norm, num_splits) for
            _ in range(non_layers[2])])
        self.NL_3_idx = sorted([(layers[2] - (i + 1)) for i in range(
            non_layers[2])])
        self.NL_4 = nn.ModuleList([Non_local(2048, bn_norm, num_splits) for
            _ in range(non_layers[3])])
        self.NL_4_idx = sorted([(layers[3] - (i + 1)) for i in range(
            non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=
        False, with_se=False, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=
        False, with_se=False, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * 4, num_splits)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * 4, reduction)
        else:
            self.se = nn.Identity()
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
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, last_stride, bn_norm, num_splits, with_ibn, with_se,
        with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = get_norm(bn_norm, 64, num_splits)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm,
            num_splits, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm,
            num_splits, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm,
            num_splits, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride,
            bn_norm, num_splits, with_se=with_se)
        self.random_init()
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm='BN',
        num_splits=1, with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion, num_splits))
        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, bn_norm, num_splits,
            with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, num_splits,
                with_ibn, with_se))
        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, num_splits):
        self.NL_1 = nn.ModuleList([Non_local(256, bn_norm, num_splits) for
            _ in range(non_layers[0])])
        self.NL_1_idx = sorted([(layers[0] - (i + 1)) for i in range(
            non_layers[0])])
        self.NL_2 = nn.ModuleList([Non_local(512, bn_norm, num_splits) for
            _ in range(non_layers[1])])
        self.NL_2_idx = sorted([(layers[1] - (i + 1)) for i in range(
            non_layers[1])])
        self.NL_3 = nn.ModuleList([Non_local(1024, bn_norm, num_splits) for
            _ in range(non_layers[2])])
        self.NL_3_idx = sorted([(layers[2] - (i + 1)) for i in range(
            non_layers[2])])
        self.NL_4 = nn.ModuleList([Non_local(2048, bn_norm, num_splits) for
            _ in range(non_layers[3])])
        self.NL_4_idx = sorted([(layers[3] - (i + 1)) for i in range(
            non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, with_ibn, baseWidth, cardinality,
        stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()
        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1,
            padding=0, bias=False)
        if with_ibn:
            self.bn1 = IBN(D * C)
        else:
            self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
            padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, last_stride, with_ibn, block, layers, baseWidth=4,
        cardinality=32):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.inplanes = 64
        self.output_size = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_ibn=with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            with_ibn=with_ibn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            with_ibn=with_ibn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=
            last_stride, with_ibn=with_ibn)
        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, with_ibn=False):
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, with_ibn, self.baseWidth,
            self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_ibn, self.
                baseWidth, self.cardinality, 1, None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Arcface') and classname.find('Circle') != -1:
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes, self.feat_dim = num_classes, feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.
                feat_dim))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.
                feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0
            ), 'features.size(0) is not equal to labels.size(0)'
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size,
            self.num_classes) + torch.pow(self.centers, 2).sum(dim=1,
            keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        classes = classes
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1000000000000.0).sum() / batch_size
        return loss


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-06):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


def reid_losses(cfg, pred_class_logits, global_features, gt_classes, prefix=''
    ) ->dict:
    loss_dict = {}
    for loss_name in cfg.MODEL.LOSSES.NAME:
        loss = getattr(Loss, loss_name)(cfg)(pred_class_logits,
            global_features, gt_classes)
        loss_dict.update(loss)
    named_loss_dict = {}
    for name in loss_dict.keys():
        named_loss_dict[prefix + name] = loss_dict[name]
    del loss_dict
    return named_loss_dict


_CURRENT_STORAGE_STACK = []


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class`EventStorage` is currently enabled.
    """
    assert len(_CURRENT_STORAGE_STACK
        ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class CrossEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self._eps = cfg.MODEL.LOSSES.CE.EPSILON
        self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
        self._scale = cfg.MODEL.LOSSES.CE.SCALE
        self._topk = 1,

    def _log_accuracy(self, pred_class_logits, gt_classes):
        """
        Log the accuracy metrics to EventStorage.
        """
        bsz = pred_class_logits.size(0)
        maxk = max(self._topk)
        _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
        pred_class = pred_class.t()
        correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))
        ret = []
        for k in self._topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1.0 / bsz))
        storage = get_event_storage()
        storage.put_scalar('cls_accuracy', ret[0])

    def __call__(self, pred_class_logits, _, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy(pred_class_logits, gt_classes)
        if self._eps >= 0:
            smooth_param = self._eps
        else:
            soft_label = F.softmax(pred_class_logits, dim=1)
            smooth_param = self._alpha * soft_label[torch.arange(soft_label
                .size(0)), gt_classes].unsqueeze(1)
        log_probs = F.log_softmax(pred_class_logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (self._num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), 1 - smooth_param)
        loss = (-targets * log_probs).mean(0).sum()
        return {'loss_cls': loss * self._scale}


class OcclusionUnit(nn.Module):

    def __init__(self, in_planes=2048):
        super(OcclusionUnit, self).__init__()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.mask_layer = nn.Linear(in_planes, 1, bias=False)

    def forward(self, x):
        SpaFeat1 = self.MaxPool1(x)
        SpaFeat2 = self.MaxPool2(x)
        SpaFeat3 = self.MaxPool3(x)
        SpaFeat4 = self.MaxPool4(x)
        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.
            size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.
            size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.
            size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.
            size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), 2)
        SpatialFeatAll = SpatialFeatAll.transpose(1, 2)
        y = self.mask_layer(SpatialFeatAll)
        mask_weight = torch.sigmoid(y[:, :, (0)])
        feat_dim = SpaFeat1.size(2) * SpaFeat1.size(3)
        mask_score = F.normalize(mask_weight[:, :feat_dim], p=1, dim=1)
        mask_weight_norm = F.normalize(mask_weight, p=1, dim=1)
        mask_score = mask_score.unsqueeze(1)
        SpaFeat1 = SpaFeat1.transpose(1, 2)
        SpaFeat1 = SpaFeat1.transpose(2, 3)
        SpaFeat1 = SpaFeat1.view((SpaFeat1.size(0), SpaFeat1.size(1) *
            SpaFeat1.size(2), -1))
        global_feats = mask_score.matmul(SpaFeat1).view(SpaFeat1.shape[0], 
            -1, 1, 1)
        return global_feats, mask_weight, mask_weight_norm


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_JDAI_CV_fast_reid(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BatchDrop(*[], **{'h_ratio': 4, 'w_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BatchNorm2dReimpl(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ChannelGate(*[], **{'in_channels': 64}), [torch.rand([4, 64, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(ContextBlock(*[], **{'inplanes': 4, 'ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Conv1x1(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Conv1x1Linear(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Conv3x3(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(ConvLayer(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(DataParallelWithCallback(*[], **{'module': _mock_layer()}), [], {'input': torch.rand([4, 4])})

    def test_010(self):
        self._check(FRN(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(FastGlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(FrozenBatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(GELU(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(GeneralizedMeanPooling(*[], **{'norm': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_016(self):
        self._check(GeneralizedMeanPoolingP(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_017(self):
        self._check(GhostBatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(LightConv3x3(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_019(self):
        self._check(MemoryEfficientSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_020(self):
        self._check(Mish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_021(self):
        self._check(OSBlock(*[], **{'in_channels': 64, 'out_channels': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_022(self):
        self._check(OcclusionUnit(*[], **{}), [torch.rand([4, 2048, 64, 64])], {})

    @_fails_compile()
    def test_023(self):
        self._check(SplAtConv2d(*[], **{'in_channels': 4, 'channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_024(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_025(self):
        self._check(TLU(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_026(self):
        self._check(rSoftMax(*[], **{'radix': 4, 'cardinality': 4}), [torch.rand([4, 4, 4, 4])], {})

