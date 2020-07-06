import sys
_module = sys.modules[__name__]
del sys
checkpoints = _module
datasets = _module
charades = _module
transforms = _module
GSTEG = _module
main = _module
models = _module
i3d = _module
AsyncTFBase = _module
AsyncTFCriterion = _module
BalanceLabels = _module
BlockGradient = _module
EqualizeGradNorm = _module
VerboseGradients = _module
layers = _module
opts = _module
train = _module
utils = _module
map = _module
tee = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from collections import OrderedDict


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torch.utils.data as data


import numpy as np


import random


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.nn.functional as F


from torch.autograd import Variable


import math


from random import random


from torch.autograd import Function


import time


import itertools


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - s % self.stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1), stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False, name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self._output_channels, kernel_size=self._kernel_shape, stride=self._stride, padding=0, bias=self._use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - s % self._stride[dim], 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        pad = pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b
        x = F.pad(x, pad)
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3], name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3], name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0, name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """
    VALID_ENDPOINTS = 'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions'

    def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return
        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')
        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes, kernel_shape=[1, 1, 1], padding=0, activation_fn=None, use_batch_norm=False, use_bias=True, name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x, only_feat=True):
        feat = self.extract_features(x)
        if only_feat:
            return feat
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3).squeeze(2)
        return logits, feat

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        feat = self.avg_pool(x)
        feat = feat.squeeze(3).squeeze(3).squeeze(2)
        return feat


class BasicModule(nn.Module):

    def __init__(self, inDim, outDim, hidden_dim=1000, dp_rate=0.3):
        super(BasicModule, self).__init__()
        self.layers = nn.Sequential(nn.Linear(inDim, hidden_dim), nn.ReLU(), nn.Dropout(p=dp_rate), nn.Linear(hidden_dim, outDim))

    def forward(self, x):
        return self.layers(x)


class AsyncTFBase(nn.Module):

    def __init__(self, dim, s_classes, o_classes, v_classes, _BaseModule=BasicModule):
        super(AsyncTFBase, self).__init__()
        self.s_classes = s_classes
        self.o_classes = o_classes
        self.v_classes = v_classes
        self.num_low_rank = 5
        self.s = nn.Sequential(nn.Linear(dim, 1000), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(1000, 1000), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(1000, self.s_classes))
        self.o = nn.Linear(dim, self.o_classes)
        self.v = nn.Linear(dim, self.v_classes)
        self.so_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.so_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        self.ov_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.ov_b = _BaseModule(dim, self.num_low_rank * self.v_classes)
        self.vs_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vs_b = _BaseModule(dim, self.num_low_rank * self.s_classes)
        self.ss_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.ss_b = _BaseModule(dim, self.num_low_rank * self.s_classes)
        self.oo_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.oo_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        self.vv_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vv_b = _BaseModule(dim, self.num_low_rank * self.v_classes)
        self.so_t_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.so_t_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        self.ov_t_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.ov_t_b = _BaseModule(dim, self.num_low_rank * self.v_classes)
        self.vs_t_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vs_t_b = _BaseModule(dim, self.num_low_rank * self.s_classes)
        self.os_t_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.os_t_b = _BaseModule(dim, self.num_low_rank * self.s_classes)
        self.vo_t_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vo_t_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        self.sv_t_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.sv_t_b = _BaseModule(dim, self.num_low_rank * self.v_classes)

    def forward(self, rgb_feat):
        s = self.s(rgb_feat)
        o = self.o(rgb_feat)
        v = self.v(rgb_feat)
        feat = rgb_feat
        so_a = self.so_a(feat).view(-1, self.s_classes, self.num_low_rank)
        so_b = self.so_b(feat).view(-1, self.num_low_rank, self.o_classes)
        so = torch.bmm(so_a, so_b)
        ov_a = self.ov_a(feat).view(-1, self.o_classes, self.num_low_rank)
        ov_b = self.ov_b(feat).view(-1, self.num_low_rank, self.v_classes)
        ov = torch.bmm(ov_a, ov_b)
        vs_a = self.vs_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vs_b = self.vs_b(feat).view(-1, self.num_low_rank, self.s_classes)
        vs = torch.bmm(vs_a, vs_b)
        ss_a = self.ss_a(feat).view(-1, self.s_classes, self.num_low_rank)
        ss_b = self.ss_b(feat).view(-1, self.num_low_rank, self.s_classes)
        ss = torch.bmm(ss_a, ss_b)
        oo_a = self.oo_a(feat).view(-1, self.o_classes, self.num_low_rank)
        oo_b = self.oo_b(feat).view(-1, self.num_low_rank, self.o_classes)
        oo = torch.bmm(oo_a, oo_b)
        vv_a = self.vv_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vv_b = self.vv_b(feat).view(-1, self.num_low_rank, self.v_classes)
        vv = torch.bmm(vv_a, vv_b)
        so_t_a = self.so_t_a(feat).view(-1, self.s_classes, self.num_low_rank)
        so_t_b = self.so_t_b(feat).view(-1, self.num_low_rank, self.o_classes)
        so_t = torch.bmm(so_t_a, so_t_b)
        ov_t_a = self.ov_t_a(feat).view(-1, self.o_classes, self.num_low_rank)
        ov_t_b = self.ov_t_b(feat).view(-1, self.num_low_rank, self.v_classes)
        ov_t = torch.bmm(ov_t_a, ov_t_b)
        vs_t_a = self.vs_t_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vs_t_b = self.vs_t_b(feat).view(-1, self.num_low_rank, self.s_classes)
        vs_t = torch.bmm(vs_t_a, vs_t_b)
        os_t_a = self.os_t_a(feat).view(-1, self.o_classes, self.num_low_rank)
        os_t_b = self.os_t_b(feat).view(-1, self.num_low_rank, self.s_classes)
        os_t = torch.bmm(os_t_a, os_t_b)
        vo_t_a = self.vo_t_a(feat).view(-1, self.v_classes, self.num_low_rank)
        vo_t_b = self.vo_t_b(feat).view(-1, self.num_low_rank, self.o_classes)
        vo_t = torch.bmm(vo_t_a, vo_t_b)
        sv_t_a = self.sv_t_a(feat).view(-1, self.s_classes, self.num_low_rank)
        sv_t_b = self.sv_t_b(feat).view(-1, self.num_low_rank, self.v_classes)
        sv_t = torch.bmm(sv_t_a, sv_t_b)
        return s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t


class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, inputs, weights):
        ctx.save_for_backward(inputs, weights)
        return inputs.clone()

    @staticmethod
    def backward(ctx, grad_output):
        _, weights = ctx.saved_variables
        return grad_output * weights, None


def populate(dict, ind, val=0):
    if ind not in dict:
        dict[ind] = val


class BalanceLabels(nn.Module):

    def __init__(self):
        super(BalanceLabels, self).__init__()
        self.zerocounts = {}
        self.counts = {}
        self.total = 0

    def update_counts(self, target):
        n = target.shape[0]
        tt = target.sum(0)
        for j, t in enumerate(tt):
            populate(self.counts, j)
            populate(self.zerocounts, j)
            self.counts[j] += t.item()
            self.zerocounts[j] += n - t.item()
        self.total += n

    def get_weights(self, target):
        weights = torch.zeros(*target.shape)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i, j].item() == 0:
                    weights[i, j] = self.zerocounts[j]
                else:
                    weights[i, j] = self.counts[j]
        avg = self.total / 2
        return Variable(avg / weights)

    def forward(self, inputs, target):
        self.update_counts(target)
        weights = self.get_weights(target)
        return ScaleGrad.apply(inputs, weights)


def avg(iterator, weight=1.0):
    item, w = next(iterator)
    total = item.clone() * w
    n = 1.0
    for i, (item, w) in enumerate(iterator):
        w1 = 1.0 * weight ** (i + 1)
        total += item * w1 * w
        n += w1
    return total / n


class MessagePassing(object):

    def __init__(self, maxsize, w_temporal, w_spatio, decay, sigma, ns, no, nv):
        super(MessagePassing, self).__init__()
        self.maxsize = maxsize
        self.w_temporal = w_temporal
        self.w_spatio = w_spatio
        self.decay = decay
        self.sigma = sigma
        self.s_storage = {}
        self.s_storage_gt = {}
        self.o_storage = {}
        self.o_storage_gt = {}
        self.v_storage = {}
        self.v_storage_gt = {}
        self.training = self.training if hasattr(self, 'training') else True
        self.ns = ns
        self.no = no
        self.nv = nv

    def mget(self, idtime, s_size, o_size, v_size, s_storage, o_storage, v_storage, cond=lambda t, t0: True, kernel=lambda t, t0: 1):

        def meta(ids, t0, size, storage):
            try:
                return avg(((y, kernel(t, t0)) for t, y in storage[ids] if cond(t, t0)), 1.0 / self.decay)
            except (StopIteration, KeyError):
                return torch.zeros(size)
        s_out = [meta(ids, time, s_size, s_storage) for ids, time in idtime]
        o_out = [meta(ids, time, o_size, o_storage) for ids, time in idtime]
        v_out = [meta(ids, time, v_size, v_storage) for ids, time in idtime]
        return Variable(torch.stack(s_out, 0)), Variable(torch.stack(o_out, 0)), Variable(torch.stack(v_out, 0))

    def get_msg(self, idtime, time='past', s_storage=None, o_storage=None, v_storage=None):
        s_storage = self.s_storage if s_storage is None else s_storage
        o_storage = self.o_storage if o_storage is None else o_storage
        v_storage = self.v_storage if v_storage is None else v_storage
        cond = lambda t, t0: t < t0 if time == 'past' else t > t0
        kernel = lambda t, t0: math.exp(-float(t - t0) ** 2 / (2 * self.sigma ** 2))
        return self.mget(idtime, self.ns, self.no, self.nv, s_storage, o_storage, v_storage, cond, kernel)

    def get_gt_msg(self, idtime, time='past'):
        return self.get_msg(idtime, time, self.s_storage_gt, self.o_storage_gt, self.v_storage_gt)

    def mset(self, s_msg, o_msg, v_msg, idtime, s_storage, o_storage, v_storage):
        for s_m, o_m, v_m, (ids, time) in sorted(zip(s_msg, o_msg, v_msg, idtime), key=lambda x: random()):
            if ids not in s_storage:
                s_storage[ids] = []
            if ids not in o_storage:
                o_storage[ids] = []
            if ids not in v_storage:
                v_storage[ids] = []
            s_data = s_m if type(s_m) is not torch.Tensor else s_m.data.cpu()
            o_data = o_m if type(o_m) is not torch.Tensor else o_m.data.cpu()
            v_data = v_m if type(v_m) is not torch.Tensor else v_m.data.cpu()
            s_storage[ids].append((time, s_data))
            o_storage[ids].append((time, o_data))
            v_storage[ids].append((time, v_data))
            if len(s_storage[ids]) > self.maxsize:
                del s_storage[ids][0]
            if len(o_storage[ids]) > self.maxsize:
                del o_storage[ids][0]
            if len(v_storage[ids]) > self.maxsize:
                del v_storage[ids][0]

    def set_msg(self, qs, qo, qv, idtime):
        self.mset(qs, qo, qv, idtime, self.s_storage, self.o_storage, self.v_storage)

    def set_gt_msg(self, s_target, o_target, v_target, idtime):
        s_x = s_target.data.cpu()
        o_x = o_target.data.cpu()
        v_x = v_target.data.cpu()
        self.mset(s_x, o_x, v_x, idtime, self.s_storage_gt, self.o_storage_gt, self.v_storage_gt)


def gtmat(sizes, target):
    out = torch.zeros(*sizes)
    for i, t in enumerate(target):
        t = t.data[0] if type(t) is torch.Tensor else t
        if len(sizes) == 3:
            out[(i), (t), :] = 1
        else:
            out[i, t] = 1
    return out


def winsmooth(mat, kernelsize=1):
    mat.detach()
    n = mat.shape[0]
    out = mat.clone()
    for m in range(n):
        a = max(0, m - kernelsize)
        b = min(n - 1, m + kernelsize)
        out[(m), :] = mat[a:b + 1, :].mean(0)
    return out


class AsyncTFCriterion(nn.Module, MessagePassing):

    def __init__(self, args):
        memory_size = 20
        w_temporal = 0.1
        w_spatio = 0.1
        memory_decay = 1.0
        sigma = 300
        MessagePassing.__init__(self, memory_size, w_temporal, w_spatio, memory_decay, sigma, args.s_class, args.o_class, args.v_class)
        nn.Module.__init__(self)
        self.msg_n = 5
        self.cross_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.BalanceLabels = BalanceLabels()
        self.winsmooth = 1

    def forward(self, s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t, s_target, o_target, v_target, id_time, n=1, synchronous=False):
        if o_target.dim() == 1:
            None
            o_target = Variable(gtmat(o.shape, o_target.data.long()))
        if v_target.dim() == 1:
            None
            v_target = Variable(gtmat(v.shape, v_target.data.long()))
        o_target = o_target.float()
        v_target = v_target.float()
        idtime = list(zip(id_time['id'], id_time['time']))
        s_msg, o_msg, v_msg = self.get_msg(idtime, 'past')
        s_fmsg, o_fmsg, v_fmsg = self.get_msg(idtime, 'future')
        s_loss = self.cross_loss(s, s_target)
        _qs = torch.nn.Softmax(dim=1)(s)
        o_loss = self.bce_loss(o, o_target)
        _qo = torch.nn.Sigmoid()(o)
        v_loss = self.bce_loss(v, v_target)
        _qv = torch.nn.Sigmoid()(v)
        qs_before_softmax = s.clone()
        qs_before_softmax += torch.bmm(s_msg.unsqueeze(1), ss).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(ss, s_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(o_msg.unsqueeze(1), os_t).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(so_t, o_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(v_msg.unsqueeze(1), vs_t).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(sv_t, v_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qs_before_softmax += torch.bmm(so, _qo.unsqueeze(2)).squeeze() * self.w_spatio
        qs_before_softmax += torch.bmm(_qv.unsqueeze(1), vs).squeeze() * self.w_spatio
        s_loss += self.cross_loss(qs_before_softmax, s_target)
        qs = torch.nn.Softmax(dim=1)(qs_before_softmax)
        qo_before_sigmoid = o.clone()
        qo_before_sigmoid += torch.bmm(o_msg.unsqueeze(1), oo).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(oo, o_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(v_msg.unsqueeze(1), vo_t).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(ov_t, v_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(s_msg.unsqueeze(1), so_t).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(os_t, s_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qo_before_sigmoid += torch.bmm(_qs.unsqueeze(1), so).squeeze() * self.w_spatio
        qo_before_sigmoid += torch.bmm(ov, _qv.unsqueeze(2)).squeeze() * self.w_spatio
        o_loss += self.bce_loss(qo_before_sigmoid, o_target)
        qo = torch.nn.Sigmoid()(qo_before_sigmoid)
        qv_before_sigmoid = v.clone()
        qv_before_sigmoid += torch.bmm(v_msg.unsqueeze(1), vv).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(vv, v_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(s_msg.unsqueeze(1), sv_t).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(vs_t, s_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(o_msg.unsqueeze(1), ov_t).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(vo_t, o_fmsg.unsqueeze(2)).squeeze() * self.w_temporal
        qv_before_sigmoid += torch.bmm(vs, _qs.unsqueeze(2)).squeeze() * self.w_spatio
        qv_before_sigmoid += torch.bmm(_qo.unsqueeze(1), ov).squeeze() * self.w_spatio
        v_loss += self.bce_loss(qv_before_sigmoid, v_target)
        qv = torch.nn.Sigmoid()(qv_before_sigmoid)
        self.set_msg(_qs, _qo, _qv, idtime)
        loss = s_loss + o_loss + v_loss
        if not synchronous or n > self.msg_n:
            s_out, o_out, v_out = qs.clone(), qo.clone(), qv.clone()
            if synchronous:
                s_out = winsmooth(s_out, kernelsize=self.winsmooth)
                o_out = winsmooth(o_out, kernelsize=self.winsmooth)
                v_out = winsmooth(v_out, kernelsize=self.winsmooth)
            return s_out, o_out, v_out, loss
        else:
            return self.forward(s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t, s_target, o_target, v_target, id_time, n=n + 1, synchronous=synchronous)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AsyncTFBase,
     lambda: ([], {'dim': 4, 's_classes': 4, 'o_classes': 4, 'v_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BalanceLabels,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (BasicModule,
     lambda: ([], {'inDim': 4, 'outDim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Unit3D,
     lambda: ([], {'in_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
]

class Test_yaohungt_Gated_Spatio_Temporal_Energy_Graph(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

