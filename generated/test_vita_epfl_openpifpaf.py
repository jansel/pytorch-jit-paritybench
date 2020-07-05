import sys
_module = sys.modules[__name__]
del sys
openpifpaf = _module
annotation = _module
benchmark = _module
count_ops = _module
datasets = _module
coco = _module
collate = _module
constants = _module
factory = _module
headmeta = _module
image_list = _module
decoder = _module
caf_scored = _module
caf_seeds = _module
cif_hr = _module
cif_seeds = _module
field_config = _module
generator = _module
cifcaf = _module
cifdet = _module
instance_scorer = _module
nms = _module
occupancy = _module
profiler = _module
profiler_autograd = _module
utils = _module
encoder = _module
annrescaler = _module
caf = _module
cif = _module
eval_coco = _module
export_onnx = _module
logs = _module
migrate = _module
network = _module
basenetworks = _module
factory = _module
heads = _module
losses = _module
nets = _module
trainer = _module
optimize = _module
predict = _module
show = _module
animation_frame = _module
canvas = _module
cli = _module
fields = _module
painters = _module
train = _module
train_instance_scorer = _module
transforms = _module
annotations = _module
compose = _module
crop = _module
hflip = _module
image = _module
minsize = _module
multi_scale = _module
pad = _module
preprocess = _module
random = _module
rotate = _module
scale = _module
unclipped = _module
video = _module
visualizer = _module
base = _module
cifhr = _module
seeds = _module
setup = _module
test_clis = _module
test_forward = _module
test_help = _module
test_image_scale = _module
test_input_processing = _module
test_localization = _module
test_network = _module
test_onnx_export = _module
test_train = _module
test_transforms = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
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


import logging


import time


import torchvision.models


import torchvision


import functools


from typing import Any


from typing import List


from typing import Tuple


from typing import Union


import random


class InstanceScorer(torch.nn.Module):

    def __init__(self, in_features=35):
        super(InstanceScorer, self).__init__()
        self.compute_layers = torch.nn.Sequential(torch.nn.Linear(in_features, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1), torch.nn.Sigmoid())

    def forward(self, x):
        return self.compute_layers(x - 0.5)

    def from_annotation(self, ann):
        v = torch.tensor([ann.scale()] + ann.data[:, (2)].tolist() + ann.joint_scales.tolist()).float()
        with torch.no_grad():
            return float(self.forward(v).item())


class GetPif(torch.nn.Module):

    def forward(self, heads):
        return heads[0]


class GetPifC(torch.nn.Module):

    def forward(self, heads):
        return heads[0][0]


LOG = logging.getLogger(__name__)


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, net, shortname, stride, out_features):
        super(BaseNetwork, self).__init__()
        self.net = net
        self.shortname = shortname
        self.stride = stride
        self.out_features = out_features
        LOG.info('stride = %d', self.stride)
        LOG.info('output features = %d', self.out_features)

    def forward(self, *args):
        return self.net(*args)


class InvertedResidualK(torch.nn.Module):
    """This is exactly the same as torchvision.models.shufflenet.InvertedResidual
    but with a dilation parameter."""

    def __init__(self, inp, oup, stride, *, layer_norm, dilation=1, kernel_size=3):
        super(InvertedResidualK, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        assert dilation == 1 or kernel_size == 3
        padding = 1
        if dilation != 1:
            padding = dilation
        elif kernel_size != 3:
            padding = (kernel_size - 1) // 2
        if self.stride > 1:
            self.branch1 = torch.nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=kernel_size, stride=self.stride, padding=padding, dilation=dilation), layer_norm(inp), torch.nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), torch.nn.ReLU(inplace=True))
        self.branch2 = torch.nn.Sequential(torch.nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), torch.nn.ReLU(inplace=True), self.depthwise_conv(branch_features, branch_features, kernel_size=kernel_size, stride=self.stride, padding=padding, dilation=dilation), layer_norm(branch_features), torch.nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), layer_norm(branch_features), torch.nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(in_f, out_f, kernel_size, stride=1, padding=0, bias=False, dilation=1):
        return torch.nn.Conv2d(in_f, out_f, kernel_size, stride, padding, bias=bias, groups=in_f, dilation=dilation)

    def forward(self, *args):
        x = args[0]
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = torchvision.models.shufflenetv2.channel_shuffle(out, 2)
        return out


class ShuffleNetV2K(torch.nn.Module):
    """Based on torchvision.models.ShuffleNetV2 where
    the kernel size in stages 2,3,4 is 5 instead of 3."""

    def __init__(self, stages_repeats, stages_out_channels, *, layer_norm=None):
        super(ShuffleNetV2K, self).__init__()
        if layer_norm is None:
            layer_norm = torch.nn.BatchNorm2d
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), layer_norm(output_channels), torch.nn.ReLU(inplace=True))
        input_channels = output_channels
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidualK(input_channels, output_channels, 2, layer_norm=layer_norm)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualK(output_channels, output_channels, 1, kernel_size=5, layer_norm=layer_norm))
            setattr(self, name, torch.nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = torch.nn.Sequential(torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), layer_norm(output_channels), torch.nn.ReLU(inplace=True))

    def forward(self, *args):
        x = args[0]
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x


@functools.lru_cache(maxsize=16)
def index_field_torch(shape, *, device=None, n_unsqueeze=2):
    yx = np.indices(shape, dtype=np.float32)
    xy = np.flip(yx, axis=0)
    xy = torch.from_numpy(xy.copy())
    if device is not None:
        xy = xy.to(device, non_blocking=True)
    for _ in range(n_unsqueeze):
        xy = torch.unsqueeze(xy, 0)
    return xy


class CifCafCollector(torch.nn.Module):

    def __init__(self, cif_indices, caf_indices):
        super(CifCafCollector, self).__init__()
        self.cif_indices = cif_indices
        self.caf_indices = caf_indices
        LOG.debug('cif = %s, caf = %s', cif_indices, caf_indices)

    @staticmethod
    def selector(inputs, index):
        if not isinstance(index, (list, tuple)):
            return inputs[index]
        for ind in index:
            inputs = inputs[ind]
        return inputs

    @staticmethod
    def concat_fields(fields):
        fields = [(f.view(f.shape[0], f.shape[1], f.shape[2] * f.shape[3], *f.shape[4:]) if len(f.shape) == 6 else f.view(f.shape[0], f.shape[1], f.shape[2], *f.shape[3:])) for f in fields]
        return torch.cat(fields, dim=2)

    @staticmethod
    def concat_heads(heads):
        if not heads:
            return None
        if len(heads) == 1:
            return heads[0]
        return torch.cat(heads, dim=1)

    def forward(self, *args):
        heads = args[0]
        cif_heads = [self.concat_fields(self.selector(heads, head_index)) for head_index in self.cif_indices]
        caf_heads = [self.concat_fields(self.selector(heads, head_index)) for head_index in self.caf_indices]
        cif_head = self.concat_heads(cif_heads)
        caf_head = self.concat_heads(caf_heads)
        index_field = index_field_torch(cif_head.shape[-2:], device=cif_head.device)
        if cif_head is not None:
            cif_head[:, :, 1:3] += index_field
        if caf_head is not None:
            caf_head[:, :, 1:3] += index_field
            caf_head[:, :, 3:5] += index_field
            caf_head = caf_head[:, :, (0, 1, 2, 5, 7, 3, 4, 6, 8)]
        return cif_head, caf_head


class CifdetCollector(torch.nn.Module):

    def __init__(self, indices):
        super(CifdetCollector, self).__init__()
        self.indices = indices
        LOG.debug('cifdet = %s', indices)

    @staticmethod
    def selector(inputs, index):
        if not isinstance(index, (list, tuple)):
            return inputs[index]
        for ind in index:
            inputs = inputs[ind]
        return inputs

    @staticmethod
    def concat_fields(fields):
        fields = [(f.view(f.shape[0], f.shape[1], f.shape[2] * f.shape[3], *f.shape[4:]) if len(f.shape) == 6 else f.view(f.shape[0], f.shape[1], f.shape[2], *f.shape[3:])) for f in fields]
        return torch.cat(fields, dim=2)

    @staticmethod
    def concat_heads(heads):
        if not heads:
            return None
        if len(heads) == 1:
            return heads[0]
        return torch.cat(heads, dim=1)

    def forward(self, *args):
        heads = args[0]
        cifdet_heads = [self.concat_fields(self.selector(heads, head_index)) for head_index in self.indices]
        cifdet_head = self.concat_heads(cifdet_heads)
        index_field = index_field_torch(cifdet_head.shape[-2:], device=cifdet_head.device)
        cifdet_head[:, :, 1:3] += index_field
        cifdet_head = cifdet_head[:, :, (0, 1, 2, 5, 3, 4, 6)]
        return cifdet_head,


class PifHFlip(torch.nn.Module):

    def __init__(self, keypoints, hflip):
        super(PifHFlip, self).__init__()
        flip_indices = torch.LongTensor([(keypoints.index(hflip[kp_name]) if kp_name in hflip else kp_i) for kp_i, kp_name in enumerate(keypoints)])
        LOG.debug('hflip indices: %s', flip_indices)
        self.register_buffer('flip_indices', flip_indices)

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)
        out[1][:, :, (0), :, :] *= -1.0
        return out


class PafHFlip(torch.nn.Module):

    def __init__(self, keypoints, skeleton, hflip):
        super(PafHFlip, self).__init__()
        skeleton_names = [(keypoints[j1 - 1], keypoints[j2 - 1]) for j1, j2 in skeleton]
        flipped_skeleton_names = [(hflip[j1] if j1 in hflip else j1, hflip[j2] if j2 in hflip else j2) for j1, j2 in skeleton_names]
        LOG.debug('skeleton = %s, flipped_skeleton = %s', skeleton_names, flipped_skeleton_names)
        flip_indices = list(range(len(skeleton)))
        reverse_direction = []
        for paf_i, (n1, n2) in enumerate(skeleton_names):
            if (n1, n2) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n1, n2))
            if (n2, n1) in flipped_skeleton_names:
                flip_indices[paf_i] = flipped_skeleton_names.index((n2, n1))
                reverse_direction.append(paf_i)
        LOG.debug('hflip indices: %s, reverse: %s', flip_indices, reverse_direction)
        self.register_buffer('flip_indices', torch.LongTensor(flip_indices))
        self.register_buffer('reverse_direction', torch.LongTensor(reverse_direction))

    def forward(self, *args):
        out = []
        for field in args:
            field = torch.index_select(field, 1, self.flip_indices)
            field = torch.flip(field, dims=[len(field.shape) - 1])
            out.append(field)
        out[1][:, :, (0), :, :] *= -1.0
        out[2][:, :, (0), :, :] *= -1.0
        for paf_i in self.reverse_direction:
            cc = torch.clone(out[1][:, (paf_i)])
            out[1][:, (paf_i)] = out[2][:, (paf_i)]
            out[2][:, (paf_i)] = cc
        return out


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [(lam * l) for lam, l in zip(self.lambdas, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) if loss_values else None
        return total_loss, flat_head_losses


class MultiHeadLossAutoTune(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()
        if not lambdas:
            lambdas = [(1.0) for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)
        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters
        self.log_sigmas = torch.nn.Parameter(torch.zeros((len(lambdas),), dtype=torch.float64), requires_grad=True)
        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args
        LOG.debug('losses = %d, fields = %d, targets = %d', len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll for l, f, t in zip(self.losses, head_fields, head_targets) for ll in l(f, t)]
        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.log_sigmas) == len(flat_head_losses)
        loss_values = [(lam * l / (2.0 * log_sigma.exp() ** 2)) for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses) if l is not None]
        auto_reg = [(lam * log_sigma) for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses) if l is not None]
        total_loss = sum(loss_values) + sum(auto_reg) if loss_values else None
        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(param.abs().max(dim=1)[0].clamp(min=1e-06).sum() for param in self.sparse_task_parameters)
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss
        return total_loss, flat_head_losses


def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)
    logb = 3.0 * torch.tanh(logb / 3.0)
    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def logl1_loss(logx, t, **kwargs):
    """Swap in replacement for functional.l1_loss."""
    return torch.nn.functional.l1_loss(logx, torch.log(t), **kwargs)


def quadrant(xys):
    q = torch.zeros((xys.shape[1],), dtype=torch.long)
    q[xys[(0), :] < 0.0] += 1
    q[xys[(1), :] < 0.0] += 2
    return q


def quadrant_margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))
    diffs = x - t
    qs = quadrant(diffs)
    norms = diffs.norm(dim=0)
    m1 = norms[qs == 0] > max_r1[qs == 0]
    m2 = norms[qs == 1] > max_r2[qs == 1]
    m3 = norms[qs == 2] > max_r3[qs == 2]
    m4 = norms[qs == 3] > max_r4[qs == 3]
    return torch.sum(norms[qs == 0][m1] - max_r1[qs == 0][m1]) + torch.sum(norms[qs == 1][m2] - max_r2[qs == 1][m2]) + torch.sum(norms[qs == 2][m3] - max_r3[qs == 2][m3]) + torch.sum(norms[qs == 3][m4] - max_r4[qs == 3][m4])


class Shell(torch.nn.Module):

    def __init__(self, base_net, head_nets, *, process_heads=None, cross_talk=0.0):
        super(Shell, self).__init__()
        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.process_heads = process_heads
        self.cross_talk = cross_talk

    def forward(self, *args):
        image_batch = args[0]
        if self.training and self.cross_talk:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        x = self.base_net(image_batch)
        head_outputs = [hn(x) for hn in self.head_nets]
        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)
        return head_outputs


class Shell2Scale(torch.nn.Module):

    def __init__(self, base_net, head_nets, *, reduced_stride=3):
        super(Shell2Scale, self).__init__()
        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.reduced_stride = reduced_stride

    @staticmethod
    def merge_heads(original_h, reduced_h, logb_component_indices, stride):
        mask = reduced_h[0] > original_h[0][:, :, :stride * reduced_h[0].shape[2]:stride, :stride * reduced_h[0].shape[3]:stride]
        mask_vector = torch.stack((mask, mask), dim=2)
        for ci, (original_c, reduced_c) in enumerate(zip(original_h, reduced_h)):
            if ci == 0:
                reduced_c = reduced_c * 0.5
            elif ci in logb_component_indices:
                reduced_c = torch.log(torch.exp(reduced_c) * stride)
            else:
                reduced_c = reduced_c * stride
            if len(original_c.shape) == 4:
                original_c[:, :, :stride * reduced_c.shape[2]:stride, :stride * reduced_c.shape[3]:stride][mask] = reduced_c[mask]
            elif len(original_c.shape) == 5:
                original_c[:, :, :, :stride * reduced_c.shape[3]:stride, :stride * reduced_c.shape[4]:stride][mask_vector] = reduced_c[mask_vector]
            else:
                raise Exception('cannot process component with shape {}'.format(original_c.shape))

    def forward(self, *args):
        original_input = args[0]
        original_x = self.base_net(original_input)
        original_heads = [hn(original_x) for hn in self.head_nets]
        reduced_input = original_input[:, :, ::self.reduced_stride, ::self.reduced_stride]
        reduced_x = self.base_net(reduced_input)
        reduced_heads = [hn(reduced_x) for hn in self.head_nets]
        logb_component_indices = [(2,), (3, 4)]
        for original_h, reduced_h, lci in zip(original_heads, reduced_heads, logb_component_indices):
            self.merge_heads(original_h, reduced_h, lci, self.reduced_stride)
        return original_heads


class ShellMultiScale(torch.nn.Module):

    def __init__(self, base_net, head_nets, *, process_heads=None, include_hflip=True):
        super(ShellMultiScale, self).__init__()
        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)
        self.pif_hflip = heads.PifHFlip(head_nets[0].meta.keypoints, datasets.constants.HFLIP)
        self.paf_hflip = heads.PafHFlip(head_nets[1].meta.keypoints, head_nets[1].meta.skeleton, datasets.constants.HFLIP)
        self.paf_hflip_dense = heads.PafHFlip(head_nets[2].meta.keypoints, head_nets[2].meta.skeleton, datasets.constants.HFLIP)
        self.process_heads = process_heads
        self.include_hflip = include_hflip

    def forward(self, *args):
        original_input = args[0]
        head_outputs = []
        for hflip in ([False, True] if self.include_hflip else [False]):
            for reduction in [1, 1.5, 2, 3, 5]:
                if reduction == 1.5:
                    x_red = torch.ByteTensor([(i % 3 != 2) for i in range(original_input.shape[3])])
                    y_red = torch.ByteTensor([(i % 3 != 2) for i in range(original_input.shape[2])])
                    reduced_input = original_input[:, :, (y_red), :]
                    reduced_input = reduced_input[:, :, :, (x_red)]
                else:
                    reduced_input = original_input[:, :, ::reduction, ::reduction]
                if hflip:
                    reduced_input = torch.flip(reduced_input, dims=[3])
                reduced_x = self.base_net(reduced_input)
                head_outputs += [hn(reduced_x) for hn in self.head_nets]
        if self.include_hflip:
            for mscale_i in range(5, 10):
                head_i = mscale_i * 3
                head_outputs[head_i] = self.pif_hflip(*head_outputs[head_i])
                head_outputs[head_i + 1] = self.paf_hflip(*head_outputs[head_i + 1])
                head_outputs[head_i + 2] = self.paf_hflip_dense(*head_outputs[head_i + 2])
        if self.process_heads is not None:
            head_outputs = self.process_heads(*head_outputs)
        return head_outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GetPif,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GetPifC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InstanceScorer,
     lambda: ([], {}),
     lambda: ([torch.rand([35, 35])], {}),
     True),
]

class Test_vita_epfl_openpifpaf(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

