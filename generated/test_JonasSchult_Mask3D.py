import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
evaluate_semantic_instance = _module
util = _module
util_3d = _module
conf = _module
outdoor_semseg = _module
base_preprocessing = _module
matterport_preprocessing = _module
rio_preprocessing = _module
s3dis_preprocessing = _module
scannet_preprocessing = _module
semantic_kitti_preprocessing = _module
stpls3d_preprocessing = _module
random_cuboid = _module
scannet200 = _module
scannet200_constants = _module
scannet200_splits = _module
semseg = _module
utils = _module
main_instance_segmentation = _module
models = _module
criterion = _module
mask3d = _module
matcher = _module
metrics = _module
confusionmatrix = _module
misc = _module
model = _module
modules = _module
common = _module
helpers_3detr = _module
resnet_block = _module
senet_block = _module
position_embedding = _module
res16unet = _module
resnet = _module
resunet = _module
wrapper = _module
merge_exports = _module
pointnet2_modules = _module
pointnet2_test = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
trainer = _module
trainer = _module
gradflow_check = _module
kfold = _module
pc_visualizations = _module
point_cloud_utils = _module
pointops2 = _module
functions = _module
pointops = _module
pointops2 = _module
pointops_ablation = _module
test_attention_op_step1 = _module
test_attention_op_step1_v2 = _module
test_attention_op_step2 = _module
test_relative_pos_encoding_op_step1 = _module
test_relative_pos_encoding_op_step1_v2 = _module
test_relative_pos_encoding_op_step1_v3 = _module
test_relative_pos_encoding_op_step2 = _module
test_relative_pos_encoding_op_step2_v2 = _module
setup = _module
src = _module
utils = _module
box_util = _module
eval_det = _module
metric_util = _module
nms = _module
nn_distance = _module
pc_util = _module
tf_logger = _module
tf_visualizer = _module

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


import inspect


from copy import deepcopy


from uuid import uuid4


import torch


from scipy import stats


import logging


from typing import List


from typing import Optional


from typing import Union


from typing import Tuple


from random import random


import numpy as np


from torch.utils.data import Dataset


from itertools import product


from random import sample


from random import uniform


from random import choice


from random import randrange


import numpy


import scipy


import torch.nn.functional as F


from torch import nn


import torch.nn as nn


from torch.nn import functional as F


from torch.cuda.amp import autocast


from scipy.optimize import linear_sum_assignment


import torch.distributed as dist


import torchvision


from torch import Tensor


from functools import partial


import copy


from enum import Enum


import random


from torch.nn import Module


from torch.autograd import gradcheck


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import matplotlib


from collections import defaultdict


from sklearn.cluster import DBSCAN


import functools


import time


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -torch.abs(gt_class_logits)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio, class_weights):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, 'CLASS WEIGHTS DO NOT MATCH'
            empty_weight[:-1] = torch.tensor(self.class_weights)
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=253)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        loss_masks = []
        loss_dices = []
        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs['pred_masks'][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]
            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1], device=target_mask.device)[:int(self.num_points * target_mask.shape[1])]
            else:
                point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)
            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()
            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        return {'loss_mask': torch.sum(torch.stack(loss_masks)), 'loss_dice': torch.sum(torch.stack(loss_dices))}
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t[mask_type] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(src_masks, lambda logits: calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, mask_type), 'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks, mask_type)}
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {'labels': self.loss_labels, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type)

    def forward(self, outputs, targets, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets, mask_type)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, mask_type))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, mask_type)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = ['matcher: {}'.format(self.matcher.__repr__(_repr_indent=8)), 'losses: {}'.format(self.losses), 'weight_dict: {}'.format(self.weight_dict), 'num_classes: {}'.format(self.num_classes), 'eos_coef: {}'.format(self.eos_coef), 'num_points: {}'.format(self.num_points), 'oversample_ratio: {}'.format(self.oversample_ratio), 'importance_sample_ratio: {}'.format(self.importance_sample_ratio)]
        _repr_indent = 4
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


ACTIVATION_DICT = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': partial(nn.LeakyReLU, negative_slope=0.1)}


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    Used for nn.Transformer that uses a HW x N x C rep
    """

    def forward(self, x):
        """
        x: HW x N x C
        permute to N x C x HW
        Apply BN on C
        permute back
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        x = x.permute(2, 0, 1)
        return x


NORM_DICT = {'bn': BatchNormDim1Swap, 'bn1d': nn.BatchNorm1d, 'id': nn.Identity, 'ln': nn.LayerNorm}


WEIGHT_INIT_DICT = {'xavier_uniform': nn.init.xavier_uniform_}


class GenericMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, norm_fn_name=None, activation='relu', use_conv=False, dropout=None, hidden_use_bias=False, output_use_bias=True, output_use_activation=False, output_use_norm=False, weight_init_name=None):
        super().__init__()
        activation = ACTIVATION_DICT[activation]
        norm = None
        if norm_fn_name is not None:
            norm = NORM_DICT[norm_fn_name]
        if norm_fn_name == 'ln' and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)
        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]
        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            if use_conv:
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
            layers.append(layer)
            if norm:
                layers.append(norm(x))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
        layers.append(layer)
        if output_use_norm:
            layers.append(norm(output_dim))
        if output_use_activation:
            layers.append(activation())
        self.layers = nn.Sequential(*layers)
        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

    def do_weight_init(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for _, param in self.named_parameters():
            if param.dim() > 1:
                func(param)

    def forward(self, x):
        output = self.layers(x)
        return output


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device), torch.ones((src_range[0].shape[0], 3), device=src_range[0].device)]
    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]
    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape
    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (pred_xyz - src_range[0][:, None, :]) * dst_diff / src_diff + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):

    def __init__(self, temperature=10000, normalize=False, scale=None, pos_type='fourier', d_pos=None, d_in=3, gauss_scale=1.0):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ['sine', 'fourier']
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == 'fourier':
            assert d_pos is not None
            assert d_pos % 2 == 0
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer('gauss_B', B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        orig_xyz = xyz
        xyz = orig_xyz.clone()
        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)
        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        rems = num_channels - ndim * xyz.shape[2]
        assert ndim % 2 == 0, f'Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}'
        final_embeds = []
        prev_dim = 0
        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                cdim += 2
                rems -= 2
            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2
        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]
        orig_xyz = xyz
        xyz = orig_xyz.clone()
        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)
        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        if self.pos_type == 'sine':
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == 'fourier':
            with torch.no_grad():
                out = self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f'Unknown {self.pos_type}')
        return out

    def extra_repr(self):
        st = f'type={self.pos_type}, scale={self.scale}, normalize={self.normalize}'
        if hasattr(self, 'gauss_B'):
            st += f', gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}'
        return st


class PositionalEncoding3D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / 10000 ** (torch.arange(0, channels, 2).float() / channels)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum('bi,j->bij', pos_x, self.inv_freq)
        sin_inp_y = torch.einsum('bi,j->bij', pos_y, self.inv_freq)
        sin_inp_z = torch.einsum('bi,j->bij', pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, :self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class ConvType(Enum):
    """
    Define the kernel region type
    """
    HYPERCUBE = 0, 'HYPERCUBE'
    SPATIAL_HYPERCUBE = 1, 'SPATIAL_HYPERCUBE'
    SPATIO_TEMPORAL_HYPERCUBE = 2, 'SPATIO_TEMPORAL_HYPERCUBE'
    HYPERCROSS = 3, 'HYPERCROSS'
    SPATIAL_HYPERCROSS = 4, 'SPATIAL_HYPERCROSS'
    SPATIO_TEMPORAL_HYPERCROSS = 5, 'SPATIO_TEMPORAL_HYPERCROSS'
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, 'SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS '

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), 'conv_type must be of ConvType'
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [kernel_size] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [kernel_size] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        axis_types = [ME.RegionType.HYPER_CUBE] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, 'Dimension must be a positive integer'
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(kernel_size, stride, dilation, region_type=region_type, axis_types=None, dimension=D)
    return ME.MinkowskiConvolution(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, kernel_generator=kernel_generator, dimension=D)


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz, npoint):
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class Mask3D(nn.Module):

    def __init__(self, config, hidden_dim, num_queries, num_heads, dim_feedforward, sample_sizes, shared_decoder, num_classes, num_decoders, dropout, pre_norm, positional_encoding_type, non_parametric_queries, train_on_segments, normalize_pos_enc, use_level_embed, scatter_type, hlevels, use_np_features, voxel_size, max_sample_size, random_queries, gauss_scale, random_query_both, random_normal):
        super().__init__()
        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]
        self.mask_features_head = conv(self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3)
        if self.scatter_type == 'mean':
            self.scatter_fn = scatter_mean
        elif self.scatter_type == 'max':
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(mask, p2s, dim=dim)[0]
        else:
            assert False, 'Scatter function not known'
        assert not use_np_features or non_parametric_queries, 'np features only with np queries'
        if self.non_parametric_queries:
            self.query_projection = GenericMLP(input_dim=self.mask_dim, hidden_dims=[self.mask_dim], output_dim=self.mask_dim, use_conv=True, output_use_activation=True, hidden_use_bias=True)
            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(nn.Linear(sizes[-1], hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        elif self.random_query_both:
            self.query_projection = GenericMLP(input_dim=2 * self.mask_dim, hidden_dims=[2 * self.mask_dim], output_dim=2 * self.mask_dim, use_conv=True, output_use_activation=True, hidden_use_bias=True)
        else:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_pos = nn.Embedding(num_queries, hidden_dim)
        if self.use_level_embed:
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)
        self.mask_embed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)
        if self.pos_enc_type == 'legacy':
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == 'fourier':
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type='fourier', d_pos=self.mask_dim, gauss_scale=self.gauss_scale, normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == 'sine':
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type='sine', d_pos=self.mask_dim, normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'
        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()
        num_shared = self.num_decoders if not self.shared_decoder else 1
        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(CrossAttentionLayer(d_model=self.mask_dim, nhead=self.num_heads, dropout=self.dropout, normalize_before=self.pre_norm))
                tmp_squeeze_attention.append(nn.Linear(sizes[hlevel], self.mask_dim))
                tmp_self_attention.append(SelfAttentionLayer(d_model=self.mask_dim, nhead=self.num_heads, dropout=self.dropout, normalize_before=self.pre_norm))
                tmp_ffn_attention.append(FFNLayer(d_model=self.mask_dim, dim_feedforward=dim_feedforward, dropout=self.dropout, normalize_before=self.pre_norm))
            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []
        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i].decomposed_features:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]
                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(), input_range=[scene_min, scene_max])
                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))
        return pos_encodings_pcd

    def forward(self, x, point2segment=None, raw_coordinates=None, is_eval=False):
        pcd_features, aux = self.backbone(x)
        batch_size = len(x.decomposed_coordinates)
        with torch.no_grad():
            coordinates = me.SparseTensor(features=raw_coordinates, coordinate_manager=aux[-1].coordinate_manager, coordinate_map_key=aux[-1].coordinate_map_key, device=aux[-1].device)
            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))
            coords.reverse()
        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)
        if self.train_on_segments:
            mask_segments = []
            for i, mask_feature in enumerate(mask_features.decomposed_features):
                mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))
        sampled_coords = None
        if self.non_parametric_queries:
            fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(), self.num_queries).squeeze(0).long() for i in range(len(x.decomposed_coordinates))]
            sampled_coords = torch.stack([coordinates.decomposed_features[i][fps_idx[i].long(), :] for i in range(len(fps_idx))])
            mins = torch.stack([coordinates.decomposed_features[i].min(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
            maxs = torch.stack([coordinates.decomposed_features[i].max(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
            query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])
            query_pos = self.query_projection(query_pos)
            if not self.use_np_features:
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = torch.stack([pcd_features.decomposed_features[i][fps_idx[i].long(), :] for i in range(len(fps_idx))])
                queries = self.np_feature_projection(queries)
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:
            query_pos = torch.rand(batch_size, self.mask_dim, self.num_queries, device=x.device) - 0.5
            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = torch.rand(batch_size, 2 * self.mask_dim, self.num_queries, device=x.device) - 0.5
            else:
                query_pos_feat = torch.randn(batch_size, 2 * self.mask_dim, self.num_queries, device=x.device)
            queries = query_pos_feat[:, :self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim:, :].permute((2, 0, 1))
        else:
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)
        predictions_class = []
        predictions_mask = []
        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                if self.train_on_segments:
                    output_class, outputs_mask, attn_mask = self.mask_module(queries, mask_features, mask_segments, len(aux) - hlevel - 1, ret_attn_mask=True, point2segment=point2segment, coords=coords)
                else:
                    output_class, outputs_mask, attn_mask = self.mask_module(queries, mask_features, None, len(aux) - hlevel - 1, ret_attn_mask=True, point2segment=None, coords=coords)
                decomposed_aux = aux[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features
                curr_sample_size = max([pcd.shape[0] for pcd in decomposed_aux])
                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError('only a single point gives nans in cross-attention')
                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])
                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = decomposed_aux[k].shape[0]
                    if pcd_size <= curr_sample_size:
                        idx = torch.zeros(curr_sample_size, dtype=torch.long, device=queries.device)
                        midx = torch.ones(curr_sample_size, dtype=torch.bool, device=queries.device)
                        idx[:pcd_size] = torch.arange(pcd_size, device=queries.device)
                        midx[:pcd_size] = False
                    else:
                        idx = torch.randperm(decomposed_aux[k].shape[0], device=queries.device)[:curr_sample_size]
                        midx = torch.zeros(curr_sample_size, dtype=torch.bool, device=queries.device)
                    rand_idx.append(idx)
                    mask_idx.append(midx)
                batched_aux = torch.stack([decomposed_aux[k][rand_idx[k], :] for k in range(len(rand_idx))])
                batched_attn = torch.stack([decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))])
                batched_pos_enc = torch.stack([pos_encodings_pcd[hlevel][0][k][rand_idx[k], :] for k in range(len(rand_idx))])
                batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False
                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])
                src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux.permute((1, 0, 2)))
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]
                output = self.cross_attention[decoder_counter][i](queries.permute((1, 0, 2)), src_pcd, memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)), memory_key_padding_mask=None, pos=batched_pos_enc.permute((1, 0, 2)), query_pos=query_pos)
                output = self.self_attention[decoder_counter][i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos)
                queries = self.ffn_attention[decoder_counter][i](output).permute((1, 0, 2))
                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)
        if self.train_on_segments:
            output_class, outputs_mask = self.mask_module(queries, mask_features, mask_segments, 0, ret_attn_mask=False, point2segment=point2segment, coords=coords)
        else:
            output_class, outputs_mask = self.mask_module(queries, mask_features, None, 0, ret_attn_mask=False, point2segment=None, coords=coords)
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
        return {'pred_logits': predictions_class[-1], 'pred_masks': predictions_mask[-1], 'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask), 'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None, 'backbone_features': pcd_features}

    def mask_module(self, query_feat, mask_features, mask_segments, num_pooling_steps, ret_attn_mask=True, point2segment=None, coords=None):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        output_masks = []
        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(mask_features.C[-1, 0] + 1):
                output_masks.append(mask_features.decomposed_features[i] @ mask_embed[i].T)
        output_masks = torch.cat(output_masks)
        outputs_mask = me.SparseTensor(features=output_masks, coordinate_manager=mask_features.coordinate_manager, coordinate_map_key=mask_features.coordinate_map_key)
        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask.float())
            attn_mask = me.SparseTensor(features=attn_mask.F.detach().sigmoid() < 0.5, coordinate_manager=attn_mask.coordinate_manager, coordinate_map_key=attn_mask.coordinate_map_key)
            if point2segment is not None:
                return outputs_class, output_segments, attn_mask
            else:
                return outputs_class, outputs_mask.decomposed_features, attn_mask
        if point2segment is not None:
            return outputs_class, output_segments
        else:
            return outputs_class, outputs_mask.decomposed_features

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        return [{'pred_logits': a, 'pred_masks': b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, 1 - targets)
    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float=1, cost_mask: float=1, cost_dice: float=1, num_points: int=0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_type):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)
            tgt_ids = targets[b]['labels'].clone()
            filter_ignore = tgt_ids == 253
            tgt_ids[filter_ignore] = 0
            cost_class = -out_prob[:, tgt_ids]
            cost_class[:, filter_ignore] = -1.0
            out_mask = outputs['pred_masks'][b].T
            tgt_mask = targets[b][mask_type]
            if self.num_points != -1:
                point_idx = torch.randperm(tgt_mask.shape[1], device=tgt_mask.device)[:int(self.num_points * tgt_mask.shape[1])]
            else:
                point_idx = torch.arange(tgt_mask.shape[1], device=tgt_mask.device)
            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask[:, point_idx], tgt_mask[:, point_idx])
                cost_dice = batch_dice_loss_jit(out_mask[:, point_idx], tgt_mask[:, point_idx])
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = 'Matcher ' + self.__class__.__name__
        body = ['cost_class: {}'.format(self.cost_class), 'cost_mask: {}'.format(self.cost_mask), 'cost_dice: {}'.format(self.cost_dice)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(ME.MinkowskiInstanceNorm(n_channels), ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
    else:
        raise ValueError(f'Norm type: {norm_type} not supported')


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, bn_momentum=0.1, D=3):
        super().__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, conv_type=conv_type, D=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, bn_momentum=0.1, D=3):
        super().__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(BottleneckBase):
    NORM_TYPE = NormType.BATCH_NORM


class BottleneckIN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BottleneckINBN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16, D=-1):
        super().__init__()
        self.fc = nn.Sequential(ME.MinkowskiLinear(channel, channel // reduction), ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(channel // reduction, channel), ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling(dimension=D)
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication(dimension=D)

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBasicBlock(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, reduction=16, D=-1):
        super().__init__(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, conv_type=conv_type, D=D)
        self.se = SELayer(planes, reduction=reduction, D=D)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, conv_type=ConvType.HYPERCUBE, D=3, reduction=16):
        super().__init__(inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, conv_type=conv_type, D=D)
        self.se = SELayer(planes * self.expansion, reduction=reduction, D=D)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Wrapper(Module):
    """
  Wrapper for the segmentation networks.
  """
    OUT_PIXEL_DIST = -1

    def __init__(self, NetClass, in_nchannel, out_nchannel, config):
        super().__init__()
        self.initialize_filter(NetClass, in_nchannel, out_nchannel, config)

    def initialize_filter(self, NetClass, in_nchannel, out_nchannel, config):
        raise NotImplementedError('Must initialize a model and a filter')

    def forward(self, x, coords, colors=None):
        soutput = self.model(x)
        if not self.training or random.random() < 0.5:
            wrapper_coords = self.filter.initialize_coords(self.model, coords, colors)
            finput = SparseTensor(soutput.F, wrapper_coords)
            soutput = self.filter(finput)
        return soutput


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
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

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


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


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


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
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
    (BatchNormDim1Swap,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
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
    (CrossAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FFNLayer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GenericMLP,
     lambda: ([], {'input_dim': 4, 'hidden_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding3D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SelfAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_JonasSchult_Mask3D(_paritybench_base):
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

