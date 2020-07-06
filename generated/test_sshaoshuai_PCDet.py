import sys
_module = sys.modules[__name__]
del sys
pcdet = _module
config = _module
datasets = _module
augmentation_utils = _module
dbsampler = _module
dataset = _module
kitti_dataset = _module
kitti_eval = _module
eval = _module
kitti_common = _module
rotate_iou = _module
models = _module
bbox_heads = _module
anchor_target_assigner = _module
rpn_head = _module
PartA2_net = _module
detectors = _module
detector3d = _module
pointpillar = _module
second_net = _module
model_utils = _module
proposal_layer = _module
proposal_target_layer = _module
pytorch_utils = _module
resnet_utils = _module
rcnn = _module
partA2_rcnn_net = _module
rpn = _module
pillar_scatter = _module
rpn_backbone = _module
rpn_unet = _module
vfe = _module
vfe_utils = _module
iou3d_nms_utils = _module
setup = _module
roiaware_pool3d_utils = _module
setup = _module
utils = _module
box_coder_utils = _module
box_utils = _module
calibration = _module
common_utils = _module
loss_utils = _module
object3d_utils = _module
setup = _module
eval_utils = _module
test = _module
train = _module
optimization = _module
fastai_optim = _module
learning_schedules_fastai = _module
train_utils = _module

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


from torch.utils.data import DataLoader


import numpy as np


import warnings


from collections import defaultdict


import torch.utils.data as torch_data


import copy


from collections import namedtuple


import torch.nn as nn


from functools import partial


from collections import OrderedDict


from typing import List


from typing import Tuple


from torch import nn


import torch.nn.functional as F


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import Function


from scipy.spatial import Delaunay


import scipy


import logging


import torch.multiprocessing as mp


import torch.distributed as dist


import random


from abc import ABCMeta


from abc import abstractmethod


import time


import re


import torch.optim as optim


import torch.optim.lr_scheduler as lr_sched


from collections import Iterable


from torch.nn.utils import parameters_to_vector


from torch._utils import _unflatten_dense_tensors


import math


from torch.nn.utils import clip_grad_norm_


def create_anchors_3d_range(feature_size, anchor_range, sizes=((1.6, 3.9, 1.56),), rotations=(0, np.pi / 2), dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][(...), (np.newaxis), :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


class AnchorGeneratorRange(object):

    def __init__(self, anchor_ranges, sizes=((1.6, 3.9, 1.56),), rotations=(0, np.pi / 2), class_name=None, match_threshold=-1, unmatch_threshold=-1, custom_values=None, dtype=np.float32, feature_map_size=None):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values
        self._feature_map_size = feature_map_size

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def custom_values(self):
        return self.custom_values

    @property
    def feature_map_size(self):
        return self._feature_map_size

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        anchors = create_anchors_3d_range(feature_map_size, self._anchor_ranges, self._sizes, self._rotations, self._dtype)
        if self._custom_values is not None:
            custom_values = np.zeros((*anchors.shape[:-1], len(self._custom_values)), dtype=self._dtype)
            for k in range(len(self._custom_values)):
                custom_values[..., k] = self._custom_values[k]
            anchors = np.concatenate((anchors, custom_values), axis=-1)
        return anchors


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, ([0, 2])].reshape([-1, 4])


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(common_utils.limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, ([0, 1, 3, 2])], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[(inds), :] = data
    return ret


class TargetAssigner(object):

    def __init__(self, anchor_generators, pos_fraction, sample_size, region_similarity_fn_name, box_coder, logger=None):
        super().__init__()
        self.anchor_generators = anchor_generators
        self.pos_fraction = pos_fraction if pos_fraction >= 0 else None
        self.sample_size = sample_size
        self.region_similarity_calculator = getattr(self, region_similarity_fn_name)
        self.box_coder = box_coder
        self.logger = logger

    def generate_anchors(self, feature_map_size=None, use_multi_head=False):
        anchors_list = []
        matched_thresholds = [a.match_threshold for a in self.anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self.anchor_generators]
        match_list, unmatch_list = [], []
        for anchor_generator, match_thresh, unmatch_thresh in zip(self.anchor_generators, matched_thresholds, unmatched_thresholds):
            if use_multi_head:
                anchors = anchor_generator.generate(anchor_generator.feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
                ndim = len(anchor_generator.feature_map_size)
                anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
                anchors = anchors.reshape(-1, anchors.shape[-1])
            else:
                anchors = anchor_generator.generate(feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
        anchors = np.concatenate(anchors_list, axis=-2)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {'anchors': anchors, 'matched_thresholds': matched_thresholds, 'unmatched_thresholds': unmatched_thresholds}

    def generate_anchors_dict(self, feature_map_size, use_multi_head=False):
        anchors_list = []
        matched_thresholds = [a.match_threshold for a in self.anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self.anchor_generators]
        match_list, unmatch_list = [], []
        anchors_dict = {a.class_name: {} for a in self.anchor_generators}
        for anchor_generator, match_thresh, unmatch_thresh in zip(self.anchor_generators, matched_thresholds, unmatched_thresholds):
            if use_multi_head:
                anchors = anchor_generator.generate(anchor_generator.feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
                ndim = len(feature_map_size)
                anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            else:
                anchors = anchor_generator.generate(feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
            class_name = anchor_generator.class_name
            anchors_dict[class_name]['anchors'] = anchors
            anchors_dict[class_name]['matched_thresholds'] = match_list[-1]
            anchors_dict[class_name]['unmatched_thresholds'] = unmatch_list[-1]
        return anchors_dict

    @staticmethod
    def nearest_iou_similarity(boxes1, boxes2):
        boxes1_bv = rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = rbbox2d_to_near_bbox(boxes2)
        ret = iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret

    def assign_v2(self, anchors_dict, gt_boxes, anchors_mask=None, gt_classes=None, gt_names=None):
        prune_anchor_fn = None if anchors_mask is None else lambda _: np.where(anchors_mask)[0]

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, ([0, 1, 3, 4, 6])]
            gt_boxes_rbv = gt_boxes[:, ([0, 1, 3, 4, 6])]
            return self.region_similarity_calculator(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self.box_coder.encode_np(boxes, anchors)
        targets_list = []
        for class_name, anchor_dict in anchors_dict.items():
            mask = np.array([(c == class_name) for c in gt_names], dtype=np.bool_)
            targets = self.create_target_np(anchor_dict['anchors'].reshape(-1, anchor_dict['anchors'].shape[-1]), gt_boxes[mask], similarity_fn, box_encoding_fn, prune_anchor_fn=prune_anchor_fn, gt_classes=gt_classes[mask], matched_threshold=anchor_dict['matched_thresholds'], unmatched_threshold=anchor_dict['unmatched_thresholds'], positive_fraction=self.pos_fraction, rpn_batch_size=self.sample_size, norm_by_num_examples=False, box_code_size=self.box_coder.code_size)
            targets_list.append(targets)
            feature_map_size = anchor_dict['anchors'].shape[:3]
        targets_dict = {'labels': [t['labels'] for t in targets_list], 'bbox_targets': [t['bbox_targets'] for t in targets_list], 'bbox_src_targets': [t['bbox_src_targets'] for t in targets_list], 'bbox_outside_weights': [t['bbox_outside_weights'] for t in targets_list]}
        targets_dict['bbox_targets'] = np.concatenate([v.reshape(*feature_map_size, -1, self.box_coder.code_size) for v in targets_dict['bbox_targets']], axis=-2)
        targets_dict['bbox_src_targets'] = np.concatenate([v.reshape(*feature_map_size, -1, self.box_coder.code_size) for v in targets_dict['bbox_src_targets']], axis=-2)
        targets_dict['labels'] = np.concatenate([v.reshape(*feature_map_size, -1) for v in targets_dict['labels']], axis=-1)
        targets_dict['bbox_outside_weights'] = np.concatenate([v.reshape(*feature_map_size, -1) for v in targets_dict['bbox_outside_weights']], axis=-1)
        targets_dict['bbox_targets'] = targets_dict['bbox_targets'].reshape(-1, self.box_coder.code_size)
        targets_dict['bbox_src_targets'] = targets_dict['bbox_src_targets'].reshape(-1, self.box_coder.code_size)
        targets_dict['labels'] = targets_dict['labels'].reshape(-1)
        targets_dict['bbox_outside_weights'] = targets_dict['bbox_outside_weights'].reshape(-1)
        return targets_dict

    def assign_multihead(self, anchors_dict, gt_boxes, anchors_mask=None, gt_classes=None, gt_names=None):
        prune_anchor_fn = None if anchors_mask is None else lambda _: np.where(anchors_mask)[0]

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, ([0, 1, 3, 4, 6])]
            gt_boxes_rbv = gt_boxes[:, ([0, 1, 3, 4, 6])]
            return self.region_similarity_calculator(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self.box_coder.encode_np(boxes, anchors)
        targets_list = []
        for class_name, anchor_dict in anchors_dict.items():
            mask = np.array([(c == class_name) for c in gt_names], dtype=np.bool_)
            targets = self.create_target_np(anchor_dict['anchors'].reshape(-1, anchor_dict['anchors'].shape[-1]), gt_boxes[mask], similarity_fn, box_encoding_fn, prune_anchor_fn=prune_anchor_fn, gt_classes=gt_classes[mask], matched_threshold=anchor_dict['matched_thresholds'], unmatched_threshold=anchor_dict['unmatched_thresholds'], positive_fraction=self.pos_fraction, rpn_batch_size=self.sample_size, norm_by_num_examples=False, box_code_size=self.box_coder.code_size)
            targets_list.append(targets)
        targets_dict = {'labels': [t['labels'] for t in targets_list], 'bbox_targets': [t['bbox_targets'] for t in targets_list], 'bbox_outside_weights': [t['bbox_outside_weights'] for t in targets_list]}
        targets_dict['bbox_targets'] = np.concatenate([v.reshape(-1, self.box_coder.code_size) for v in targets_dict['bbox_targets']], axis=0)
        targets_dict['labels'] = np.concatenate([v.reshape(-1) for v in targets_dict['labels']], axis=0)
        targets_dict['bbox_outside_weights'] = np.concatenate([v.reshape(-1) for v in targets_dict['bbox_outside_weights']], axis=0)
        return targets_dict

    def create_target_np(self, all_anchors, gt_boxes, similarity_fn, box_encoding_fn, prune_anchor_fn=None, gt_classes=None, matched_threshold=0.6, unmatched_threshold=0.45, bbox_inside_weight=None, positive_fraction=None, rpn_batch_size=300, norm_by_num_examples=False, box_code_size=7):
        """Modified from FAIR detectron.
        Args:
            all_anchors: [num_of_anchors, box_ndim] float tensor.
            gt_boxes: [num_gt_boxes, box_ndim] float tensor.
            similarity_fn: a function, accept anchors and gt_boxes, return
                similarity matrix(such as IoU).
            box_encoding_fn: a function, accept gt_boxes and anchors, return
                box encodings(offsets).
            prune_anchor_fn: a function, accept anchors, return indices that
                indicate valid anchors.
            gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
                start with 1.
            matched_threshold: float, iou greater than matched_threshold will
                be treated as positives.
            unmatched_threshold: float, iou smaller than unmatched_threshold will
                be treated as negatives.
            bbox_inside_weight: unused
            positive_fraction: [0-1] float or None. if not None, we will try to
                keep ratio of pos/neg equal to positive_fraction when sample.
                if there is not enough positives, it fills the rest with negatives
            rpn_batch_size: int. sample size
            norm_by_num_examples: bool. norm box_weight by number of examples, but
                I recommend to do this outside.
        Returns:
            labels, bbox_targets, bbox_outside_weights
        """
        total_anchors = all_anchors.shape[0]
        if prune_anchor_fn is not None:
            inds_inside = prune_anchor_fn(all_anchors)
            anchors = all_anchors[(inds_inside), :]
            if not isinstance(matched_threshold, float):
                matched_threshold = matched_threshold[inds_inside]
            if not isinstance(unmatched_threshold, float):
                unmatched_threshold = unmatched_threshold[inds_inside]
        else:
            anchors = all_anchors
            inds_inside = None
        num_inside = len(inds_inside) if inds_inside is not None else total_anchors
        box_ndim = all_anchors.shape[1]
        if self.logger is not None:
            self.logger.info('total_anchors: {}'.format(total_anchors))
            self.logger.info('inds_inside: {}'.format(num_inside))
            self.logger.info('anchors.shape: {}'.format(anchors.shape))
        if gt_classes is None:
            gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
        labels = np.empty((num_inside,), dtype=np.int32)
        gt_ids = np.empty((num_inside,), dtype=np.int32)
        labels.fill(-1)
        gt_ids.fill(-1)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
            anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside), anchor_to_gt_argmax]
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            anchors_with_max_overlap = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds]
            gt_ids[pos_inds] = gt_inds
            bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
        else:
            bg_inds = np.arange(num_inside)
        fg_inds = np.where(labels > 0)[0]
        fg_max_overlap = None
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_max_overlap = anchor_to_gt_max[fg_inds]
        gt_pos_ids = gt_ids[fg_inds]
        if positive_fraction is not None:
            num_fg = int(positive_fraction * rpn_batch_size)
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(fg_inds, size=len(fg_inds) - num_fg, replace=False)
                labels[disable_inds] = -1
                fg_inds = np.where(labels > 0)[0]
            num_bg = rpn_batch_size - np.sum(labels > 0)
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
                labels[enable_inds] = 0
            bg_inds = np.where(labels == 0)[0]
        elif len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        bbox_targets = np.zeros((num_inside, box_code_size), dtype=all_anchors.dtype)
        bbox_src_targets = np.zeros((num_inside, box_code_size), dtype=all_anchors.dtype)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[(anchor_to_gt_argmax[fg_inds]), :]
            fg_anchors = anchors[(fg_inds), :]
            bbox_targets[(fg_inds), :] = box_encoding_fn(fg_gt_boxes, fg_anchors)
            temp_src_gt_boxes = fg_gt_boxes.copy()
            temp_src_gt_boxes[:, 0:3] = fg_gt_boxes[:, 0:3] - fg_anchors[:, 0:3]
            bbox_src_targets[(fg_inds), :] = temp_src_gt_boxes
        bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)
        if norm_by_num_examples:
            num_examples = np.sum(labels >= 0)
            num_examples = np.maximum(1.0, num_examples)
            bbox_outside_weights[labels > 0] = 1.0 / num_examples
        else:
            bbox_outside_weights[labels > 0] = 1.0
        if inds_inside is not None:
            labels = unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_src_targets = unmap(bbox_src_targets, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
        ret = {'labels': labels, 'bbox_targets': bbox_targets, 'bbox_outside_weights': bbox_outside_weights, 'assigned_anchors_overlap': fg_max_overlap, 'positive_gt_id': gt_pos_ids, 'bbox_src_targets': bbox_src_targets}
        if inds_inside is not None:
            ret['assigned_anchors_inds'] = inds_inside[fg_inds]
        else:
            ret['assigned_anchors_inds'] = fg_inds
        return ret

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self.anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    def num_anchors_per_location_class(self, class_name):
        if isinstance(class_name, int):
            class_name = self.classes[class_name]
        assert class_name in self.classes
        class_idx = self.classes.index(class_name)
        return self.anchor_generators[class_idx].num_anchors_per_localization

    @property
    def classes(self):
        return [a.class_name for a in self.anchor_generators]


class AnchorHead(nn.Module):

    def __init__(self, grid_size, anchor_target_cfg):
        super().__init__()
        anchor_cfg = anchor_target_cfg.ANCHOR_GENERATOR
        anchor_generators = []
        self.num_class = len(cfg.CLASS_NAMES)
        for cur_name in cfg.CLASS_NAMES:
            cur_cfg = None
            for a_cfg in anchor_cfg:
                if a_cfg['class_name'] == cur_name:
                    cur_cfg = a_cfg
                    break
            assert cur_cfg is not None, 'Not found anchor config: %s' % cur_name
            anchor_generator = AnchorGeneratorRange(anchor_ranges=cur_cfg['anchor_range'], sizes=cur_cfg['sizes'], rotations=cur_cfg['rotations'], class_name=cur_cfg['class_name'], match_threshold=cur_cfg['matched_threshold'], unmatch_threshold=cur_cfg['unmatched_threshold'])
            anchor_generators.append(anchor_generator)
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)()
        self.target_assigner = TargetAssigner(anchor_generators=anchor_generators, pos_fraction=anchor_target_cfg.SAMPLE_POS_FRACTION, sample_size=anchor_target_cfg.SAMPLE_SIZE, region_similarity_fn_name=anchor_target_cfg.REGION_SIMILARITY_FN, box_coder=self.box_coder)
        self.num_anchors_per_location = self.target_assigner.num_anchors_per_location
        self.box_code_size = self.box_coder.code_size
        feature_map_size = grid_size[:2] // anchor_target_cfg.DOWNSAMPLED_FACTOR
        feature_map_size = [*feature_map_size, 1][::-1]
        ret = self.target_assigner.generate_anchors(feature_map_size)
        anchors_dict = self.target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret['anchors'].reshape([-1, 7])
        self.anchor_cache = {'anchors': anchors, 'anchors_dict': anchors_dict}
        self.forward_ret_dict = None
        self.build_losses(cfg.MODEL.LOSSES)

    def build_losses(self, losses_cfg):
        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']
        rpn_code_weights = code_weights[3:7] if losses_cfg.RPN_REG_LOSS == 'bin-based' else code_weights
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=rpn_code_weights)
        self.dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()

    def assign_targets(self, gt_boxes):
        """
        :param gt_boxes: (B, N, 8)
        :return:
        """
        gt_boxes = gt_boxes.cpu().numpy()
        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, (7)]
        gt_boxes = gt_boxes[:, :, :7]
        targets_dict_list = []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1]
            cur_gt_names = np.array(cfg.CLASS_NAMES)[cur_gt_classes.astype(np.int32) - 1]
            cur_target_dict = self.target_assigner.assign_v2(anchors_dict=self.anchor_cache['anchors_dict'], gt_boxes=cur_gt, gt_classes=cur_gt_classes, gt_names=cur_gt_names)
            targets_dict_list.append(cur_target_dict)
        targets_dict = {}
        for key in targets_dict_list[0].keys():
            val = np.stack([x[key] for x in targets_dict_list], axis=0)
            targets_dict[key] = val
        return targets_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[(...), dim:dim + 1]) * torch.cos(boxes2[(...), dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[(...), dim:dim + 1]) * torch.sin(boxes2[(...), dim:dim + 1])
        boxes1 = torch.cat([boxes1[(...), :dim], rad_pred_encoding, boxes1[(...), dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[(...), :dim], rad_tg_encoding, boxes2[(...), dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period_torch(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype, device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict
        anchors = forward_ret_dict['anchors']
        box_preds = forward_ret_dict['box_preds']
        cls_preds = forward_ret_dict['cls_preds']
        box_dir_cls_preds = forward_ret_dict['dir_cls_preds']
        box_cls_labels = forward_ret_dict['box_cls_labels']
        box_reg_targets = forward_ret_dict['box_reg_targets']
        batch_size = int(box_preds.shape[0])
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        num_class = self.num_class
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), num_class + 1, dtype=box_preds.dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        if cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros']:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
            one_hot_targets = one_hot_targets[(...), 1:]
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
        loss_weights_dict = loss_cfgs.LOSS_WEIGHTS
        cls_loss = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        cls_loss_reduced = cls_loss.sum() / batch_size
        cls_loss_reduced = cls_loss_reduced * loss_weights_dict['rpn_cls_weight']
        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location)
        if loss_cfgs.RPN_REG_LOSS == 'smooth-l1':
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
            loc_loss_reduced = loc_loss.sum() / batch_size
        else:
            raise NotImplementedError
        loc_loss_reduced = loc_loss_reduced * loss_weights_dict['rpn_loc_weight']
        rpn_loss = loc_loss_reduced + cls_loss_reduced
        tb_dict = {'rpn_loss_loc': loc_loss_reduced.item(), 'rpn_loss_cls': cls_loss_reduced.item()}
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(anchors, box_reg_targets, dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS['dir_offset'], num_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins'])
            dir_logits = box_dir_cls_preds.view(batch_size, -1, cfg.MODEL.RPN.RPN_HEAD.ARGS['num_direction_bins'])
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['rpn_dir_weight']
            rpn_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


class Empty(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


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


class RPNV2(AnchorHead):

    def __init__(self, num_class, args, anchor_target_cfg, grid_size, **kwargs):
        super().__init__(grid_size=grid_size, anchor_target_cfg=anchor_target_cfg)
        self._use_direction_classifier = args['use_direction_classifier']
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])
        if args['use_norm']:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)
        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []
        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(nn.ZeroPad2d(1), Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]), BatchNorm2d(args['num_filters'][i]), nn.ReLU())
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(ConvTranspose2d(args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i], stride=args['upsample_strides'][i]), BatchNorm2d(args['num_upsample_filters'][i]), nn.ReLU())
            deblocks.append(deblock)
        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']
        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]), BatchNorm2d(c_in), nn.ReLU())
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if args['encode_background_as_zeros']:
            num_cls = self.num_anchors_per_location * num_class
        else:
            num_cls = self.num_anchors_per_location * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = self.num_anchors_per_location * self.box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if args['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(c_in, self.num_anchors_per_location * args['num_direction_bins'], 1)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in, bev=None, **kwargs):
        ups = []
        x = x_in
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            ups.append(self.deblocks[i](x))
        if self._concat_input:
            ups.append(x_in)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        ret_dict['spatial_features_last'] = x
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({'box_preds': box_preds, 'cls_preds': cls_preds})
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds
        ret_dict['anchors'] = torch.from_numpy(self.anchor_cache['anchors'])
        if self.training:
            targets_dict = self.assign_targets(gt_boxes=kwargs['gt_boxes'])
            ret_dict.update({'box_cls_labels': torch.from_numpy(targets_dict['labels']), 'box_reg_targets': torch.from_numpy(targets_dict['bbox_targets']), 'reg_src_targets': torch.from_numpy(targets_dict['bbox_src_targets']), 'reg_weights': torch.from_numpy(targets_dict['bbox_outside_weights'])})
        self.forward_ret_dict = ret_dict
        return ret_dict


bbox_head_modules = {'RPNV2': RPNV2}


def get_maxiou3d_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    :param rois: (N, 7)
    :param roi_labels: (N)
    :param gt_boxes: (N, 8)
    :return:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = roi_labels == k
        gt_mask = gt_labels == k
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]
    return max_overlaps, gt_assignment


def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, roi_sampler_cfg):
    if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
        hard_bg_rois_num = int(bg_rois_per_this_image * roi_sampler_cfg.HARD_BG_RATIO)
        easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        hard_bg_inds = hard_bg_inds[rand_idx]
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        easy_bg_inds = easy_bg_inds[rand_idx]
        bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
    elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
        hard_bg_rois_num = bg_rois_per_this_image
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        bg_inds = hard_bg_inds[rand_idx]
    elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
        easy_bg_rois_num = bg_rois_per_this_image
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        bg_inds = easy_bg_inds[rand_idx]
    else:
        raise NotImplementedError
    return bg_inds


def sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d, roi_raw_scores, roi_labels, roi_sampler_cfg):
    """
    :param roi_boxes3d: (B, M, 7 + ?) [x, y, z, w, l, h, ry] in LiDAR coords
    :param gt_boxes3d: (B, N, 7 + ? + 1) [x, y, z, w, l, h, ry, class]
    :param roi_raw_scores: (B, N)
    :param roi_labels: (B, N)
    :return
        batch_rois: (B, N, 7)
        batch_gt_of_rois: (B, N, 7 + 1)
        batch_roi_iou: (B, N)
    """
    batch_size = roi_boxes3d.size(0)
    fg_rois_per_image = int(np.round(roi_sampler_cfg.FG_RATIO * roi_sampler_cfg.ROI_PER_IMAGE))
    code_size = roi_boxes3d.shape[-1]
    batch_rois = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE, code_size).zero_()
    batch_gt_of_rois = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1).zero_()
    batch_roi_iou = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE).zero_()
    batch_roi_raw_scores = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE).zero_()
    batch_roi_labels = gt_boxes3d.new(batch_size, roi_sampler_cfg.ROI_PER_IMAGE).zero_().long()
    for idx in range(batch_size):
        cur_roi, cur_gt, cur_roi_raw_scores, cur_roi_labels = roi_boxes3d[idx], gt_boxes3d[idx], roi_raw_scores[idx], roi_labels[idx]
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if len(cfg.CLASS_NAMES) == 1:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
        else:
            cur_gt_labels = cur_gt[:, (-1)].long()
            max_overlaps, gt_assignment = get_maxiou3d_with_same_class(cur_roi, cur_roi_labels, cur_gt[:, 0:7], cur_gt_labels)
        fg_thresh = min(roi_sampler_cfg.REG_FG_THRESH, roi_sampler_cfg.CLS_FG_THRESH)
        fg_inds = torch.nonzero(max_overlaps >= fg_thresh).view(-1)
        easy_bg_inds = torch.nonzero(max_overlaps < roi_sampler_cfg.CLS_BG_THRESH_LO).view(-1)
        hard_bg_inds = torch.nonzero((max_overlaps < roi_sampler_cfg.REG_FG_THRESH) & (max_overlaps >= roi_sampler_cfg.CLS_BG_THRESH_LO)).view(-1)
        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
        if fg_num_rois > 0 and bg_num_rois > 0:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
            bg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, roi_sampler_cfg)
        elif fg_num_rois > 0 and bg_num_rois == 0:
            rand_num = np.floor(np.random.rand(roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            bg_rois_per_this_image = roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, roi_sampler_cfg)
            fg_rois_per_this_image = 0
        else:
            None
            None
            raise NotImplementedError
        roi_list, roi_iou_list, roi_gt_list, roi_score_list, roi_labels_list = [], [], [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois = cur_roi[fg_inds]
            gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
            fg_iou3d = max_overlaps[fg_inds]
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)
            roi_score_list.append(cur_roi_raw_scores[fg_inds])
            roi_labels_list.append(cur_roi_labels[fg_inds])
        if bg_rois_per_this_image > 0:
            bg_rois = cur_roi[bg_inds]
            gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
            bg_iou3d = max_overlaps[bg_inds]
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)
            roi_score_list.append(cur_roi_raw_scores[bg_inds])
            roi_labels_list.append(cur_roi_labels[bg_inds])
        rois = torch.cat(roi_list, dim=0)
        iou_of_rois = torch.cat(roi_iou_list, dim=0)
        gt_of_rois = torch.cat(roi_gt_list, dim=0)
        cur_roi_raw_scores = torch.cat(roi_score_list, dim=0)
        cur_roi_labels = torch.cat(roi_labels_list, dim=0)
        batch_rois[idx] = rois
        batch_gt_of_rois[idx] = gt_of_rois
        batch_roi_iou[idx] = iou_of_rois
        batch_roi_raw_scores[idx] = cur_roi_raw_scores
        batch_roi_labels[idx] = cur_roi_labels
    return batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores, batch_roi_labels


def proposal_target_layer(input_dict, roi_sampler_cfg):
    rois = input_dict['rois']
    roi_raw_scores = input_dict['roi_raw_scores']
    roi_labels = input_dict['roi_labels']
    gt_boxes = input_dict['gt_boxes']
    batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores, batch_roi_labels = sample_rois_for_rcnn(rois, gt_boxes, roi_raw_scores, roi_labels, roi_sampler_cfg)
    reg_valid_mask = (batch_roi_iou > roi_sampler_cfg.REG_FG_THRESH).long()
    if roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
        batch_cls_label = (batch_roi_iou > roi_sampler_cfg.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > roi_sampler_cfg.CLS_BG_THRESH) & (batch_roi_iou < roi_sampler_cfg.CLS_FG_THRESH)
        batch_cls_label[invalid_mask > 0] = -1
    elif roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
        fg_mask = batch_roi_iou > roi_sampler_cfg.CLS_FG_THRESH
        bg_mask = batch_roi_iou < roi_sampler_cfg.CLS_BG_THRESH
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        batch_cls_label = (fg_mask > 0).float()
        batch_cls_label[interval_mask] = batch_roi_iou[interval_mask] * 2 - 0.5
    else:
        raise NotImplementedError
    output_dict = {'rcnn_cls_labels': batch_cls_label.view(-1), 'reg_valid_mask': reg_valid_mask.view(-1), 'gt_of_rois': batch_gt_of_rois, 'gt_iou': batch_roi_iou, 'rois': batch_rois, 'roi_raw_scores': batch_roi_raw_scores, 'roi_labels': batch_roi_labels}
    return output_dict


class RCNNHead(nn.Module):

    def __init__(self, rcnn_target_config):
        super().__init__()
        self.forward_ret_dict = None
        self.rcnn_target_config = rcnn_target_config
        self.box_coder = getattr(box_coder_utils, rcnn_target_config.BOX_CODER)()
        losses_cfg = cfg.MODEL.LOSSES
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=code_weights)

    def assign_targets(self, batch_size, rcnn_dict):
        with torch.no_grad():
            targets_dict = proposal_target_layer(rcnn_dict, roi_sampler_cfg=self.rcnn_target_config)
        rois = targets_dict['rois']
        gt_of_rois = targets_dict['gt_of_rois']
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, (6)] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, (6)] = gt_of_rois[:, :, (6)] - roi_ry
        for k in range(batch_size):
            gt_of_rois[k] = common_utils.rotate_pc_along_z_torch(gt_of_rois[k].unsqueeze(dim=1), -(roi_ry[k] + np.pi / 2)).squeeze(dim=1)
        ry_label = gt_of_rois[:, :, (6)] % (2 * np.pi)
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)
        flag = ry_label > np.pi
        ry_label[flag] = ry_label[flag] - np.pi * 2
        ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, (6)] = ry_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        LOSS_WEIGHTS = loss_cfgs.LOSS_WEIGHTS
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict
        code_size = self.box_coder.code_size
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].float().view(-1)
        reg_valid_mask = forward_ret_dict['reg_valid_mask']
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][(...), 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][(...), 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = rcnn_cls_labels.shape[0]
        rcnn_loss = 0
        if loss_cfgs.RCNN_CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels, reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = rcnn_loss_cls * LOSS_WEIGHTS['rcnn_cls_weight']
        else:
            raise NotImplementedError
        rcnn_loss += rcnn_loss_cls
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()
        if fg_sum == 0:
            temp_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[0].unsqueeze(dim=0)
            faked_reg_target = temp_rcnn_reg.detach()
            rcnn_loss_reg = self.reg_loss_func(temp_rcnn_reg, faked_reg_target)
            rcnn_loss_reg = rcnn_loss_reg.sum() / 1.0
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        else:
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
            if loss_cfgs.RCNN_REG_LOSS == 'smooth-l1':
                rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, (6)] = 0
                reg_targets = self.box_coder.encode_torch(gt_boxes3d_ct.view(rcnn_batch_size, code_size)[fg_mask], rois_anchor[fg_mask])
                rcnn_loss_reg = self.reg_loss_func(rcnn_reg.view(rcnn_batch_size, -1)[fg_mask].unsqueeze(dim=0), reg_targets.unsqueeze(dim=0))
                rcnn_loss_reg = rcnn_loss_reg.sum() / max(fg_sum, 0)
                rcnn_loss_reg = rcnn_loss_reg * LOSS_WEIGHTS['rcnn_reg_weight']
                tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
                if loss_cfgs.CORNER_LOSS_REGULARIZATION:
                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, (6)].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.box_coder.decode_torch(fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors).view(-1, code_size)
                    rcnn_boxes3d = common_utils.rotate_pc_along_z_torch(rcnn_boxes3d.unsqueeze(dim=1), roi_ry + np.pi / 2).squeeze(dim=1)
                    rcnn_boxes3d[:, 0:3] += roi_xyz
                    loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                    loss_corner = loss_corner.mean()
                    loss_corner = loss_corner * LOSS_WEIGHTS['rcnn_corner_weight']
                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner
            else:
                raise NotImplementedError
        rcnn_loss += rcnn_loss_reg
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict


class FCRCNN(RCNNHead):

    def __init__(self, num_point_features, rcnn_cfg, **kwargs):
        super().__init__(rcnn_target_config=cfg.MODEL.RCNN.TARGET_CONFIG)
        self.SA_modules = nn.ModuleList()
        block = self.post_act_block
        c0 = rcnn_cfg.SHARED_FC[0] // 2
        self.conv_part = spconv.SparseSequential(block(4, 64, 3, padding=1, indice_key='rcnn_subm1'), block(64, c0, 3, padding=1, indice_key='rcnn_subm1_1'))
        self.conv_rpn = spconv.SparseSequential(block(num_point_features, 64, 3, padding=1, indice_key='rcnn_subm2'), block(64, c0, 3, padding=1, indice_key='rcnn_subm1_2'))
        shared_fc_list = []
        pool_size = rcnn_cfg.ROI_AWARE_POOL_SIZE
        pre_channel = rcnn_cfg.SHARED_FC[0] * pool_size * pool_size * pool_size
        for k in range(1, rcnn_cfg.SHARED_FC.__len__()):
            shared_fc_list.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.SHARED_FC[k], bn=True))
            pre_channel = rcnn_cfg.SHARED_FC[k]
            if k != rcnn_cfg.SHARED_FC.__len__() - 1 and rcnn_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(rcnn_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        channel_in = rcnn_cfg.SHARED_FC[-1]
        cls_channel = 1
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.CLS_FC[k], bn=True))
            pre_channel = rcnn_cfg.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.REG_FC[k], bn=True))
            pre_channel = rcnn_cfg.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, self.box_coder.code_size, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)
        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(out_size=rcnn_cfg.ROI_AWARE_POOL_SIZE, max_pts_each_voxel=128)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'subm':
            m = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key), nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01), nn.ReLU())
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key), nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01), nn.ReLU())
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False), nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01), nn.ReLU())
        else:
            raise NotImplementedError
        return m

    def roiaware_pool(self, batch_rois, rcnn_dict):
        """
        :param batch_rois: (B, N, 7 + ?) [x, y, z, w, l, h, rz] in LiDAR coords
        :param rcnn_dict:
        :return:
        """
        voxel_centers = rcnn_dict['voxel_centers']
        rpn_features = rcnn_dict['rpn_seg_features']
        coords = rcnn_dict['coordinates']
        rpn_seg_score = rcnn_dict['rpn_seg_scores'].detach()
        rpn_seg_mask = rpn_seg_score > cfg.MODEL.RPN.BACKBONE.SEG_MASK_SCORE_THRESH
        rpn_part_offsets = rcnn_dict['rpn_part_offsets'].clone().detach()
        rpn_part_offsets[rpn_seg_mask == 0] = 0
        part_features = torch.cat((rpn_part_offsets, rpn_seg_score.view(-1, 1)), dim=1)
        batch_size = batch_rois.shape[0]
        pooled_part_features_list, pooled_rpn_features_list = [], []
        for bs_idx in range(batch_size):
            bs_mask = coords[:, (0)] == bs_idx
            cur_voxel_centers = voxel_centers[bs_mask]
            cur_part_features = part_features[bs_mask]
            cur_rpn_features = rpn_features[bs_mask]
            cur_roi = batch_rois[bs_idx][:, 0:7].contiguous()
            pooled_part_features = self.roiaware_pool3d_layer.forward(cur_roi, cur_voxel_centers, cur_part_features, pool_method='avg')
            pooled_rpn_features = self.roiaware_pool3d_layer.forward(cur_roi, cur_voxel_centers, cur_rpn_features, pool_method='max')
            pooled_part_features_list.append(pooled_part_features)
            pooled_rpn_features_list.append(pooled_rpn_features)
        pooled_part_features = torch.cat(pooled_part_features_list, dim=0)
        pooled_rpn_features = torch.cat(pooled_rpn_features_list, dim=0)
        return pooled_part_features, pooled_rpn_features

    def _break_up_pc(self, pc):
        xyz = pc[(...), 0:3].contiguous()
        features = pc[(...), 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def fake_sparse_idx(self, sparse_idx, batch_size_rcnn):
        None
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def forward(self, rcnn_dict):
        """
        :param input_data: input dict
        :return:
        """
        rois = rcnn_dict['rois']
        batch_size = rois.shape[0]
        if self.training:
            targets_dict = self.assign_targets(batch_size, rcnn_dict)
            rois = targets_dict['rois']
            rcnn_dict['roi_raw_scores'] = targets_dict['roi_raw_scores']
            rcnn_dict['roi_labels'] = targets_dict['roi_labels']
        pooled_part_features, pooled_rpn_features = self.roiaware_pool(rois, rcnn_dict)
        batch_size_rcnn = pooled_part_features.shape[0]
        sparse_shape = np.array(pooled_part_features.shape[1:4], dtype=np.int32)
        sparse_idx = pooled_part_features.sum(dim=-1).nonzero()
        if sparse_idx.shape[0] < 3:
            sparse_idx = self.fake_sparse_idx(sparse_idx, batch_size_rcnn)
            if self.training:
                targets_dict['rcnn_cls_labels'].fill_(-1)
                targets_dict['reg_valid_mask'].fill_(-1)
        part_features = pooled_part_features[sparse_idx[:, (0)], sparse_idx[:, (1)], sparse_idx[:, (2)], sparse_idx[:, (3)]]
        rpn_features = pooled_rpn_features[sparse_idx[:, (0)], sparse_idx[:, (1)], sparse_idx[:, (2)], sparse_idx[:, (3)]]
        coords = sparse_idx.int()
        part_features = spconv.SparseConvTensor(part_features, coords, sparse_shape, batch_size_rcnn)
        rpn_features = spconv.SparseConvTensor(rpn_features, coords, sparse_shape, batch_size_rcnn)
        x_part = self.conv_part(part_features)
        x_rpn = self.conv_rpn(rpn_features)
        merged_feature = torch.cat((x_rpn.features, x_part.features), dim=1)
        shared_feature = spconv.SparseConvTensor(merged_feature, coords, sparse_shape, batch_size_rcnn)
        shared_feature = shared_feature.dense().view(batch_size_rcnn, -1, 1)
        shared_feature = self.shared_fc_layer(shared_feature)
        rcnn_cls = self.cls_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_reg = self.reg_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg, 'rois': rois, 'roi_raw_scores': rcnn_dict['roi_raw_scores'], 'roi_labels': rcnn_dict['roi_labels']}
        if self.training:
            ret_dict.update(targets_dict)
        self.forward_ret_dict = ret_dict
        return ret_dict


class SpConvRCNN(RCNNHead):

    def __init__(self, num_point_features, rcnn_cfg, **kwargs):
        super().__init__(rcnn_target_config=cfg.MODEL.RCNN.TARGET_CONFIG)
        self.SA_modules = nn.ModuleList()
        block = self.post_act_block
        self.conv_part = spconv.SparseSequential(block(4, 64, 3, padding=1, indice_key='rcnn_subm1'), block(64, 64, 3, padding=1, indice_key='rcnn_subm1_1'))
        self.conv_rpn = spconv.SparseSequential(block(num_point_features, 64, 3, padding=1, indice_key='rcnn_subm2'), block(64, 64, 3, padding=1, indice_key='rcnn_subm1_2'))
        self.conv_down = spconv.SparseSequential(block(128, 128, 3, padding=1, indice_key='rcnn_subm2'), block(128, 128, 3, padding=1, indice_key='rcnn_subm2'), spconv.SparseMaxPool3d(kernel_size=2, stride=2), block(128, 128, 3, padding=1, indice_key='rcnn_subm3'), block(128, rcnn_cfg.SHARED_FC[0], 3, padding=1, indice_key='rcnn_subm3'))
        shared_fc_list = []
        pool_size = rcnn_cfg.ROI_AWARE_POOL_SIZE // 2
        pre_channel = rcnn_cfg.SHARED_FC[0] * pool_size * pool_size * pool_size
        for k in range(1, rcnn_cfg.SHARED_FC.__len__()):
            shared_fc_list.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.SHARED_FC[k], bn=True))
            pre_channel = rcnn_cfg.SHARED_FC[k]
            if k != rcnn_cfg.SHARED_FC.__len__() - 1 and rcnn_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(rcnn_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        channel_in = rcnn_cfg.SHARED_FC[-1]
        cls_channel = 1
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.CLS_FC[k], bn=True))
            pre_channel = rcnn_cfg.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.REG_FC[k], bn=True))
            pre_channel = rcnn_cfg.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, self.box_coder.code_size, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)
        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(out_size=rcnn_cfg.ROI_AWARE_POOL_SIZE, max_pts_each_voxel=128)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'subm':
            m = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key), nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01), nn.ReLU())
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key), nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01), nn.ReLU())
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False), nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01), nn.ReLU())
        else:
            raise NotImplementedError
        return m

    def roiaware_pool(self, batch_rois, rcnn_dict):
        """
        :param batch_rois: (B, N, 7 + ?) [x, y, z, w, l, h, rz] in LiDAR coords
        :param rcnn_dict:
        :return:
        """
        voxel_centers = rcnn_dict['voxel_centers']
        rpn_features = rcnn_dict['rpn_seg_features']
        coords = rcnn_dict['coordinates']
        rpn_seg_score = rcnn_dict['rpn_seg_scores'].detach()
        rpn_seg_mask = rpn_seg_score > cfg.MODEL.RPN.BACKBONE.SEG_MASK_SCORE_THRESH
        rpn_part_offsets = rcnn_dict['rpn_part_offsets'].clone().detach()
        rpn_part_offsets[rpn_seg_mask == 0] = 0
        part_features = torch.cat((rpn_part_offsets, rpn_seg_score.view(-1, 1)), dim=1)
        batch_size = batch_rois.shape[0]
        pooled_part_features_list, pooled_rpn_features_list = [], []
        for bs_idx in range(batch_size):
            bs_mask = coords[:, (0)] == bs_idx
            cur_voxel_centers = voxel_centers[bs_mask]
            cur_part_features = part_features[bs_mask]
            cur_rpn_features = rpn_features[bs_mask]
            cur_roi = batch_rois[bs_idx][:, 0:7].contiguous()
            pooled_part_features = self.roiaware_pool3d_layer.forward(cur_roi, cur_voxel_centers, cur_part_features, pool_method='avg')
            pooled_rpn_features = self.roiaware_pool3d_layer.forward(cur_roi, cur_voxel_centers, cur_rpn_features, pool_method='max')
            pooled_part_features_list.append(pooled_part_features)
            pooled_rpn_features_list.append(pooled_rpn_features)
        pooled_part_features = torch.cat(pooled_part_features_list, dim=0)
        pooled_rpn_features = torch.cat(pooled_rpn_features_list, dim=0)
        return pooled_part_features, pooled_rpn_features

    def _break_up_pc(self, pc):
        xyz = pc[(...), 0:3].contiguous()
        features = pc[(...), 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def fake_sparse_idx(self, sparse_idx, batch_size_rcnn):
        None
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def forward(self, rcnn_dict):
        """
        :param input_data: input dict
        :return:
        """
        rois = rcnn_dict['rois']
        batch_size = rois.shape[0]
        if self.training:
            targets_dict = self.assign_targets(batch_size, rcnn_dict)
            rois = targets_dict['rois']
            rcnn_dict['roi_raw_scores'] = targets_dict['roi_raw_scores']
            rcnn_dict['roi_labels'] = targets_dict['roi_labels']
        pooled_part_features, pooled_rpn_features = self.roiaware_pool(rois, rcnn_dict)
        batch_size_rcnn = pooled_part_features.shape[0]
        sparse_shape = np.array(pooled_part_features.shape[1:4], dtype=np.int32)
        sparse_idx = pooled_part_features.sum(dim=-1).nonzero()
        if sparse_idx.shape[0] < 3:
            sparse_idx = self.fake_sparse_idx(sparse_idx, batch_size_rcnn)
            if self.training:
                targets_dict['rcnn_cls_labels'].fill_(-1)
                targets_dict['reg_valid_mask'].fill_(-1)
        part_features = pooled_part_features[sparse_idx[:, (0)], sparse_idx[:, (1)], sparse_idx[:, (2)], sparse_idx[:, (3)]]
        rpn_features = pooled_rpn_features[sparse_idx[:, (0)], sparse_idx[:, (1)], sparse_idx[:, (2)], sparse_idx[:, (3)]]
        coords = sparse_idx.int()
        part_features = spconv.SparseConvTensor(part_features, coords, sparse_shape, batch_size_rcnn)
        rpn_features = spconv.SparseConvTensor(rpn_features, coords, sparse_shape, batch_size_rcnn)
        x_part = self.conv_part(part_features)
        x_rpn = self.conv_rpn(rpn_features)
        merged_feature = torch.cat((x_rpn.features, x_part.features), dim=1)
        shared_feature = spconv.SparseConvTensor(merged_feature, coords, sparse_shape, batch_size_rcnn)
        x = self.conv_down(shared_feature)
        shared_feature = x.dense().view(batch_size_rcnn, -1, 1)
        shared_feature = self.shared_fc_layer(shared_feature)
        rcnn_cls = self.cls_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_reg = self.reg_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg, 'rois': rois, 'roi_raw_scores': rcnn_dict['roi_raw_scores'], 'roi_labels': rcnn_dict['roi_labels']}
        if self.training:
            ret_dict.update(targets_dict)
        self.forward_ret_dict = ret_dict
        return ret_dict


rcnn_modules = {'FCRCNN': FCRCNN, 'SpConvRCNN': SpConvRCNN}


class BackBone8x(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = self.post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        ret = {'spatial_features': spatial_features}
        return ret

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False), norm_fn(out_channels), nn.ReLU())
        else:
            raise NotImplementedError
        return m


class PointPillarsScatter(nn.Module):

    def __init__(self, input_channels=64, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """
        super().__init__()
        self.nchannels = input_channels

    def forward(self, voxel_features, coords, batch_size, **kwargs):
        output_shape = kwargs['output_shape']
        nz, ny, nx = output_shape
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(self.nchannels, nz * nx * ny, dtype=voxel_features.dtype, device=voxel_features.device)
            batch_mask = coords[:, (0)] == batch_itt
            this_coords = coords[(batch_mask), :]
            indices = this_coords[:, (1)] * nz + this_coords[:, (2)] * nx + this_coords[:, (3)]
            indices = indices.type(torch.long)
            voxels = voxel_features[(batch_mask), :]
            voxels = voxels.t()
            canvas[:, (indices)] = voxels
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.nchannels * nz, ny, nx)
        return batch_canvas


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key)


class UNetHead(nn.Module):

    def __init__(self, unet_target_cfg):
        super().__init__()
        self.gt_extend_width = unet_target_cfg.GT_EXTEND_WIDTH
        if 'MEAN_SIZE' in unet_target_cfg:
            self.mean_size = unet_target_cfg.MEAN_SIZE
        self.target_generated_on = unet_target_cfg.GENERATED_ON
        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        self.forward_ret_dict = None

    def assign_targets(self, batch_points, gt_boxes, generate_bbox_reg_labels=False):
        """
        :param points: [(N1, 3), (N2, 3), ...]
        :param gt_boxes: (B, M, 8)
        :param gt_classes: (B, M)
        :param gt_names: (B, M)
        :return:
        """
        batch_size = gt_boxes.shape[0]
        cls_labels_list, part_reg_labels_list, bbox_reg_labels_list = [], [], []
        for k in range(batch_size):
            if True or self.target_generated_on == 'head_cpu':
                cur_cls_labels, cur_part_reg_labels, cur_bbox_reg_labels = self.generate_part_targets_cpu(points=batch_points[k], gt_boxes=gt_boxes[k][:, 0:7], gt_classes=gt_boxes[k][:, (7)], generate_bbox_reg_labels=generate_bbox_reg_labels)
            else:
                raise NotImplementedError
            cls_labels_list.append(cur_cls_labels)
            part_reg_labels_list.append(cur_part_reg_labels)
            bbox_reg_labels_list.append(cur_bbox_reg_labels)
        cls_labels = torch.cat(cls_labels_list, dim=0)
        part_reg_labels = torch.cat(part_reg_labels_list, dim=0)
        bbox_reg_labels = torch.cat(bbox_reg_labels_list, dim=0) if generate_bbox_reg_labels else None
        targets_dict = {'seg_labels': cls_labels, 'part_labels': part_reg_labels, 'bbox_reg_labels': bbox_reg_labels}
        return targets_dict

    def generate_part_targets_cpu(self, points, gt_boxes, gt_classes, generate_bbox_reg_labels=False):
        """
        :param voxel_centers: (N, 3) [x, y, z]
        :param gt_boxes: (M, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
        """
        k = gt_boxes.__len__() - 1
        while k > 0 and gt_boxes[k].sum() == 0:
            k -= 1
        gt_boxes = gt_boxes[:k + 1]
        gt_classes = gt_classes[:k + 1]
        extend_gt_boxes = common_utils.enlarge_box3d(gt_boxes, extra_width=self.gt_extend_width)
        cls_labels = torch.zeros(points.shape[0]).int()
        part_reg_labels = torch.zeros((points.shape[0], 3)).float()
        bbox_reg_labels = torch.zeros((points.shape[0], 7)).float() if generate_bbox_reg_labels else None
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points, gt_boxes).long()
        extend_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points, extend_gt_boxes).long()
        for k in range(gt_boxes.shape[0]):
            fg_pt_flag = point_indices[k] > 0
            fg_points = points[fg_pt_flag]
            cls_labels[fg_pt_flag] = gt_classes[k]
            fg_enlarge_flag = extend_point_indices[k] > 0
            ignore_flag = fg_pt_flag ^ fg_enlarge_flag
            cls_labels[ignore_flag] = -1
            transformed_points = fg_points - gt_boxes[(k), 0:3]
            transformed_points = common_utils.rotate_pc_along_z_torch(transformed_points.view(1, -1, 3), -gt_boxes[k, 6])
            part_reg_labels[fg_pt_flag] = transformed_points / gt_boxes[(k), 3:6] + torch.tensor([0.5, 0.5, 0]).float()
            if generate_bbox_reg_labels:
                center3d = gt_boxes[(k), 0:3].clone()
                center3d[2] += gt_boxes[k][5] / 2
                bbox_reg_labels[(fg_pt_flag), 0:3] = center3d - fg_points
                bbox_reg_labels[fg_pt_flag, 6] = gt_boxes[k, 6]
                cur_mean_size = torch.tensor(self.mean_size[cfg.CLASS_NAMES[gt_classes[k] - 1]])
                bbox_reg_labels[(fg_pt_flag), 3:6] = (gt_boxes[(k), 3:6] - cur_mean_size) / cur_mean_size
        return cls_labels, part_reg_labels, bbox_reg_labels

    def get_loss(self, forward_ret_dict=None):
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict
        tb_dict = {}
        u_seg_preds = forward_ret_dict['u_seg_preds'].squeeze(dim=-1)
        u_reg_preds = forward_ret_dict['u_reg_preds']
        u_cls_labels, u_reg_labels = forward_ret_dict['seg_labels'], forward_ret_dict['part_labels']
        u_cls_target = (u_cls_labels > 0).float()
        pos_mask = u_cls_labels > 0
        pos = pos_mask.float()
        neg = (u_cls_labels == 0).float()
        u_cls_weights = pos + neg
        pos_normalizer = pos.sum()
        u_cls_weights = u_cls_weights / torch.clamp(pos_normalizer, min=1.0)
        u_loss_cls = self.cls_loss_func(u_seg_preds, u_cls_target, weights=u_cls_weights)
        u_loss_cls_pos = (u_loss_cls * pos).sum()
        u_loss_cls_neg = (u_loss_cls * neg).sum()
        u_loss_cls = u_loss_cls.sum()
        loss_unet = u_loss_cls
        if pos_normalizer > 0:
            u_loss_reg = F.binary_cross_entropy(torch.sigmoid(u_reg_preds[pos_mask]), u_reg_labels[pos_mask])
            loss_unet += u_loss_reg
            tb_dict['rpn_u_loss_reg'] = u_loss_reg.item()
        tb_dict['rpn_loss_u_cls'] = u_loss_cls.item()
        tb_dict['rpn_loss_u_cls_pos'] = u_loss_cls_pos.item()
        tb_dict['rpn_loss_u_cls_neg'] = u_loss_cls_neg.item()
        tb_dict['rpn_loss_unet'] = loss_unet.item()
        tb_dict['rpn_pos_num'] = pos_normalizer.item()
        return loss_unet, tb_dict


class UNetV0(UNetHead):

    def __init__(self, input_channels, **kwargs):
        super().__init__(unet_target_cfg=cfg.MODEL.RPN.BACKBONE.TARGET_CONFIG)
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = self.post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')
        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.seg_cls_layer = nn.Linear(16, 1, bias=True)
        self.seg_reg_layer = nn.Linear(16, 3, bias=True)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        :param x: x.features (N, C1)
        :param out_channels: C2
        :return:
        """
        features = x.features
        n, in_channels = features.shape
        assert in_channels % out_channels == 0 and in_channels >= out_channels
        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False), norm_fn(out_channels), nn.ReLU())
        else:
            raise NotImplementedError
        return m

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        ret_dict = {'spatial_features': spatial_features}
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
        seg_features = x_up1.features
        seg_cls_preds = self.seg_cls_layer(seg_features)
        seg_reg_preds = self.seg_reg_layer(seg_features)
        ret_dict.update({'u_seg_preds': seg_cls_preds, 'u_reg_preds': seg_reg_preds, 'seg_features': seg_features})
        if self.training:
            if self.target_generated_on == 'dataset':
                targets_dict = {'seg_labels': kwargs['seg_labels'], 'part_labels': kwargs['part_labels'], 'bbox_reg_labels': kwargs.get('bbox_reg_labels', None)}
            else:
                batch_size = x_up1.batch_size
                bs_idx, coords = x_up1.indices[:, (0)].cpu(), x_up1.indices[:, 1:].cpu()
                voxel_size = torch.tensor(cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE)
                pc_range = torch.tensor(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
                voxel_centers = (coords[:, ([2, 1, 0])].float() + 0.5) * voxel_size + pc_range[0:3]
                batch_points = [voxel_centers[bs_idx == k] for k in range(batch_size)]
                targets_dict = self.assign_targets(batch_points=batch_points, gt_boxes=kwargs['gt_boxes'].cpu())
            ret_dict['seg_labels'] = targets_dict['seg_labels']
            ret_dict['part_labels'] = targets_dict['part_labels']
            ret_dict['bbox_reg_labels'] = targets_dict.get('bbox_reg_labels', None)
        self.forward_ret_dict = ret_dict
        return ret_dict


class UNetV2(UNetHead):

    def __init__(self, input_channels, **kwargs):
        super().__init__(unet_target_cfg=cfg.MODEL.RPN.BACKBONE.TARGET_CONFIG)
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = self.post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')
        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.seg_cls_layer = nn.Linear(16, 1, bias=True)
        self.seg_reg_layer = nn.Linear(16, 3, bias=True)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        :param x: x.features (N, C1)
        :param out_channels: C2
        :return:
        """
        features = x.features
        n, in_channels = features.shape
        assert in_channels % out_channels == 0 and in_channels >= out_channels
        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key), norm_fn(out_channels), nn.ReLU())
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False), norm_fn(out_channels), nn.ReLU())
        else:
            raise NotImplementedError
        return m

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        ret_dict = {'spatial_features': spatial_features}
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
        seg_features = x_up1.features
        seg_cls_preds = self.seg_cls_layer(seg_features)
        seg_reg_preds = self.seg_reg_layer(seg_features)
        ret_dict.update({'u_seg_preds': seg_cls_preds, 'u_reg_preds': seg_reg_preds, 'seg_features': seg_features})
        if self.training:
            if self.target_generated_on == 'dataset':
                targets_dict = {'seg_labels': kwargs['seg_labels'], 'part_labels': kwargs['part_labels'], 'bbox_reg_labels': kwargs.get('bbox_reg_labels', None)}
            else:
                batch_size = x_up1.batch_size
                bs_idx, coords = x_up1.indices[:, (0)].cpu(), x_up1.indices[:, 1:].cpu()
                voxel_size = torch.tensor(cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE)
                pc_range = torch.tensor(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
                voxel_centers = (coords[:, ([2, 1, 0])].float() + 0.5) * voxel_size + pc_range[0:3]
                batch_points = [voxel_centers[bs_idx == k] for k in range(batch_size)]
                targets_dict2 = self.assign_targets(batch_points=batch_points, gt_boxes=kwargs['gt_boxes'].cpu())
            ret_dict['seg_labels'] = targets_dict['seg_labels']
            ret_dict['part_labels'] = targets_dict['part_labels']
            ret_dict['bbox_reg_labels'] = targets_dict.get('bbox_reg_labels', None)
        self.forward_ret_dict = ret_dict
        return ret_dict


rpn_modules = {'UNetV2': UNetV2, 'UNetV0': UNetV0, 'BackBone8x': BackBone8x, 'PointPillarsScatter': PointPillarsScatter}


class VoxelFeatureExtractor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError


class MeanVoxelFeatureExtractor(VoxelFeatureExtractor):

    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        return cfg.DATA_CONFIG.NUM_POINT_FEATURES['use']

    def forward(self, features, num_voxels, **kwargs):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param kwargs:
        :return:
        """
        points_mean = features[:, :, :].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        if use_norm:
            self.linear = nn.Linear(in_channels, self.units, bias=False)
            self.norm = nn.BatchNorm1d(self.units, eps=0.001, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, self.units, bias=True)
            self.norm = Empty(self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        total_points, voxel_points, channels = x.shape
        x = self.norm(x.view(-1, channels)).view(total_points, voxel_points, channels)
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


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


class PillarFeatureNetOld2(VoxelFeatureExtractor):

    def __init__(self, num_input_features=4, use_norm=True, num_filters=(64,), with_distance=False, voxel_size=(0.2, 0.2, 4), pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetOld2'
        assert len(num_filters) > 0
        num_input_features += 6
        if with_distance:
            num_input_features += 1
        self.with_distance = with_distance
        self.num_filters = num_filters
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, features, num_voxels, coords):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param coors:
        :return:
        """
        dtype = features.dtype
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, (0)] = features[:, :, (0)] - (coords[:, (3)].unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, (1)] = features[:, :, (1)] - (coords[:, (2)].unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, (2)] = features[:, :, (2)] - (coords[:, (1)].unsqueeze(1) * self.vz + self.z_offset)
        features_ls = [features, f_cluster, f_center]
        if self.with_distance:
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


vfe_modules = {'MeanVoxelFeatureExtractor': MeanVoxelFeatureExtractor, 'PillarFeatureNetOld2': PillarFeatureNetOld2}


class Detector3D(nn.Module):

    def __init__(self, num_class, dataset):
        super().__init__()
        self.num_class = num_class
        self.dataset = dataset
        self.grid_size = dataset.voxel_generator.grid_size
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.vfe = self.rpn_net = self.rpn_head = self.rcnn_net = None

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build_networks(self, model_cfg):
        vfe_cfg = model_cfg.VFE
        self.vfe = vfe_modules[vfe_cfg.NAME](num_input_features=cfg.DATA_CONFIG.NUM_POINT_FEATURES['use'], voxel_size=cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE, pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE, **vfe_cfg.ARGS)
        voxel_feature_num = self.vfe.get_output_feature_dim()
        rpn_cfg = model_cfg.RPN
        self.rpn_net = rpn_modules[rpn_cfg.BACKBONE.NAME](input_channels=voxel_feature_num, **rpn_cfg.BACKBONE.ARGS)
        rpn_head_cfg = model_cfg.RPN.RPN_HEAD
        self.rpn_head = bbox_head_modules[rpn_head_cfg.NAME](num_class=self.num_class, args=rpn_head_cfg.ARGS, grid_size=self.grid_size, anchor_target_cfg=rpn_head_cfg.TARGET_CONFIG)
        rcnn_cfg = model_cfg.RCNN
        if rcnn_cfg.ENABLED:
            self.rcnn_net = rcnn_modules[rcnn_cfg.NAME](num_point_features=cfg.MODEL.RCNN.NUM_POINT_FEATURES, rcnn_cfg=rcnn_cfg)

    def update_global_step(self):
        self.global_step += 1

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            pass

    def forward(self, input_dict):
        raise NotImplementedError

    def predict_boxes(self, rpn_ret_dict, rcnn_ret_dict, input_dict):
        batch_size = input_dict['batch_size']
        if rcnn_ret_dict is None:
            batch_anchors = rpn_ret_dict['anchors'].view(1, -1, rpn_ret_dict['anchors'].shape[-1]).repeat(batch_size, 1, 1)
            num_anchors = batch_anchors.shape[1]
            batch_cls_preds = rpn_ret_dict['rpn_cls_preds'].view(batch_size, num_anchors, -1).float()
            batch_box_preds = self.rpn_head.box_coder.decode_with_head_direction_torch(box_preds=rpn_ret_dict['rpn_box_preds'].view(batch_size, num_anchors, -1), anchors=batch_anchors, dir_cls_preds=rpn_ret_dict.get('rpn_dir_cls_preds', None), num_dir_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('num_direction_bins', None), dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_offset', None), dir_limit_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_limit_offset', None), use_binary_dir_classifier=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('use_binary_dir_classifier', False))
        else:
            batch_rois = rcnn_ret_dict['rois']
            code_size = self.rcnn_net.box_coder.code_size
            batch_cls_preds = rcnn_ret_dict['rcnn_cls'].view(batch_size, -1)
            if cfg.MODEL.LOSSES.RCNN_REG_LOSS == 'smooth-l1':
                roi_ry = batch_rois[:, :, (6)].view(-1)
                roi_xyz = batch_rois[:, :, 0:3].view(-1, 3)
                local_rois = batch_rois.clone().detach()
                local_rois[:, :, 0:3] = 0
                rcnn_boxes3d = self.rcnn_net.box_coder.decode_torch(rcnn_ret_dict['rcnn_reg'].view(local_rois.shape[0], -1, code_size), local_rois).view(-1, code_size)
                rcnn_boxes3d = common_utils.rotate_pc_along_z_torch(rcnn_boxes3d.unsqueeze(dim=1), roi_ry + np.pi / 2).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz
                batch_box_preds = rcnn_boxes3d.view(batch_size, -1, code_size)
            else:
                raise NotImplementedError
        pred_dicts, recall_dicts = self.post_processing(batch_cls_preds, batch_box_preds, rcnn_ret_dict, input_dict)
        return pred_dicts, recall_dicts

    def post_processing(self, batch_cls_preds, batch_box_preds, rcnn_ret_dict, input_dict):
        recall_dict = {'gt': 0}
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            recall_dict['roi_%s' % str(cur_thresh)] = 0
            recall_dict['rcnn_%s' % str(cur_thresh)] = 0
        pred_dicts = []
        batch_size = batch_cls_preds.shape[0]
        batch_index = np.arange(batch_size)
        batch_gt_boxes = input_dict.get('gt_boxes', None)
        for index, cls_preds, box_preds in zip(batch_index, batch_cls_preds, batch_box_preds):
            if not cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros'] and rcnn_ret_dict is None:
                cls_preds = cls_preds[(...), 1:]
            normalized_scores = torch.sigmoid(cls_preds)
            if rcnn_ret_dict is not None and batch_gt_boxes is not None:
                self.generate_recall_record(box_preds, rcnn_ret_dict['rois'][index], batch_gt_boxes[index], recall_dict, thresh_list=cfg.MODEL.TEST.RECALL_THRESH_LIST)
            if cfg.MODEL.TEST.MULTI_CLASSES_NMS:
                selected, final_labels = self.multi_classes_nms(rank_scores=cls_preds, normalized_scores=normalized_scores, box_preds=box_preds, score_thresh=cfg.MODEL.TEST.SCORE_THRESH, nms_thresh=cfg.MODEL.TEST.NMS_THRESH, nms_type=cfg.MODEL.TEST.NMS_TYPE)
                final_boxes = box_preds[selected]
                final_scores = cls_preds[selected] if cfg.MODEL.TEST.USE_RAW_SCORE else normalized_scores[selected]
            else:
                if len(cls_preds.shape) > 1 and cls_preds.shape[1] > 1:
                    rank_scores, class_labels = torch.max(cls_preds, dim=-1)
                    normalized_scores = torch.sigmoid(rank_scores)
                    class_labels = class_labels + 1
                else:
                    if rcnn_ret_dict is not None:
                        class_labels = rcnn_ret_dict['roi_labels'][index]
                    else:
                        class_labels = cls_preds.new_ones(cls_preds.shape[0])
                    rank_scores = cls_preds.view(-1)
                    normalized_scores = normalized_scores.view(-1)
                selected = self.class_agnostic_nms(rank_scores=rank_scores, normalized_scores=normalized_scores, box_preds=box_preds, score_thresh=cfg.MODEL.TEST.SCORE_THRESH, nms_thresh=cfg.MODEL.TEST.NMS_THRESH, nms_type=cfg.MODEL.TEST.NMS_TYPE)
                final_labels = class_labels[selected]
                final_scores = rank_scores[selected] if cfg.MODEL.TEST.USE_RAW_SCORE else normalized_scores[selected]
                final_boxes = box_preds[selected]
            record_dict = {'boxes': final_boxes, 'scores': final_scores, 'labels': final_labels}
            if rcnn_ret_dict is not None:
                record_dict['roi_raw_scores'] = rcnn_ret_dict['roi_raw_scores'][index][selected]
                record_dict['rois'] = rcnn_ret_dict['rois'][index][selected]
                mask = record_dict['rois'][:, 3:6].sum(dim=1) > 0
                if mask.sum() != record_dict['rois'].shape[0]:
                    common_utils.dict_select(record_dict, mask)
            cur_pred_dict = self.dataset.generate_prediction_dict(input_dict, index, record_dict)
            pred_dicts.append(cur_pred_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def multi_classes_nms(rank_scores, normalized_scores, box_preds, score_thresh, nms_thresh, nms_type='nms_gpu'):
        """
        :param rank_scores: (N, num_classes)
        :param box_preds: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :param score_thresh: (N) or float
        :param nms_thresh: (N) or float
        :param nms_type:
        :return:
        """
        assert rank_scores.shape[1] == len(cfg.CLASS_NAMES), 'Rank_score shape: %s' % str(rank_scores.shape)
        selected_list = []
        selected_labels = []
        num_classes = rank_scores.shape[1]
        boxes_for_nms = box_utils.boxes3d_to_bevboxes_lidar_torch(box_preds)
        score_thresh = score_thresh if isinstance(score_thresh, list) else [score_thresh for x in range(num_classes)]
        nms_thresh = nms_thresh if isinstance(nms_thresh, list) else [nms_thresh for x in range(num_classes)]
        for k in range(0, num_classes):
            class_scores_keep = normalized_scores[:, (k)] >= score_thresh[k]
            if class_scores_keep.int().sum() > 0:
                original_idxs = class_scores_keep.nonzero().view(-1)
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                cur_rank_scores = rank_scores[class_scores_keep, k]
                cur_selected = getattr(iou3d_nms_utils, nms_type)(cur_boxes_for_nms, cur_rank_scores, nms_thresh[k])
                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])
                selected_labels.append(torch.full([cur_selected.shape[0]], k + 1, dtype=torch.int64, device=box_preds.device))
        selected = torch.cat(selected_list, dim=0) if selected_list.__len__() > 0 else []
        return selected, selected_labels

    @staticmethod
    def class_agnostic_nms(rank_scores, normalized_scores, box_preds, score_thresh, nms_thresh, nms_type='nms_gpu'):
        scores_mask = normalized_scores >= score_thresh
        rank_scores_masked = rank_scores[scores_mask]
        cur_selected = []
        if rank_scores_masked.shape[0] > 0:
            box_preds = box_preds[scores_mask]
            rank_scores_nms, indices = torch.topk(rank_scores_masked, k=min(cfg.MODEL.TEST.NMS_PRE_MAXSIZE_LAST, rank_scores_masked.shape[0]))
            box_preds_nms = box_preds[indices]
            boxes_for_nms = box_utils.boxes3d_to_bevboxes_lidar_torch(box_preds_nms)
            keep_idx = getattr(iou3d_nms_utils, nms_type)(boxes_for_nms, rank_scores_nms, nms_thresh)
            cur_selected = indices[keep_idx[:cfg.MODEL.TEST.NMS_POST_MAXSIZE_LAST]]
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[cur_selected]
        return selected

    def generate_recall_record(self, box_preds, rois, gt_boxes, recall_dict, thresh_list=(0.5, 0.7)):
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if cur_gt.sum() > 0:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois, cur_gt)
            iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds, cur_gt)
            for cur_thresh in thresh_list:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
                recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
            recall_dict['gt'] += cur_gt.shape[0]
            iou3d_rcnn = iou3d_rcnn.max(dim=1)[0]
            gt_iou = iou3d_rcnn
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return gt_iou

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
        if 'version' in checkpoint:
            None
        logger.info('==> Done')
        return it, epoch


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

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name='', instance_norm=False, instance_norm_func=None):
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
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class Conv2d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm2d)


class SharedMLP(nn.Sequential):

    def __init__(self, args: List[int], *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str='', instance_norm: bool=False):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact, instance_norm=instance_norm))


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: int, out_size: int, *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)


class FC(nn.Sequential):

    def __init__(self, in_size: int, out_size: int, *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)
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


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        :param rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate, (x, y, z) is the bottom center of rois
        :param pts: (npoints, 3)
        :param pts_feature: (npoints, C)
        :param out_size: int or tuple, like 7 or (7, 7, 7)
        :param pool_method: 'max' or 'avg'
        :return
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)
        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)
        ctx.roiaware_pool3d_for_backward = pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward
        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)
        return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


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
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Empty,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanVoxelFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PFNLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_sshaoshuai_PCDet(_paritybench_base):
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

