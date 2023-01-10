import sys
_module = sys.modules[__name__]
del sys
build = _module
convert_weights = _module
demo = _module
dump = _module
models = _module
darknet = _module
network_blocks = _module
yolo_fpn = _module
yolo_head = _module
yolo_pafpn = _module
yolox = _module
onnx_inference = _module
openvino_inference = _module
conf = _module
default = _module
yolov3 = _module
yolox_l = _module
yolox_m = _module
yolox_nano = _module
yolox_s = _module
yolox_tiny = _module
yolox_x = _module
nano = _module
yolox_voc_s = _module
ppyoloe_l = _module
ppyoloe_m = _module
ppyoloe_s = _module
ppyoloe_x = _module
hubconf = _module
paddle2torch = _module
ppyoloe = _module
assigner = _module
atss_assigner = _module
tal_assigner = _module
iou2d_calculator = _module
utils = _module
exp = _module
ppyoloe_base = _module
backbone = _module
drop = _module
losses = _module
neck = _module
network_blocks = _module
ppyoloe = _module
ppyoloe_head = _module
setup = _module
tests = _module
test_model_utils = _module
tools = _module
demo = _module
eval = _module
export_onnx = _module
export_torchscript = _module
train = _module
trt = _module
core = _module
launch = _module
trainer = _module
data = _module
data_augment = _module
data_prefetcher = _module
dataloading = _module
datasets = _module
coco = _module
coco_classes = _module
datasets_wrapper = _module
mosaicdetection = _module
voc = _module
voc_classes = _module
operators = _module
samplers = _module
evaluators = _module
coco_evaluator = _module
voc_eval = _module
voc_evaluator = _module
base_exp = _module
yolox_base = _module
layers = _module
fast_coco_eval_api = _module
build = _module
darknet = _module
losses = _module
network_blocks = _module
yolo_fpn = _module
yolo_head = _module
yolo_pafpn = _module
yolox = _module
allreduce_norm = _module
boxes = _module
checkpoint = _module
compat = _module
demo_utils = _module
dist = _module
ema = _module
logger = _module
lr_scheduler = _module
metric = _module
model_utils = _module
setup_env = _module
visualize = _module

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


from collections import OrderedDict


import torch


from typing import Dict


from typing import List


from typing import Tuple


import torch.nn as nn


import torch.distributed as dist


import numpy as np


import torch.nn.functional as F


import random


import math


import re


from torch.utils.cpp_extension import CppExtension


from torch import nn


import time


import warnings


import torch.backends.cudnn as cudnn


from torch.nn.parallel import DistributedDataParallel as DDP


import torch.multiprocessing as mp


from torch.utils.tensorboard import SummaryWriter


import uuid


from torch.utils.data.dataloader import DataLoader as torchDataLoader


from torch.utils.data.dataloader import default_collate


from functools import wraps


from torch.utils.data.dataset import ConcatDataset as torchConcatDataset


from torch.utils.data.dataset import Dataset as torchDataset


import itertools


from typing import Optional


from torch.utils.data.sampler import BatchSampler as torchBatchSampler


from torch.utils.data.sampler import Sampler


from collections import ChainMap


from abc import ABCMeta


from abc import abstractmethod


from torch.nn import Module


import copy


from torch.hub import load_state_dict_from_url


from torch import distributed as dist


import torchvision


import functools


from copy import deepcopy


import inspect


from collections import defaultdict


from collections import deque


from typing import Sequence


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def cast_tensor_type(x, scale=1.0, dtype=None):
    if dtype == 'fp16':
        x = (x / scale).half()
    return x


class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        if self.dtype == 'fp16':
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                overlaps = overlaps.float()
            return overlaps
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(scale={self.scale}, dtype={self.dtype})'
        return repr_str


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], axis=-1)


def check_points_inside_bboxes(points, bboxes, center_radius_tensor=None, eps=1e-09):
    """
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1] Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = points.unsqueeze(0).unsqueeze(0)
    x, y = points.chunk(2, axis=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, axis=-1)
    if center_radius_tensor is not None:
        center_radius_tensor = center_radius_tensor.unsqueeze([0, 1])
        bboxes_cx = (xmin + xmax) / 2.0
        bboxes_cy = (ymin + ymax) / 2.0
        xmin_sampling = bboxes_cx - center_radius_tensor
        ymin_sampling = bboxes_cy - center_radius_tensor
        xmax_sampling = bboxes_cx + center_radius_tensor
        ymax_sampling = bboxes_cy + center_radius_tensor
        xmin = torch.maximum(xmin, xmin_sampling)
        ymin = torch.maximum(ymin, ymin_sampling)
        xmax = torch.minimum(xmax, xmax_sampling)
        ymax = torch.minimum(ymax, ymax_sampling)
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    bbox_ltrb = torch.cat([l, t, r, b], axis=-1)
    return bbox_ltrb.min(axis=-1)[0] > eps


def compute_max_iou_anchor(ious):
    """
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(axis=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute(0, 2, 1)
    return is_max_iou


def compute_max_iou_gt(ious):
    """
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(axis=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.astype(ious.dtype)


class ATSSAssigner(nn.Module):

    def __init__(self, topk=9, num_classes=80, force_gt_matching=False, eps=1e-09):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.iou_similarity = BboxOverlaps2D()

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list, pad_gt_mask):
        pad_gt_mask = pad_gt_mask.repeat(1, 1, self.topk).bool()
        gt2anchor_distances_list = torch.split(gt2anchor_distances, num_anchors_list, dim=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list, num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = torch.topk(distances, self.topk, axis=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask, topk_idxs, torch.zeros_like(topk_idxs))
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
            is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk)
        is_in_topk_list = torch.cat(is_in_topk_list, axis=-1)
        topk_idxs_list = torch.cat(topk_idxs_list, axis=-1)
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(self, anchor_bboxes, num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask, bg_index, gt_scores=None, pred_bboxes=None):
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3
        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores
        ious = self.iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)).norm(2, dim=-1).reshape([batch_size, -1, num_anchors])
        is_in_topk, topk_idxs = self._gather_topk_pyramid(gt2anchor_distances, num_anchors_list, pad_gt_mask)
        iou_candidates = ious * is_in_topk
        temp_iou_candidates = iou_candidates.flatten(end_dim=-2)
        fe_num, num_anchors = temp_iou_candidates.shape
        temp_inds = topk_idxs.flatten(end_dim=-2) + (torch.arange(fe_num, device=topk_idxs.device) * num_anchors).unsqueeze(-1)
        iou_threshold = temp_iou_candidates.view(-1)[temp_inds]
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(axis=-1, keepdim=True) + iou_threshold.std(axis=-1, keepdim=True)
        is_in_topk = torch.where(iou_candidates > iou_threshold.repeat([1, 1, num_anchors]), is_in_topk, torch.zeros_like(is_in_topk))
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile([1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        assigned_gt_index = (assigned_gt_index + batch_ind * num_max_boxes).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_index.flatten()]
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten()]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])
        assigned_scores = F.one_hot(assigned_labels.long(), self.num_classes + 1).float()
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = assigned_scores[:, :, :self.num_classes]
        if pred_bboxes is not None:
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            raise NotImplementedError
        return assigned_labels.long(), assigned_bboxes, assigned_scores


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-09):
    """
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, axis=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).tile([1, 1, topk])
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk


def iou_similarity(box1, box2, eps=1e-09):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)
    box2 = box2.unsqueeze(1)
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-09, num_classes=80):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, pred_scores, pred_bboxes, anchor_points, num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask, bg_index, gt_scores=None):
        """This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3
        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.long()
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=pred_scores.device).unsqueeze(-1)
        bbox_cls_scores = torch.zeros((batch_size, num_max_boxes, num_anchors), dtype=torch.float, device=pred_scores.device)
        for i in range(batch_size):
            bbox_cls_scores[i] = pred_scores[i, gt_labels[i].squeeze(-1)]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask.repeat([1, 1, self.topk]))
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask
        mask_positive_sum = mask_positive.sum(axis=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(axis=-2)
        assigned_gt_index = mask_positive.argmax(axis=-2)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gt_labels.flatten()[assigned_gt_index]
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index]
        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = assigned_scores[:, :, :bg_index]
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(axis=-1, keepdim=True)[0]
        max_ious_per_instance = (ious * mask_positive).max(axis=-1, keepdim=True)[0]
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics
        return assigned_labels, assigned_bboxes, assigned_scores


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class ConvBNLayer(nn.Module):

    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=filter_size, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CSPResStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_out, n, stride, act='relu', attn='eca'):
        super(CSPResStage, self).__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[block_fn(ch_mid // 2, ch_mid // 2, act=act, shortcut=True) for i in range(n)])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class RepVggBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', deploy=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deploy = deploy
        if self.deploy == False:
            self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
            self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=None)
        else:
            self.conv = nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1, groups=1)
        self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act

    def forward(self, x):
        if self.deploy:
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def switch_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1, groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__(self.conv1)
        self.__delattr__(self.conv2)
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

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
            assert isinstance(branch, nn.BatchNorm2d)
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


class BasicBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPResNet(nn.Module):

    def __init__(self, layers=[3, 6, 6, 3], channels=[64, 128, 256, 512, 1024], act='swish', return_idx=[0, 1, 2, 3, 4], use_large_stem=False, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        if use_large_stem:
            self.stem = nn.Sequential(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act), ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act), ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))
        else:
            self.stem = nn.Sequential(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act), ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act))
        n = len(channels) - 1
        self.stages = nn.Sequential(*[CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act) for i in range(n)])
        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = inputs
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


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


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1.0, eps=1e-10, reduction='none'):
        super(GIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clamp(0)
        h_inter = (ykis2 - ykis1).clamp(0)
        overlap = w_inter * h_inter
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union
        return iou, overlap, union

    def forward(self, pbox, gbox, iou_weight=1.0, loc_reweight=None):
        x1, y1, x2, y2 = torch.split(pbox, 1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, 1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)
        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - (area_c - union) / area_c
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


class VarifocalLoss(nn.Module):

    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """
        仅适用于当前任务。调用binary_cross_entropy不进行reduction。后乘上weight，再进行sum
        :param pred_score:
        :param gt_score:
        :param label:
        :param alpha:
        :param gamma:
        :return:
        """
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = (F.binary_cross_entropy(pred_score, gt_score, reduction='none') * weight).sum()
        return loss


class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(score, label, weight=weight, reduction='sum')
        return loss


class BboxLoss(nn.Module):

    def __init__(self, num_classes, reg_max):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = GIoULoss()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, assigned_labels, assigned_bboxes, assigned_scores, assigned_scores_sum):
        mask_positive = assigned_labels != self.num_classes
        num_pos = mask_positive.sum()
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), mask_positive).unsqueeze(-1)
            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum
            dist_mask = mask_positive.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([1])
            loss_iou = torch.zeros([1])
            loss_dfl = torch.zeros([1])
        return loss_l1, loss_iou, loss_dfl

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target
        target_right = target_left + 1
        weight_left = target_right - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class SPP(nn.Module):

    def __init__(self, ch_in, ch_out, k, pool_size, act='swish'):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        y = self.conv(y)
        return y


class CSPStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()
        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock':
                self.convs.add_module(str(i), BasicBlock(next_ch_in, ch_mid, act=act, shortcut=False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):

    def __init__(self, in_channels=[256, 512, 1024], out_channels=[1024, 512, 256], norm_type='bn', act='leaky', stage_fn='CSPStage', block_fn='BasicBlock', stage_num=1, block_num=3, drop_block=False, block_size=3, keep_prob=0.9, spp=False, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act
        self.num_blocks = len(in_channels)
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        self.fpn_stages = nn.ModuleList()
        self.fpn_routes = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2
            stage = nn.Sequential()
            for j in range(stage_num):
                if stage_fn == 'CSPStage':
                    stage.add_module(str(j), CSPStage(block_fn, ch_in if j == 0 else ch_out, ch_out, block_num, act=act, spp=spp and i == 0))
                else:
                    raise NotImplementedError
            if drop_block:
                stage.append(DropBlock2d(drop_prob=1 - keep_prob, block_size=block_size))
            self.fpn_stages.append(stage)
            if i < self.num_blocks - 1:
                self.fpn_routes.append(ConvBNLayer(ch_in=ch_out, ch_out=ch_out // 2, filter_size=1, stride=1, padding=0, act=act))
            ch_pre = ch_out
        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(ConvBNLayer(ch_in=out_channels[i + 1], ch_out=out_channels[i + 1], filter_size=3, stride=2, padding=1, act=act))
            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(str(j), eval(stage_fn)(block_fn, ch_in if j == 0 else ch_out, ch_out, block_num, act=act, spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock2d(block_size, keep_prob))
            pan_stages.append(stage)
        self.pan_stages = nn.Sequential(*pan_stages[::-1])
        self.pan_routes = nn.Sequential(*pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats = []
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], axis=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.0)
        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], axis=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)
        return pan_feats[::-1]


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class PPYOLOE(nn.Module):

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, targets=None, extra_info=None):
        body_feats = self.backbone(x)
        neck_feats = self.neck(body_feats)
        if self.training:
            assert targets is not None and extra_info is not None
            yolo_losses = self.head(neck_feats, targets, extra_info)
            return yolo_losses
        else:
            outputs = self.head(neck_feats)
            return outputs


class ESEAttn(nn.Module):

    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


def batch_distance2bbox(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def batch_distance2bbox_cxcywh(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = -lt + points
    x2y2 = rb + points
    cxcy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    out_bbox = torch.cat([cxcy, wh], -1)
    return out_bbox


class PPYOLOEHead(nn.Module):

    def __init__(self, in_channels=[1024, 512, 256], width_mult=1.0, num_classes=80, act='swish', fpn_strides=(32, 16, 8), grid_cell_scale=5.0, grid_cell_offset=0.5, reg_max=16, static_assigner_epoch=4, use_varifocal_loss=True, static_assigner=ATSSAssigner(9, num_classes=80), assigner=TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0), eval_input_size=[], loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}, atss_topk=9):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, 'len(in_channels) should > 0'
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.varifocal_loss = VarifocalLoss()
        self.focal_loss = FocalLoss()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max)
        self.eval_input_size = eval_input_size
        self.static_assigner_epoch = static_assigner_epoch
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4 * (self.reg_max + 1), 3, padding=1))
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()
        self.atss_topk = atss_topk
        self.atss_assign = static_assigner
        self.assigner = assigner

    def _init_weights(self, prior_prob=0.01):
        for conv in self.pred_cls:
            b = conv.bias.view(-1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        for conv in self.pred_reg:
            b = conv.bias.view(-1)
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = torch.nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(), requires_grad=False)
        if self.eval_input_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets, extra_info):
        anchors, anchor_points, num_anchors_list, stride_tensor = self.generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset, device=targets.device)
        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)
        return self.get_loss([cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor], targets, extra_info)

    def forward_eval(self, feats):
        if self.eval_input_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats, device=feats[0].device)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))
        cls_score_list = torch.cat(cls_score_list, axis=-1)
        reg_dist_list = torch.cat(reg_dist_list, axis=-1)
        pred_bboxes = batch_distance2bbox_cxcywh(anchor_points, reg_dist_list.permute(0, 2, 1))
        pred_bboxes *= stride_tensor
        return torch.cat([pred_bboxes, torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype), cls_score_list.permute(0, 2, 1)], axis=-1)

    def _generate_anchors(self, feats=None, device='cuda:0'):
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_input_size[0] / stride)
                w = int(self.eval_input_size[1] / stride)
            shift_x = torch.arange(end=w, device=device) + self.grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], axis=-1)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward(self, feats, targets=None, extra_info=None):
        assert len(feats) == len(self.fpn_strides), 'The size of feats is not equal to size of fpn_strides'
        if self.training:
            return self.forward_train(feats, targets, extra_info)
        else:
            return self.forward_eval(feats)

    def bbox_decode(self, anchor_points, pred_dist):
        batch_size, n_anchors, _ = pred_dist.shape
        pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def get_loss(self, head_outs, targets, extra_info):
        pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor = head_outs
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]
        pad_gt_mask = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        if extra_info['epoch'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = self.atss_assign(anchors, num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask, bg_index=self.num_classes, pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(pred_scores.detach(), pred_bboxes.detach() * stride_tensor, anchor_points, num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask, bg_index=self.num_classes)
            alpha_l = -1
        assigned_bboxes /= stride_tensor
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            loss_cls = self.varifocal_loss(pred_scores, assigned_scores, one_hot_label)
        else:
            loss_cls = self.focal_loss(pred_scores, assigned_scores, alpha_l)
        assigned_scores_sum = assigned_scores.sum()
        loss_cls /= assigned_scores_sum
        loss_l1, loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, assigned_labels, assigned_bboxes, assigned_scores, assigned_scores_sum)
        loss = self.loss_weight['class'] * loss_cls + self.loss_weight['iou'] * loss_iou + self.loss_weight['dfl'] * loss_dfl
        out_dict = {'total_loss': loss, 'loss_cls': loss_cls, 'loss_iou': loss_iou, 'loss_dfl': loss_dfl, 'loss_l1': loss_l1}
        return out_dict

    def generate_anchors_for_grid_cell(self, feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu'):
        """
        Like ATSS, generate anchors based on grid size.
        Args:
            feats (List[Tensor]): shape[s, (b, c, h, w)]
            fpn_strides (tuple|list): shape[s], stride for each scale feature
            grid_cell_size (float): anchor size
            grid_cell_offset (float): The range is between 0 and 1.
        Returns:
            anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
            anchor_points (Tensor): shape[l, 2], "x, y" format.
            num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
            stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
        """
        assert len(feats) == len(fpn_strides)
        anchors = []
        anchor_points = []
        num_anchors_list = []
        stride_tensor = []
        for feat, stride in zip(feats, fpn_strides):
            _, _, h, w = feat.shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack([shift_x - cell_half_size, shift_y - cell_half_size, shift_x + cell_half_size, shift_y + cell_half_size], axis=-1).clone()
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).clone()
            anchors.append(anchor.reshape([-1, 4]))
            anchor_points.append(anchor_point.reshape([-1, 2]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feat.dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchors, anchor_points, num_anchors_list, stride_tensor


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ResLayer(nn.Module):
    """Residual layer with `in_channels` inputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act='lrelu')
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act='lrelu')

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Darknet(nn.Module):
    depth2blocks = {(21): [1, 2, 2, 1], (53): [2, 8, 8, 4]}

    def __init__(self, depth, in_channels=3, stem_out_channels=32, out_features=('dark3', 'dark4', 'dark5')):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        self.stem = nn.Sequential(BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act='lrelu'), *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2))
        in_channels = stem_out_channels * 2
        num_blocks = Darknet.depth2blocks[depth]
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2
        self.dark5 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[3], stride=2), *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2))

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int=1):
        """starts with conv layer then has `num_blocks` `ResLayer`"""
        return [BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act='lrelu'), *[ResLayer(in_channels * 2) for _ in range(num_blocks)]]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(*[BaseConv(in_filters, filters_list[0], 1, stride=1, act='lrelu'), BaseConv(filters_list[0], filters_list[1], 3, stride=1, act='lrelu'), SPPBottleneck(in_channels=filters_list[1], out_channels=filters_list[0], activation='lrelu'), BaseConv(filters_list[0], filters_list[1], 3, stride=1, act='lrelu'), BaseConv(filters_list[1], filters_list[0], 1, stride=1, act='lrelu')])
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class CSPDarknet(nn.Module):

    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), depthwise=False, act='silu'):
        super().__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2, act=act), CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act))
        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2, act=act), CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2, act=act), CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark5 = nn.Sequential(Conv(base_channels * 8, base_channels * 16, 3, 2, act=act), SPPBottleneck(base_channels * 16, base_channels * 16, activation=act), CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class IOUloss(nn.Module):

    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        br = torch.min(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)
        if self.loss_type == 'iou':
            loss = 1 - iou ** 2
        elif self.loss_type == 'giou':
            c_tl = torch.min(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
            c_br = torch.max(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, depth=53, in_features=['dark3', 'dark4', 'dark5']):
        super().__init__()
        self.backbone = Darknet(depth)
        self.in_features = in_features
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act='lrelu')

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(*[self._make_cbl(in_filters, filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1)])
        return m

    def load_pretrained_model(self, filename='./weights/darknet53.mix.pth'):
        with open(filename, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        None
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)
        outputs = out_dark3, out_dark4, x0
        return outputs


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
        br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


_TORCH_VER = [int(x) for x in torch.__version__.split('.')[:2]]


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing='ij')
    else:
        return torch.meshgrid(*tensors)


class YOLOXHead(nn.Module):

    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act='silu', depthwise=False):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act), Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)]))
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act), Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)]))
            self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * self.num_classes, kernel_size=1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0))
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction='none')
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IOUloss(reduction='none')
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)
        if self.training:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype)
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)
        cls_preds = outputs[:, :, 5:]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0
        num_gts = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                try:
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs)
                except RuntimeError as e:
                    if 'CUDA out of memory. ' not in str(e):
                        raise
                    logger.error('OOM RuntimeError is raised due to the huge memory cost during label assignment.                            CPU mode is applied in this batch. If you want to avoid this issue,                            try to reduce the batch size or image size.')
                    torch.cuda.empty_cache()
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, 'cpu')
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes, self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds], expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask])
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg
        if self.use_l1:
            loss_l1 = self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1)

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-08):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, mode='gpu'):
        if mode == 'cpu':
            None
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        if mode == 'cpu':
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(gt_classes, self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-08)
        if mode == 'cpu':
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        with torch.amp.autocast(enabled=False):
            cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction='none').sum(-1)
        del cls_preds_
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * ~is_in_boxes_and_center
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        if mode == 'cpu':
            gt_matched_classes = gt_matched_classes
            fg_mask = fg_mask
            pred_ious_this_matching = pred_ious_this_matching
            matched_gt_inds = matched_gt_inds
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = (y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        center_radius = 2.5
        gt_bboxes_per_image_l = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, depth=1.0, width=1.0, in_features=('dark3', 'dark4', 'dark5'), in_channels=[256, 512, 1024], depthwise=False, act='silu'):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)
        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)
        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)
        outputs = pan_out2, pan_out1, pan_out0
        return outputs


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, extra_info=None):
        fpn_outs = self.backbone(x)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {'total_loss': loss, 'iou_loss': iou_loss, 'l1_loss': l1_loss, 'conf_loss': conf_loss, 'cls_loss': cls_loss, 'num_fg': num_fg}
        else:
            outputs = self.head(fpn_outs)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CSPDarknet,
     lambda: ([], {'dep_mul': 4, 'wid_mul': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (CSPLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropBlock2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Focus,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GIoULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (IOUloss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PPYOLOE,
     lambda: ([], {'backbone': _mock_layer(), 'neck': _mock_layer(), 'head': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResLayer,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VarifocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (YOLOFPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_Nioolek_PPYOLOE_pytorch(_paritybench_base):
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

