import sys
_module = sys.modules[__name__]
del sys
gather_models = _module
default_runtime = _module
base_dynamic = _module
base_static = _module
detection_onnxruntime_dynamic = _module
detection_onnxruntime_static = _module
boxam_vis_demo = _module
featmap_vis_demo = _module
image_demo = _module
large_image_demo = _module
conf = _module
stat = _module
mmyolo = _module
datasets = _module
transforms = _module
mix_img_transforms = _module
transforms = _module
utils = _module
yolov5_coco = _module
yolov5_voc = _module
deploy = _module
models = _module
dense_heads = _module
yolov5_head = _module
layers = _module
bbox_nms = _module
object_detection = _module
engine = _module
hooks = _module
switch_to_deploy_hook = _module
yolov5_param_scheduler_hook = _module
yolox_mode_switch_hook = _module
optimizers = _module
yolov5_optim_constructor = _module
yolov7_optim_wrapper_constructor = _module
backbones = _module
base_backbone = _module
csp_darknet = _module
csp_resnet = _module
cspnext = _module
efficient_rep = _module
yolov7_backbone = _module
data_preprocessors = _module
data_preprocessor = _module
ppyoloe_head = _module
rtmdet_head = _module
yolov5_head = _module
yolov6_head = _module
yolov7_head = _module
yolox_head = _module
detectors = _module
yolo_detector = _module
ema = _module
yolo_bricks = _module
losses = _module
iou_loss = _module
necks = _module
base_yolo_neck = _module
cspnext_pafpn = _module
ppyoloe_csppan = _module
yolov5_pafpn = _module
yolov6_pafpn = _module
yolov7_pafpn = _module
yolox_pafpn = _module
plugins = _module
cbam = _module
task_modules = _module
assigners = _module
batch_atss_assigner = _module
batch_task_aligned_assigner = _module
batch_yolov7_assigner = _module
utils = _module
coders = _module
distance_point_bbox_coder = _module
yolov5_bbox_coder = _module
yolox_bbox_coder = _module
misc = _module
registry = _module
testing = _module
_utils = _module
boxam_utils = _module
collect_env = _module
labelme_utils = _module
large_image = _module
misc = _module
setup_env = _module
version = _module
backbone = _module
focus = _module
bbox_code = _module
bbox_coder = _module
model = _module
backendwrapper = _module
model = _module
nms = _module
ort_nms = _module
trt_nms = _module
build_engine = _module
export = _module
setup = _module
test_datasets = _module
test_transforms = _module
test_mix_img_transforms = _module
test_transforms = _module
test_utils = _module
test_yolov5_coco = _module
test_yolov5_voc = _module
conftest = _module
test_mmyolo_models = _module
test_object_detection = _module
test_engine = _module
test_switch_to_deploy_hook = _module
test_yolov5_param_scheduler_hook = _module
test_yolox_mode_switch_hook = _module
test_optimizers = _module
test_yolov5_optim_constructor = _module
test_yolov7_optim_wrapper_constructor = _module
test_models = _module
test_backbone = _module
test_csp_darknet = _module
test_csp_resnet = _module
test_efficient_rep = _module
test_yolov7_backbone = _module
utils = _module
test_data_preprocessor = _module
test_data_preprocessor = _module
test_dense_heads = _module
test_ppyoloe_head = _module
test_rtmdet_head = _module
test_yolov5_head = _module
test_yolov6_head = _module
test_yolov7_head = _module
test_yolox_head = _module
test_yolo_detector = _module
test_layers = _module
test_ema = _module
test_yolo_bricks = _module
test_necks = _module
test_cspnext_pafpn = _module
test_ppyoloe_csppan = _module
test_yolov5_pafpn = _module
test_yolov6_pafpn = _module
test_yolov7_pafpn = _module
test_yolox_pafpn = _module
test_plugins = _module
test_cbam = _module
test_task_modules = _module
test_assigners = _module
test_batch_atss_assigner = _module
test_batch_task_aligned_assigner = _module
test_coders = _module
test_distance_point_bbox_coder = _module
test_yolov5_bbox_coder = _module
test_yolox_bbox_coder = _module
test_misc = _module
test_collect_env = _module
test_setup_env = _module
benchmark = _module
browse_coco_json = _module
browse_dataset = _module
dataset_analysis = _module
optimize_anchors = _module
balloon2coco = _module
labelme2coco = _module
yolo2coco = _module
coco_split = _module
download_dataset = _module
extract_subcoco = _module
ppyoloe_to_mmyolo = _module
rtmdet_to_mmyolo = _module
yolov5_to_mmyolo = _module
yolov6_to_mmyolo = _module
yolov7_to_mmyolo = _module
yolox_to_mmyolo = _module
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


import time


from collections import OrderedDict


import torch


import collections


import copy


from abc import ABCMeta


from abc import abstractmethod


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Union


import numpy as np


from numpy import random


import math


from typing import List


from functools import partial


from torch import Tensor


from typing import Callable


from typing import Dict


import torch.nn as nn


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.functional as F


import warnings


import torchvision


from collections import namedtuple


from numpy import ndarray


from torch.utils.cpp_extension import BuildExtension


import random


from torch import nn


from torch.utils.data import Dataset


from torch.nn.modules import GroupNorm


import itertools


from scipy.optimize import differential_evolution


from itertools import repeat


class ImplicitA(nn.Module):
    """Implicit add layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 0.
        std (float): Std value of implicit module. Defaults to 0.02
    """

    def __init__(self, in_channels: int, mean: float=0.0, std: float=0.02):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit + x


class ImplicitM(nn.Module):
    """Implicit multiplier layer in YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        mean (float): Mean value of implicit module. Defaults to 1.
        std (float): Std value of implicit module. Defaults to 0.02.
    """

    def __init__(self, in_channels: int, mean: float=1.0, std: float=0.02):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        return self.implicit * x


def bbox_overlaps(pred: torch.Tensor, target: torch.Tensor, iou_mode: str='ciou', bbox_format: str='xywh', siou_theta: float=4.0, eps: float=1e-07) ->torch.Tensor:
    """Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou')
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)
    bbox1_x1, bbox1_y1 = pred[:, 0], pred[:, 1]
    bbox1_x2, bbox1_y2 = pred[:, 2], pred[:, 3]
    bbox2_x1, bbox2_y1 = target[:, 0], target[:, 1]
    bbox2_x2, bbox2_y2 = target[:, 2], target[:, 3]
    overlap = (torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1)).clamp(0) * (torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1)).clamp(0)
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = w1 * h1 + w2 * h2 - overlap + eps
    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_w = enclose_wh[:, 0]
    enclose_h = enclose_wh[:, 1]
    if iou_mode == 'ciou':
        enclose_area = enclose_w ** 2 + enclose_h ** 2 + eps
        rho2_left_item = (bbox2_x1 + bbox2_x2 - (bbox1_x1 + bbox1_x2)) ** 2 / 4
        rho2_right_item = (bbox2_y1 + bbox2_y2 - (bbox1_y1 + bbox1_y2)) ** 2 / 4
        rho2 = rho2_left_item + rho2_right_item
        wh_ratio = 4 / math.pi ** 2 * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))
        ious = ious - (rho2 / enclose_area + alpha * wh_ratio)
    elif iou_mode == 'giou':
        convex_area = enclose_w * enclose_h + eps
        ious = ious - (convex_area - union) / convex_area
    elif iou_mode == 'siou':
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        sigma = torch.pow(sigma_cw ** 2 + sigma_ch ** 2, 0.5)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha, sin_beta)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (sigma_cw / enclose_w) ** 2
        rho_y = (sigma_ch / enclose_h) ** 2
        gamma = 2 - angle_cost
        distance_cost = 1 - torch.exp(-1 * gamma * rho_x) + (1 - torch.exp(-1 * gamma * rho_y))
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), siou_theta) + torch.pow(1 - torch.exp(-1 * omiga_h), siou_theta)
        ious = ious - (distance_cost + shape_cost) * 0.5
    return ious.clamp(min=-1.0, max=1.0)


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        return_iou (bool): If True, return loss and iou.
    """

    def __init__(self, iou_mode: str='ciou', bbox_format: str='xywh', eps: float=1e-07, reduction: str='mean', loss_weight: float=1.0, return_iou: bool=True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        assert iou_mode in ('ciou', 'siou', 'giou')
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]=None, avg_factor: Optional[float]=None, reduction_override: Optional[Union[str, bool]]=None) ->Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            loss or tuple(loss, iou):
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        iou = bbox_overlaps(pred, target, iou_mode=self.iou_mode, bbox_format=self.bbox_format, eps=self.eps)
        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight, reduction, avg_factor)
        if self.return_iou:
            return loss, iou
        else:
            return loss


def bbox_center_distance(bboxes: Tensor, priors: Tensor) ->Tuple[Tensor, Tensor]:
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 4) for bbox, "xyxy" format.
        priors (Tensor): Shape (num_priors, 4) for priors, "xyxy" format.

    Returns:
        distances (Tensor): Center distances between bboxes and priors,
            shape (num_priors, n).
        priors_points (Tensor): Priors cx cy points,
            shape (num_priors, 2).
    """
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = torch.stack((bbox_cx, bbox_cy), dim=1)
    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    priors_points = torch.stack((priors_cx, priors_cy), dim=1)
    distances = (bbox_points[:, None, :] - priors_points[None, :, :]).pow(2).sum(-1).sqrt()
    return distances, priors_points


def select_candidates_in_gts(priors_points: Tensor, gt_bboxes: Tensor, eps: float=1e-09) ->Tensor:
    """Select the positive priors' center in gt.

    Args:
        priors_points (Tensor): Model priors points,
            shape(num_priors, 2)
        gt_bboxes (Tensor): Ground true bboxes,
            shape(batch_size, num_gt, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): shape(batch_size, num_gt, num_priors)
    """
    batch_size, num_gt, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])
    priors_number = priors_points.size(0)
    priors_points = priors_points.unsqueeze(0).repeat(batch_size * num_gt, 1, 1)
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, priors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, priors_number, 1)
    bbox_deltas = torch.cat([priors_points - gt_bboxes_lt, gt_bboxes_rb - priors_points], dim=-1)
    bbox_deltas = bbox_deltas.reshape([batch_size, num_gt, priors_number, -1])
    return bbox_deltas.min(axis=-1)[0] > eps


def select_highest_overlaps(pos_mask: Tensor, overlaps: Tensor, num_gt: int) ->Tuple[Tensor, Tensor, Tensor]:
    """If an anchor box is assigned to multiple gts, the one with the highest
    iou will be selected.

    Args:
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
        overlaps (Tensor): IoU between all bbox and ground truth,
            shape(batch_size, num_gt, num_priors)
        num_gt (int): Number of ground truth.
    Return:
        gt_idx_pre_prior (Tensor): Target ground truth index,
            shape(batch_size, num_priors)
        fg_mask_pre_prior (Tensor): Force matching ground truth,
            shape(batch_size, num_priors)
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
    """
    fg_mask_pre_prior = pos_mask.sum(axis=-2)
    if fg_mask_pre_prior.max() > 1:
        mask_multi_gts = (fg_mask_pre_prior.unsqueeze(1) > 1).repeat([1, num_gt, 1])
        index = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(index, num_gt)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1)
        pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask)
        fg_mask_pre_prior = pos_mask.sum(axis=-2)
    gt_idx_pre_prior = pos_mask.argmax(axis=-2)
    return gt_idx_pre_prior, fg_mask_pre_prior, pos_mask


def yolov6_iou_calculator(bbox1: Tensor, bbox2: Tensor, eps: float=1e-09) ->Tensor:
    """Calculate iou for batch.

    Args:
        bbox1 (Tensor): shape(batch size, num_gt, 4)
        bbox2 (Tensor): shape(batch size, num_priors, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): IoU, shape(size, num_gt, num_priors)
    """
    bbox1 = bbox1.unsqueeze(2)
    bbox2 = bbox2.unsqueeze(1)
    bbox1_x1y1, bbox1_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    bbox2_x1y1, bbox2_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]
    overlap = (torch.minimum(bbox1_x2y2, bbox2_x2y2) - torch.maximum(bbox1_x1y1, bbox2_x1y1)).clip(0).prod(-1)
    bbox1_area = (bbox1_x2y2 - bbox1_x1y1).clip(0).prod(-1)
    bbox2_area = (bbox2_x2y2 - bbox2_x1y1).clip(0).prod(-1)
    union = bbox1_area + bbox2_area - overlap + eps
    return overlap / union


class BatchTaskAlignedAssigner(nn.Module):
    """This code referenced to
    https://github.com/meituan/YOLOv6/blob/main/yolov6/
    assigners/tal_assigner.py.
    Batch Task aligned assigner base on the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.
    Assign a corresponding gt bboxes or background to a batch of
    predicted bboxes. Each bbox will be assigned with `0` or a
    positive integer indicating the ground truth index.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        num_classes (int): number of class
        topk (int): number of bbox selected in each level
        alpha (float): Hyper-parameters related to alignment_metrics.
            Defaults to 1.0
        beta (float): Hyper-parameters related to alignment_metrics.
            Defaults to 6.
        eps (float): Eps to avoid log(0). Default set to 1e-9
    """

    def __init__(self, num_classes: int, topk: int=13, alpha: float=1.0, beta: float=6.0, eps: float=1e-07):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor, gt_labels: Tensor, gt_bboxes: Tensor, pad_bbox_flag: Tensor) ->dict:
        """Assign gt to bboxes.

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)
        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bboxes,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors,  shape (num_priors, 4)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict) Assigned result:
                assigned_labels (Tensor): Assigned labels,
                    shape(batch_size, num_priors)
                assigned_bboxes (Tensor): Assigned boxes,
                    shape(batch_size, num_priors, 4)
                assigned_scores (Tensor): Assigned scores,
                    shape(batch_size, num_priors, num_classes)
                fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                    shape(batch_size, num_priors)
        """
        priors = priors[:, :2]
        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)
        assigned_result = {'assigned_labels': gt_bboxes.new_full(pred_scores[..., 0].shape, self.num_classes), 'assigned_bboxes': gt_bboxes.new_full(pred_bboxes.shape, 0), 'assigned_scores': gt_bboxes.new_full(pred_scores.shape, 0), 'fg_mask_pre_prior': gt_bboxes.new_full(pred_scores[..., 0].shape, 0)}
        if num_gt == 0:
            return assigned_result
        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag, batch_size, num_gt)
        assigned_gt_idxs, fg_mask_pre_prior, pos_mask = select_highest_overlaps(pos_mask, overlaps, num_gt)
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior, batch_size, num_gt)
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (alignment_metrics * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * norm_align_metric
        assigned_result['assigned_labels'] = assigned_labels
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        return assigned_result

    def get_pos_mask(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor, gt_labels: Tensor, gt_bboxes: Tensor, pad_bbox_flag: Tensor, batch_size: int, num_gt: int) ->Tuple[Tensor, Tensor, Tensor]:
        """Get possible mask.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors, shape (num_priors, 2)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            pos_mask (Tensor): Possible mask,
                shape(batch_size, num_gt, num_priors)
            alignment_metrics (Tensor): Alignment metrics,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps of gt_bboxes and pred_bboxes,
                shape(batch_size, num_gt, num_priors)
        """
        alignment_metrics, overlaps = self.get_box_metrics(pred_bboxes, pred_scores, gt_labels, gt_bboxes, batch_size, num_gt)
        is_in_gts = select_candidates_in_gts(priors, gt_bboxes)
        topk_metric = self.select_topk_candidates(alignment_metrics * is_in_gts, topk_mask=pad_bbox_flag.repeat([1, 1, self.topk]).bool())
        pos_mask = topk_metric * is_in_gts * pad_bbox_flag
        return pos_mask, alignment_metrics, overlaps

    def get_box_metrics(self, pred_bboxes: Tensor, pred_scores: Tensor, gt_labels: Tensor, gt_bboxes: Tensor, batch_size: int, num_gt: int) ->Tuple[Tensor, Tensor]:
        """Compute alignment metric between all bbox and gt.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            alignment_metrics (Tensor): Align metric,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps, shape(batch_size, num_gt, num_priors)
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[idx[0], idx[1]]
        overlaps = yolov6_iou_calculator(gt_bboxes, pred_bboxes)
        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return alignment_metrics, overlaps

    def select_topk_candidates(self, alignment_gt_metrics: Tensor, using_largest_topk: bool=True, topk_mask: Optional[Tensor]=None) ->Tensor:
        """Compute alignment metric between all bbox and gt.

        Args:
            alignment_gt_metrics (Tensor): Alignment metric of gt candidates,
                shape(batch_size, num_gt, num_priors)
            using_largest_topk (bool): Controls whether to using largest or
                smallest elements.
            topk_mask (Tensor): Topk mask,
                shape(batch_size, num_gt, self.topk)
        Returns:
            Tensor: Topk candidates mask,
                shape(batch_size, num_gt, num_priors)
        """
        num_priors = alignment_gt_metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(alignment_gt_metrics, self.topk, axis=-1, largest=using_largest_topk)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_priors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor, assigned_gt_idxs: Tensor, fg_mask_pre_prior: Tensor, batch_size: int, num_gt: int) ->Tuple[Tensor, Tensor, Tensor]:
        """Get assigner info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_idxs (Tensor): Assigned ground truth indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                shape(batch_size, num_priors)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned bboxes,
                shape(batch_size, num_priors)
            assigned_scores (Tensor): Assigned scores,
                shape(batch_size, num_priors)
        """
        batch_ind = torch.arange(end=batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        assigned_gt_idxs = assigned_gt_idxs + batch_ind * num_gt
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idxs]
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idxs]
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        force_gt_scores_mask = fg_mask_pre_prior[:, :, None].repeat(1, 1, self.num_classes)
        assigned_scores = torch.where(force_gt_scores_mask > 0, assigned_scores, torch.full_like(assigned_scores, 0))
        return assigned_labels, assigned_bboxes, assigned_scores


def _cat_multi_level_tensor_in_place(*multi_level_tensor, place_hold_var):
    """concat multi-level tensor in place."""
    for level_tensor in multi_level_tensor:
        for i, var in enumerate(level_tensor):
            if len(var) > 0:
                level_tensor[i] = torch.cat(var, dim=0)
            else:
                level_tensor[i] = place_hold_var


class BatchYOLOv7Assigner(nn.Module):
    """Batch YOLOv7 Assigner.

    It consists of two assigning steps:

        1. YOLOv5 cross-grid sample assigning
        2. SimOTA assigning

    This code referenced to
    https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py.
    """

    def __init__(self, num_classes: int, num_base_priors: int, featmap_strides: Sequence[int], prior_match_thr: float=4.0, candidate_topk: int=10, iou_weight: float=3.0, cls_weight: float=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_base_priors = num_base_priors
        self.featmap_strides = featmap_strides
        self.prior_match_thr = prior_match_thr
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    @torch.no_grad()
    def forward(self, pred_results, batch_targets_normed, batch_input_shape, priors_base_sizes, grid_offset, near_neighbor_thr=0.5) ->dict:
        if batch_targets_normed.shape[1] == 0:
            num_levels = len(pred_results)
            return dict(mlvl_positive_infos=[pred_results[0].new_empty((0, 4))] * num_levels, mlvl_priors=[] * num_levels, mlvl_targets_normed=[] * num_levels)
        mlvl_positive_infos, mlvl_priors = self.yolov5_assigner(pred_results, batch_targets_normed, priors_base_sizes, grid_offset, near_neighbor_thr=near_neighbor_thr)
        mlvl_positive_infos, mlvl_priors, mlvl_targets_normed = self.simota_assigner(pred_results, batch_targets_normed, mlvl_positive_infos, mlvl_priors, batch_input_shape)
        place_hold_var = batch_targets_normed.new_empty((0, 4))
        _cat_multi_level_tensor_in_place(mlvl_positive_infos, mlvl_priors, mlvl_targets_normed, place_hold_var=place_hold_var)
        return dict(mlvl_positive_infos=mlvl_positive_infos, mlvl_priors=mlvl_priors, mlvl_targets_normed=mlvl_targets_normed)

    def yolov5_assigner(self, pred_results, batch_targets_normed, priors_base_sizes, grid_offset, near_neighbor_thr=0.5):
        num_batch_gts = batch_targets_normed.shape[1]
        assert num_batch_gts > 0
        mlvl_positive_infos, mlvl_priors = [], []
        scaled_factor = torch.ones(7, device=pred_results[0].device)
        for i in range(len(pred_results)):
            priors_base_sizes_i = priors_base_sizes[i]
            scaled_factor[2:6] = torch.tensor(pred_results[i].shape)[[3, 2, 3, 2]]
            batch_targets_scaled = batch_targets_normed * scaled_factor
            wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(wh_ratio, 1.0 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]
            if batch_targets_scaled.shape[0] == 0:
                mlvl_positive_infos.append(batch_targets_scaled.new_empty((0, 4)))
                mlvl_priors.append([])
                continue
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < near_neighbor_thr) & (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < near_neighbor_thr) & (grid_xy > 1)).T
            offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))
            batch_targets_scaled = batch_targets_scaled.repeat((5, 1, 1))[offset_inds]
            retained_offsets = grid_offset.repeat(1, offset_inds.shape[1], 1)[offset_inds]
            mlvl_positive_info = batch_targets_scaled[:, [0, 6, 2, 3]]
            retained_offsets = retained_offsets * near_neighbor_thr
            mlvl_positive_info[:, 2:] = mlvl_positive_info[:, 2:] - retained_offsets
            mlvl_positive_info[:, 2].clamp_(0, scaled_factor[2] - 1)
            mlvl_positive_info[:, 3].clamp_(0, scaled_factor[3] - 1)
            mlvl_positive_info = mlvl_positive_info.long()
            priors_inds = mlvl_positive_info[:, 1]
            mlvl_positive_infos.append(mlvl_positive_info)
            mlvl_priors.append(priors_base_sizes_i[priors_inds])
        return mlvl_positive_infos, mlvl_priors

    def simota_assigner(self, pred_results, batch_targets_normed, mlvl_positive_infos, mlvl_priors, batch_input_shape):
        num_batch_gts = batch_targets_normed.shape[1]
        assert num_batch_gts > 0
        num_levels = len(mlvl_positive_infos)
        mlvl_positive_infos_matched = [[] for _ in range(num_levels)]
        mlvl_priors_matched = [[] for _ in range(num_levels)]
        mlvl_targets_normed_matched = [[] for _ in range(num_levels)]
        for batch_idx in range(pred_results[0].shape[0]):
            targets_normed = batch_targets_normed[0]
            targets_normed = targets_normed[targets_normed[:, 0] == batch_idx]
            num_gts = targets_normed.shape[0]
            if num_gts == 0:
                continue
            _mlvl_decoderd_bboxes = []
            _mlvl_obj_cls = []
            _mlvl_priors = []
            _mlvl_positive_infos = []
            _from_which_layer = []
            for i, head_pred in enumerate(pred_results):
                _mlvl_positive_info = mlvl_positive_infos[i]
                if _mlvl_positive_info.shape[0] == 0:
                    continue
                idx = _mlvl_positive_info[:, 0] == batch_idx
                _mlvl_positive_info = _mlvl_positive_info[idx]
                _mlvl_positive_infos.append(_mlvl_positive_info)
                priors = mlvl_priors[i][idx]
                _mlvl_priors.append(priors)
                _from_which_layer.append(torch.ones(size=(_mlvl_positive_info.shape[0],)) * i)
                level_batch_idx, prior_ind, grid_x, grid_y = _mlvl_positive_info.T
                pred_positive = head_pred[level_batch_idx, prior_ind, grid_y, grid_x]
                _mlvl_obj_cls.append(pred_positive[:, 4:])
                grid = torch.stack([grid_x, grid_y], dim=1)
                pred_positive_cxcy = (pred_positive[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.featmap_strides[i]
                pred_positive_wh = (pred_positive[:, 2:4].sigmoid() * 2) ** 2 * priors * self.featmap_strides[i]
                pred_positive_xywh = torch.cat([pred_positive_cxcy, pred_positive_wh], dim=-1)
                _mlvl_decoderd_bboxes.append(pred_positive_xywh)
            _mlvl_decoderd_bboxes = torch.cat(_mlvl_decoderd_bboxes, dim=0)
            num_pred_positive = _mlvl_decoderd_bboxes.shape[0]
            if num_pred_positive == 0:
                continue
            batch_input_shape_wh = pred_results[0].new_tensor(batch_input_shape[::-1]).repeat((1, 2))
            targets_scaled_bbox = targets_normed[:, 2:6] * batch_input_shape_wh
            targets_scaled_bbox = bbox_cxcywh_to_xyxy(targets_scaled_bbox)
            _mlvl_decoderd_bboxes = bbox_cxcywh_to_xyxy(_mlvl_decoderd_bboxes)
            pair_wise_iou = bbox_overlaps(targets_scaled_bbox, _mlvl_decoderd_bboxes)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-08)
            _mlvl_obj_cls = torch.cat(_mlvl_obj_cls, dim=0).float().sigmoid()
            _mlvl_positive_infos = torch.cat(_mlvl_positive_infos, dim=0)
            _from_which_layer = torch.cat(_from_which_layer, dim=0)
            _mlvl_priors = torch.cat(_mlvl_priors, dim=0)
            gt_cls_per_image = F.one_hot(targets_normed[:, 1], self.num_classes).float().unsqueeze(1).repeat(1, num_pred_positive, 1)
            cls_preds_ = _mlvl_obj_cls[:, 1:].unsqueeze(0).repeat(num_gts, 1, 1) * _mlvl_obj_cls[:, 0:1].unsqueeze(0).repeat(num_gts, 1, 1)
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls_per_image, reduction='none').sum(-1)
            del cls_preds_
            cost = self.cls_weight * pair_wise_cls_loss + self.iou_weight * pair_wise_iou_loss
            matching_matrix = torch.zeros_like(cost)
            top_k, _ = torch.topk(pair_wise_iou, min(self.candidate_topk, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
            for gt_idx in range(num_gts):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
            targets_normed = targets_normed[matched_gt_inds]
            _mlvl_positive_infos = _mlvl_positive_infos[fg_mask_inboxes]
            _from_which_layer = _from_which_layer[fg_mask_inboxes]
            _mlvl_priors = _mlvl_priors[fg_mask_inboxes]
            for i in range(num_levels):
                layer_idx = _from_which_layer == i
                mlvl_positive_infos_matched[i].append(_mlvl_positive_infos[layer_idx])
                mlvl_priors_matched[i].append(_mlvl_priors[layer_idx])
                mlvl_targets_normed_matched[i].append(targets_normed[layer_idx])
        results = mlvl_positive_infos_matched, mlvl_priors_matched, mlvl_targets_normed_matched
        return results


class DeployFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: Tensor) ->Tensor:
        batch_size, channel, height, width = x.shape
        x = x.reshape(batch_size, channel, -1, 2, width)
        x = x.reshape(batch_size, channel, x.shape[2], 2, -1, 2)
        half_h = x.shape[2]
        half_w = x.shape[4]
        x = x.permute(0, 5, 3, 1, 2, 4)
        x = x.reshape(batch_size, channel * 4, half_h, half_w)
        return self.conv(x)


class NcnnFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: Tensor) ->Tensor:
        batch_size, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f'focus for yolox needs even feature            height and width, got {h, w}.'
        x = x.reshape(batch_size, c * h, 1, w)
        _b, _c, _h, _w = x.shape
        g = _c // 2
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)
        x = x.reshape(_b, c * h * w, 1, 1)
        _b, _c, _h, _w = x.shape
        g = _c // 2
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)
        x = x.reshape(_b, c * 4, h // 2, w // 2)
        return self.conv(x)


class GConvFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        device = next(orin_Focus.parameters()).device
        self.weight1 = torch.tensor([[1.0, 0], [0, 0]]).expand(3, 1, 2, 2)
        self.weight2 = torch.tensor([[0, 0], [1.0, 0]]).expand(3, 1, 2, 2)
        self.weight3 = torch.tensor([[0, 1.0], [0, 0]]).expand(3, 1, 2, 2)
        self.weight4 = torch.tensor([[0, 0], [0, 1.0]]).expand(3, 1, 2, 2)
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x: Tensor) ->Tensor:
        conv1 = F.conv2d(x, self.weight1, stride=2, groups=3)
        conv2 = F.conv2d(x, self.weight2, stride=2, groups=3)
        conv3 = F.conv2d(x, self.weight3, stride=2, groups=3)
        conv4 = F.conv2d(x, self.weight4, stride=2, groups=3)
        return self.conv(torch.cat([conv1, conv2, conv3, conv4], dim=1))


class TRTbatchedNMSop(torch.autograd.Function):
    """TensorRT NMS operation."""

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor, plugin_version: str='1', shareLocation: int=1, backgroundLabelId: int=-1, numClasses: int=80, topK: int=1000, keepTopK: int=100, scoreThreshold: float=0.25, iouThreshold: float=0.45, isNormalized: int=0, clipBoxes: int=0, scoreBits: int=16, caffeSemantics: int=1):
        batch_size, _, numClasses = scores.shape
        num_det = torch.randint(0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses, (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor, plugin_version: str='1', shareLocation: int=1, backgroundLabelId: int=-1, numClasses: int=80, topK: int=1000, keepTopK: int=100, scoreThreshold: float=0.25, iouThreshold: float=0.45, isNormalized: int=0, clipBoxes: int=0, scoreBits: int=16, caffeSemantics: int=1):
        out = g.op('TRT::BatchedNMSDynamic_TRT', boxes, scores, shareLocation_i=shareLocation, plugin_version_s=plugin_version, backgroundLabelId_i=backgroundLabelId, numClasses_i=numClasses, topK_i=topK, keepTopK_i=keepTopK, scoreThreshold_f=scoreThreshold, iouThreshold_f=iouThreshold, isNormalized_i=isNormalized, clipBoxes_i=clipBoxes, scoreBits_i=scoreBits, caffeSemantics_i=caffeSemantics, outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


def _batched_nms(boxes: Tensor, scores: Tensor, max_output_boxes_per_class: int=1000, iou_threshold: float=0.5, score_threshold: float=0.05, pre_top_k: int=-1, keep_top_k: int=100, box_coding: int=0):
    """Wrapper for `efficient_nms` with TensorRT.
    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x1, y1 ,x2, y2].
            Set to 1 means [x, y, w, h].
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
        (num_det, det_boxes, det_scores, det_classes),
        `num_det` of shape [N, 1]
        `det_boxes` of shape [N, num_det, 4]
        `det_scores` of shape [N, num_det]
        `det_classes` of shape [N, num_det]
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    _, _, numClasses = scores.shape
    num_det, det_boxes, det_scores, det_classes = TRTbatchedNMSop.apply(boxes, scores, '1', 1, -1, int(numClasses), min(pre_top_k, 4096), keep_top_k, score_threshold, iou_threshold, 0, 0, 16, 1)
    det_classes = det_classes.int()
    return num_det, det_boxes, det_scores, det_classes


def batched_nms(*args, **kwargs):
    """Wrapper function for `_batched_nms`."""
    return _batched_nms(*args, **kwargs)


class TRTEfficientNMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor, background_class: int=-1, box_coding: int=0, iou_threshold: float=0.45, max_output_boxes: int=100, plugin_version: str='1', score_activation: int=0, score_threshold: float=0.25):
        batch_size, _, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor, background_class: int=-1, box_coding: int=0, iou_threshold: float=0.45, max_output_boxes: int=100, plugin_version: str='1', score_activation: int=0, score_threshold: float=0.25):
        out = g.op('TRT::EfficientNMS_TRT', boxes, scores, background_class_i=background_class, box_coding_i=box_coding, iou_threshold_f=iou_threshold, max_output_boxes_i=max_output_boxes, plugin_version_s=plugin_version, score_activation_i=score_activation, score_threshold_f=score_threshold, outputs=4)
        num_det, det_boxes, det_scores, det_classes = out
        return num_det, det_boxes, det_scores, det_classes


def _efficient_nms(boxes: Tensor, scores: Tensor, max_output_boxes_per_class: int=1000, iou_threshold: float=0.5, score_threshold: float=0.05, pre_top_k: int=-1, keep_top_k: int=100, box_coding: int=0):
    """Wrapper for `efficient_nms` with TensorRT.
    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x1, y1 ,x2, y2].
            Set to 1 means [x, y, w, h].
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
        (num_det, det_boxes, det_scores, det_classes),
        `num_det` of shape [N, 1]
        `det_boxes` of shape [N, num_det, 4]
        `det_scores` of shape [N, num_det]
        `det_classes` of shape [N, num_det]
    """
    num_det, det_boxes, det_scores, det_classes = TRTEfficientNMSop.apply(boxes, scores, -1, box_coding, iou_threshold, keep_top_k, '1', 0, score_threshold)
    return num_det, det_boxes, det_scores, det_classes


def efficient_nms(*args, **kwargs):
    """Wrapper function for `_efficient_nms`."""
    return _efficient_nms(*args, **kwargs)


class ONNXNMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor, max_output_boxes_per_class: Tensor=torch.tensor([100]), iou_threshold: Tensor=torch.tensor([0.5]), score_threshold: Tensor=torch.tensor([0.05])) ->Tensor:
        device = boxes.device
        batch = scores.shape[0]
        num_det = 20
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices
        return selected_indices

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor, max_output_boxes_per_class: Tensor=torch.tensor([100]), iou_threshold: Tensor=torch.tensor([0.5]), score_threshold: Tensor=torch.tensor([0.05])):
        return g.op('NonMaxSuppression', boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, outputs=1)


_XYWH2XYXY = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]], dtype=torch.float32)


def select_nms_index(scores: Tensor, boxes: Tensor, nms_index: Tensor, batch_size: int, keep_top_k: int=-1):
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where((batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1), batched_dets.new_zeros(1))
    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(batch_inds == batch_template.unsqueeze(1), batched_labels.new_ones(1) * -1)
    N = batched_dets.shape[0]
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
    batched_labels = torch.cat((batched_labels, -batched_labels.new_ones((N, 1))), 1)
    _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(batch_size, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]
    batched_dets, batched_scores = batched_dets.split([4, 1], 2)
    batched_scores = batched_scores.squeeze(-1)
    num_dets = (batched_scores > 0).sum(1, keepdim=True)
    return num_dets, batched_dets, batched_scores, batched_labels


def onnx_nms(boxes: torch.Tensor, scores: torch.Tensor, max_output_boxes_per_class: int=100, iou_threshold: float=0.5, score_threshold: float=0.05, pre_top_k: int=-1, keep_top_k: int=100, box_coding: int=0):
    max_output_boxes_per_class = torch.tensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold])
    score_threshold = torch.tensor([score_threshold])
    batch_size, _, _ = scores.shape
    if box_coding == 1:
        boxes = boxes @ _XYWH2XYXY
    scores = scores.transpose(1, 2).contiguous()
    selected_indices = ONNXNMSop.apply(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    num_dets, batched_dets, batched_scores, batched_labels = select_nms_index(scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)
    return num_dets, batched_dets, batched_scores, batched_labels


def rtmdet_bbox_decoder(priors: Tensor, bbox_preds: Tensor, stride: Optional[Tensor]) ->Tensor:
    tl_x = priors[..., 0] - bbox_preds[..., 0]
    tl_y = priors[..., 1] - bbox_preds[..., 1]
    br_x = priors[..., 0] + bbox_preds[..., 2]
    br_y = priors[..., 1] + bbox_preds[..., 3]
    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes


def yolov5_bbox_decoder(priors: Tensor, bbox_preds: Tensor, stride: Tensor) ->Tensor:
    bbox_preds = bbox_preds.sigmoid()
    x_center = (priors[..., 0] + priors[..., 2]) * 0.5
    y_center = (priors[..., 1] + priors[..., 3]) * 0.5
    w = priors[..., 2] - priors[..., 0]
    h = priors[..., 3] - priors[..., 1]
    x_center_pred = (bbox_preds[..., 0] - 0.5) * 2 * stride + x_center
    y_center_pred = (bbox_preds[..., 1] - 0.5) * 2 * stride + y_center
    w_pred = (bbox_preds[..., 2] * 2) ** 2 * w
    h_pred = (bbox_preds[..., 3] * 2) ** 2 * h
    decoded_bboxes = torch.stack([x_center_pred, y_center_pred, w_pred, h_pred], dim=-1)
    return decoded_bboxes


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        labels = torch.stack(data_samples)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ImplicitA,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImplicitM,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_open_mmlab_mmyolo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

