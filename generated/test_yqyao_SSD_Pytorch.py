import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
coco = _module
data_augment = _module
voc0712 = _module
voc_eval = _module
demo = _module
eval = _module
layers = _module
functions = _module
detection = _module
prior_box = _module
prior_layer = _module
modules = _module
focal_loss_sigmoid = _module
focal_loss_softmax = _module
multibox_loss = _module
refine_multibox_loss = _module
weight_smooth_l1_loss = _module
weight_softmax_loss = _module
darknet = _module
dense_conv = _module
drf_res = _module
drf_vgg = _module
mobilenetv2 = _module
model_builder = _module
model_helper = _module
refine_dense_conv = _module
refine_drf_res = _module
refine_drf_vgg = _module
refine_res = _module
refine_vgg = _module
resnet = _module
vgg = _module
weave_res = _module
weave_vgg = _module
train = _module
utils = _module
augmentations = _module
averageMeter = _module
box_utils = _module
build = _module
collections = _module
convert_darknet = _module
get_class_map = _module
nms = _module
py_cpu_nms = _module
nms_wrapper = _module
timer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


from torch.nn import init


import numpy as np


import copy


import torch.utils.data as data


import torchvision.transforms as transforms


import uuid


from torchvision import transforms


import random


import math


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.nn.init as init


from torch.autograd import Variable


import time


from torch.autograd import Function


import torch.nn.functional as F


from math import sqrt as sqrt


from itertools import product as product


from math import ceil


from collections import OrderedDict


import types


from numpy import random


class PriorLayer(nn.Module):

    def __init__(self, cfg):
        super(PriorLayer, self).__init__()
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.img_wh = size_cfg.IMG_WH
        self.num_priors = len(size_cfg.ASPECT_RATIOS)
        self.feature_maps = size_cfg.FEATURE_MAPS
        self.variance = size_cfg.VARIANCE or [0.1]
        self.min_sizes = size_cfg.MIN_SIZES
        self.use_max_sizes = size_cfg.USE_MAX_SIZE
        if self.use_max_sizes:
            self.max_sizes = size_cfg.MAX_SIZES
        self.steps = size_cfg.STEPS
        self.aspect_ratios = size_cfg.ASPECT_RATIOS
        self.clip = size_cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self, img_wh, feature_maps_wh):
        self.img_wh = img_wh
        self.feature_maps_wh = feature_maps_wh
        mean = []
        for k, f in enumerate(self.feature_maps_wh):
            grid_h, grid_w = f[1], f[0]
            for i in range(grid_h):
                for j in range(grid_w):
                    f_k_h = self.img_wh[1] / self.steps[k][1]
                    f_k_w = self.img_wh[0] / self.steps[k][0]
                    cx = (j + 0.5) / f_k_w
                    cy = (i + 0.5) / f_k_h
                    s_k_h = self.min_sizes[k] / self.img_wh[1]
                    s_k_w = self.min_sizes[k] / self.img_wh[0]
                    mean += [cx, cy, s_k_w, s_k_h]
                    if self.use_max_sizes:
                        s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] / self.img_wh[0]))
                        s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] / self.img_wh[1]))
                        mean += [cx, cy, s_k_prime_w, s_k_prime_h]
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class FocalLossSigmoid(nn.Module):
    """
    sigmoid version focal loss
    """

    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.sigmoid(inputs)
        alpha_mask = self.alpha * targets
        loss_pos = -1.0 * torch.pow(1 - P, self.gamma) * torch.log(P) * targets * alpha_mask
        loss_neg = -1.0 * torch.pow(P, self.gamma) * torch.log(1 - P) * (1 - targets) * (1 - alpha_mask)
        batch_loss = loss_neg + loss_pos
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLossSoftmax(nn.Module):
    """
    softmax version focal loss
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLossSoftmax, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, (0)] * inter[:, :, (1)]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg):
        super(MultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.variance = size_cfg.VARIANCE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.threshold = cfg.TRAIN.OVERLAP
        self.OHEM = cfg.TRAIN.OHEM
        self.negpos_ratio = cfg.TRAIN.NEG_RATIO
        self.variance = size_cfg.VARIANCE
        if cfg.TRAIN.FOCAL_LOSS:
            if cfg.TRAIN.FOCAL_LOSS_TYPE == 'SOFTMAX':
                self.focaloss = FocalLossSoftmax(self.num_classes, gamma=2, size_average=False)
            else:
                self.focaloss = FocalLossSigmoid()

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            if self.num_classes == 2:
                labels = labels > 0
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        loc_t = loc_t
        conf_t = conf_t
        pos = conf_t > 0
        num_pos = pos.sum(1, keepdim=True)
        if self.OHEM:
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_hard = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
            loss_hard[pos.view(-1, 1)] = 0
            loss_hard = loss_hard.view(num, -1)
            _, loss_idx = loss_hard.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            if num_pos.data.sum() > 0:
                num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            else:
                fake_num_pos = torch.ones(32, 1).long() * 15
                num_neg = torch.clamp(self.negpos_ratio * fake_num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        else:
            loss_c = F.cross_entropy(conf_p, conf_t, size_average=False)
        if num_pos.data.sum() > 0:
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
            N = num_pos.data.sum()
        else:
            loss_l = torch.zeros(1)
            N = 1.0
        loss_l /= float(N)
        loss_c /= float(N)
        return loss_l, loss_c


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]], 1)


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def refine_match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, arm_loc_data, use_weight=False):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_boxes = decode(arm_loc_data, priors, variances)
    overlaps = jaccard(truths, decoded_boxes)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, center_size(decoded_boxes), variances)
    loc_t[idx] = loc
    conf_t[idx] = conf
    if use_weight:
        over_copy = best_truth_overlap.cpu().numpy().copy()
        over_copy.astype(np.float32)
        weight = get_iou_weights(over_copy, threshold, 0.0)
        return torch.from_numpy(weight)


class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, num_classes):
        super(RefineMultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.variance = size_cfg.VARIANCE
        self.num_classes = num_classes
        self.threshold = cfg.TRAIN.OVERLAP
        self.OHEM = cfg.TRAIN.OHEM
        self.negpos_ratio = cfg.TRAIN.NEG_RATIO
        self.object_score = cfg.MODEL.OBJECT_SCORE
        self.variance = size_cfg.VARIANCE
        if cfg.TRAIN.FOCAL_LOSS:
            if cfg.TRAIN.FOCAL_LOSS_TYPE == 'SOFTMAX':
                self.focaloss = FocalLossSoftmax(self.num_classes, gamma=2, size_average=False)
            else:
                self.focaloss = FocalLossSigmoid()

    def forward(self, predictions, targets, use_arm=False, filter_object=False, debug=False):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        if use_arm:
            arm_loc_data, arm_conf_data, loc_data, conf_data, priors = predictions
        else:
            loc_data, conf_data, _, _, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        defaults = priors.data
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            if self.num_classes == 2:
                labels = labels > 0
            if use_arm:
                bbox_weight = refine_match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx, arm_loc_data[idx].data, use_weight=False)
            else:
                match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        loc_t = loc_t
        conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        if use_arm and filter_object:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_data_temp = P[:, :, (1)]
            object_score_index = arm_conf_data_temp <= self.object_score
            pos = conf_t > 0
            pos[object_score_index.detach()] = 0
        else:
            pos = conf_t > 0
        num_pos = pos.sum(1, keepdim=True)
        if debug:
            if use_arm:
                None
            else:
                None
        if self.OHEM:
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
            loss_c[pos.view(-1, 1)] = 0
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            if num_pos.data.sum() > 0:
                num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            else:
                fake_num_pos = torch.ones(32, 1).long() * 15
                num_neg = torch.clamp(self.negpos_ratio * fake_num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        else:
            loss_c = F.cross_entropy(conf_p, conf_t, size_average=False)
        if num_pos.data.sum() > 0:
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
            N = num_pos.data.sum()
        else:
            loss_l = torch.zeros(1)
            N = 1.0
        loss_l /= float(N)
        loss_c /= float(N)
        return loss_l, loss_c


class WeightSmoothL1Loss(nn.Module):

    def __init__(self, class_num, size_average=False):
        super(WeightSmoothL1Loss, self).__init__()
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        N = inputs.size(0)
        loc_num = inputs.size(1)
        abs_out = torch.abs(inputs - targets)
        if inputs.is_cuda and not weights.is_cuda:
            weights = weights
        weights = weights.view(-1, 1)
        weights = torch.cat((weights, weights, weights, weights), 1)
        mask_big = abs_out >= 1.0
        mask_small = abs_out < 1.0
        loss_big = weights[mask_big] * (abs_out[mask_big] - 0.5)
        loss_small = weights[mask_small] * 0.5 * torch.pow(abs_out[mask_small], 2)
        loss_sum = loss_big.sum() + loss_small.sum()
        if self.size_average:
            loss = loss_sum / N * loc_num
        else:
            loss = loss_sum
        return loss


class WeightSoftmaxLoss(nn.Module):

    def __init__(self, class_num, gamma=2, size_average=True):
        super(WeightSoftmaxLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        if inputs.is_cuda and not weights.is_cuda:
            weights = weights
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        weights = weights.view(-1, 1)
        batch_loss = -weights * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class ConvBN(nn.Module):

    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01, eps=1e-05, affine=True)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1, inplace=True)


class DarknetBlock(nn.Module):

    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in // 2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


def add_extras(size):
    layers = []
    layers += [nn.Conv2d(1024, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    return layers


class Darknet19(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer1()
        self.layer2 = self._make_layer2()
        self.layer3 = self._make_layer3()
        self.layer4 = self._make_layer4()
        self.layer5 = self._make_layer5()
        self.extras = nn.ModuleList(add_extras(str(size), 1024))

    def _make_layer1(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2), ConvBN(32, 64, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer2(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2), ConvBN(64, 128, kernel_size=3, stride=1, padding=1), ConvBN(128, 64, kernel_size=1, stride=1), ConvBN(64, 128, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer3(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), ConvBN(128, 256, kernel_size=3, stride=1, padding=1), ConvBN(256, 128, kernel_size=1, stride=1), ConvBN(128, 256, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer4(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), ConvBN(256, 512, kernel_size=3, stride=1, padding=1), ConvBN(512, 256, kernel_size=1, stride=1), ConvBN(256, 512, kernel_size=3, stride=1, padding=1), ConvBN(512, 256, kernel_size=1, stride=1), ConvBN(256, 512, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def _make_layer5(self):
        layers = [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), ConvBN(512, 1024, kernel_size=3, stride=1, padding=1), ConvBN(1024, 512, kernel_size=1, stride=1), ConvBN(512, 1024, kernel_size=3, stride=1, padding=1), ConvBN(1024, 512, kernel_size=1, stride=1), ConvBN(512, 1024, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        sources = [c3, c4, c5]
        x = c5
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


class Darknet53(nn.Module):

    def __init__(self, num_blocks, size):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[4], stride=2)
        self.extras = nn.ModuleList(add_extras(str(size), 1024))
        self._init_modules()

    def _make_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in, ch_in * 2, stride=stride, padding=1)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        sources = [c3, c4, c5]
        x = c5
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def smooth_conv(size):
    layers = []
    if size == '300':
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
    else:
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 256, kernel_size=1, stride=1)]
    return layers


class DenseSSDResnet(nn.Module):

    def __init__(self, block, num_blocks, size='300', channel_size='48'):
        super(DenseSSDResnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.extras = nn.ModuleList(add_extras(str(size), 2048))
        dense_list = models.dense_conv.dense_list_res(channel_size, size)
        self.dense_list0 = nn.ModuleList(dense_list[0])
        self.dense_list1 = nn.ModuleList(dense_list[1])
        self.dense_list2 = nn.ModuleList(dense_list[2])
        self.dense_list3 = nn.ModuleList(dense_list[3])
        self.dense_list4 = nn.ModuleList(dense_list[4])
        self.dense_list5 = nn.ModuleList(dense_list[5])
        self.smooth_list = nn.ModuleList(smooth_conv(str(size)))
        self.smooth1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.dense_list0.apply(weights_init)
        self.dense_list1.apply(weights_init)
        self.dense_list2.apply(weights_init)
        self.dense_list3.apply(weights_init)
        self.dense_list4.apply(weights_init)
        self.dense_list5.apply(weights_init)
        self.smooth_list.apply(weights_init)
        self.smooth1.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        arm_sources = list()
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        dense1_p1 = self.dense_list0[0](c2)
        dense1_p2 = self.dense_list0[1](dense1_p1)
        dense1_p3 = self.dense_list0[2](dense1_p2)
        dense1_p1_conv = self.dense_list0[3](dense1_p1)
        dense1_p2_conv = self.dense_list0[4](dense1_p2)
        dense1_p3_conv = self.dense_list0[5](dense1_p3)
        c3 = self.layer2(c2)
        arm_sources.append(c3)
        dense2_p1 = self.dense_list1[0](c3)
        dense2_p2 = self.dense_list1[1](dense2_p1)
        dense2_p3 = self.dense_list1[2](dense2_p2)
        dense2_p1_conv = self.dense_list1[3](dense2_p1)
        dense2_p2_conv = self.dense_list1[4](dense2_p2)
        dense2_p3_conv = self.dense_list1[5](dense2_p3)
        c4 = self.layer3(c3)
        arm_sources.append(c4)
        dense3_up_conv = self.dense_list2[0](c4)
        dense3_up = self.dense_list2[1](dense3_up_conv)
        dense3_p1 = self.dense_list2[2](c4)
        dense3_p2 = self.dense_list2[3](dense3_p1)
        dense3_p1_conv = self.dense_list2[4](dense3_p1)
        dense3_p2_conv = self.dense_list2[5](dense3_p2)
        c5 = self.layer4(c4)
        c5_ = self.smooth1(c5)
        arm_sources.append(c5_)
        dense4_up1_conv = self.dense_list3[0](c5)
        dense4_up2_conv = self.dense_list3[1](c5)
        dense4_up1 = self.dense_list3[2](dense4_up1_conv)
        dense4_up2 = self.dense_list3[3](dense4_up2_conv)
        dense4_p = self.dense_list3[4](c5)
        dense4_p_conv = self.dense_list3[5](dense4_p)
        c6 = F.relu(self.extras[0](c5), inplace=True)
        c6 = F.relu(self.extras[1](c6), inplace=True)
        arm_sources.append(c6)
        x = c6
        dense5_up1_conv = self.dense_list4[0](c6)
        dense5_up2_conv = self.dense_list4[1](c6)
        dense5_up3_conv = self.dense_list4[2](c6)
        dense5_up1 = self.dense_list4[3](dense5_up1_conv)
        dense5_up2 = self.dense_list4[4](dense5_up2_conv)
        dense5_up3 = self.dense_list4[5](dense5_up3_conv)
        dense_out1 = torch.cat((dense1_p1_conv, c3, dense3_up, dense4_up2, dense5_up3), 1)
        dense_out1 = F.relu(self.dense_list5[0](dense_out1))
        dense_out2 = torch.cat((dense1_p2_conv, dense2_p1_conv, c4, dense4_up1, dense5_up2), 1)
        dense_out2 = F.relu(self.dense_list5[1](dense_out2))
        dense_out3 = torch.cat((dense1_p3_conv, dense2_p2_conv, dense3_p1_conv, c5_, dense5_up1), 1)
        dense_out3 = F.relu(self.dense_list5[2](dense_out3))
        dense_out4 = torch.cat((dense2_p3_conv, dense3_p2_conv, dense4_p_conv, c6), 1)
        dense_out4 = F.relu(self.dense_list5[3](dense_out4))
        sources = [dense_out1, dense_out2, dense_out3, dense_out4]
        for k, v in enumerate(self.extras):
            if k > 1:
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    tmp = x
                    index = k - 3
                    tmp = self.smooth_list[index](tmp)
                    tmp = F.relu(self.smooth_list[index + 1](tmp), inplace=True)
                    arm_sources.append(x)
                    sources.append(tmp)
        return arm_sources, sources


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def adaptive_pool(x, size):
    return F.adaptive_max_pool2d(x, size)


def adaptive_upsample(x, size):
    return F.upsample(x, size, mode='bilinear')


def trans_layers_2(raw_channels, inner_channels):
    layers = list()
    fpn_num = len(raw_channels)
    for i in range(fpn_num):
        layers += [nn.Sequential(nn.Conv2d(raw_channels[i], inner_channels[i], kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(inner_channels[i], inner_channels[i], kernel_size=3, stride=1, padding=1))]
    return layers


def weave_concat_layers_2(raw_channels, weave_add_channels, weave_channels):
    layers = list()
    weave_num = len(raw_channels)
    for i in range(weave_num):
        if i == 0:
            add_channel = weave_add_channels[i + 1][0]
        elif i == weave_num - 1:
            add_channel = weave_add_channels[i - 1][1]
        else:
            add_channel = weave_add_channels[i - 1][1] + weave_add_channels[i + 1][0]
        layers += [nn.Conv2d(raw_channels[i] + add_channel, weave_channels[i], kernel_size=1, stride=1)]
    return layers


class WeaveBlock(nn.Module):

    def __init__(self, raw_channel, weave_add_channel, dense_num):
        super(WeaveBlock, self).__init__()
        layers = list()
        for j in range(dense_num):
            layers += [nn.Conv2d(raw_channel, weave_add_channel[j], kernel_size=1, stride=1)]
        self.weave_layers = nn.ModuleList(layers)
        self._init_modules()

    def _init_modules(self):
        self.weave_layers.apply(weights_init)

    def forward(self, x):
        out = list()
        out.append(x)
        for i in range(len(self.weave_layers)):
            out.append(self.weave_layers[i](x))
        return out


def weave_layers_2(raw_channels, weave_add_channels):
    layers = list()
    num = 2
    weave_num = len(raw_channels)
    for i in range(weave_num):
        if i == 0 or i == weave_num - 1:
            layers += [WeaveBlock(raw_channels[i], weave_add_channels[i], num - 1)]
        else:
            layers += [WeaveBlock(raw_channels[i], weave_add_channels[i], num)]
    return layers


class WeaveAdapter2(nn.Module):

    def __init__(self, raw_channels, weave_add_channels, weave_channels):
        super(WeaveAdapter2, self).__init__()
        self.trans_layers = nn.ModuleList(trans_layers_2(raw_channels, weave_channels))
        self.weave_layers = nn.ModuleList(weave_layers_2(weave_channels, weave_add_channels))
        self.weave_concat_layers = nn.ModuleList(weave_concat_layers_2(weave_channels, weave_add_channels, weave_channels))
        self.weave_num = len(raw_channels)
        self._init_modules()

    def _init_modules(self):
        self.trans_layers.apply(weights_init)
        self.weave_concat_layers.apply(weights_init)

    def forward(self, x):
        trans_layers_list = list()
        weave_out = list()
        for p, t in zip(x, self.trans_layers):
            trans_layers_list.append(t(p))
        weave_list = list()
        for t, w in zip(trans_layers_list, self.weave_layers):
            weave_list.append(w(t))
        for i in range(self.weave_num):
            b, c, h, w = weave_list[i][0].size()
            if i == 0:
                up = adaptive_upsample(weave_list[i + 1][1], (h, w))
                weave = torch.cat((up, weave_list[i][0]), 1)
            elif i == self.weave_num - 1:
                pool = adaptive_pool(weave_list[i - 1][-1], (h, w))
                weave = torch.cat((pool, weave_list[i][0]), 1)
            else:
                up = adaptive_upsample(weave_list[i + 1][1], (h, w))
                pool = adaptive_pool(weave_list[i - 1][-1], (h, w))
                weave = torch.cat((up, pool, weave_list[i][0]), 1)
            weave = F.relu(self.weave_concat_layers[i](weave), inplace=True)
            weave_out.append(weave)
        return weave_out


base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512], '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]}


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


class VGG16Extractor(nn.Module):

    def __init__(self, size, channel_size='48'):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.extras = nn.ModuleList(add_extras(str(size)))
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(1024, 8)
        self.raw_channels = [512, 1024, 256, 256]
        self.weave_add_channels = [(48, 48), (48, 48), (48, 48), (48, 48)]
        self.weave_channels = [256, 256, 256, 256]
        self.weave = WeaveAdapter2(self.raw_channels, self.weave_add_channels, self.weave_channels)
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources = list()
        odm_sources = list()
        for i in range(23):
            x = self.vgg[i](x)
        c2 = x
        c2 = self.L2Norm_4_3(c2)
        arm_sources.append(c2)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        c3 = x
        c3 = self.L2Norm_5_3(c3)
        arm_sources.append(c3)
        x = F.relu(self.extras[0](x), inplace=True)
        x = F.relu(self.extras[1](x), inplace=True)
        c4 = x
        arm_sources.append(c4)
        x = F.relu(self.extras[2](x), inplace=True)
        x = F.relu(self.extras[3](x), inplace=True)
        c5 = x
        arm_sources.append(c5)
        if len(self.extras) > 4:
            x = F.relu(self.extras[4](x), inplace=True)
            x = F.relu(self.extras[5](x), inplace=True)
            c6 = x
            arm_sources.append(c6)
        odm_sources = self.weave(arm_sources)
        return arm_sources, odm_sources


class LinearBottleneck(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False, groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual
        return out


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


class MobileNet2(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, size=300, activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """
        super(MobileNet2, self).__init__()
        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.size = size
        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        self.c = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.c[0])
        self.bottlenecks = nn.ModuleList(self._make_bottlenecks())
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)
        self.extras = nn.ModuleList(add_extras(str(self.size), self.last_conv_out_ch))
        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = 'LinearBottleneck{}'.format(stage)
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t, activation=self.activation_type)
        modules[stage_name + '_0'] = first_module
        for i in range(n - 1):
            name = stage_name + '_{}'.format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6, activation=self.activation_type)
            modules[name] = module
        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = list()
        stage_name = 'Bottlenecks'
        bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1], t=1, stage=0)
        modules.append(bottleneck1)
        for i in range(1, len(self.c) - 1):
            name = stage_name + '_{}'.format(i)
            module = self._make_stage(inplanes=self.c[i], outplanes=self.c[i + 1], n=self.n[i + 1], stride=self.s[i + 1], t=self.t, stage=i)
            modules += module
        return modules

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        sources = list()
        for i in range(6):
            x = self.bottlenecks[i](x)
        sources.append(x)
        for i in range(6, 13):
            x = self.bottlenecks[i](x)
        sources.append(x)
        for i in range(13, len(self.bottlenecks)):
            x = self.bottlenecks[i](x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.img_wh = size_cfg.IMG_WH
        self.num_priors = len(size_cfg.ASPECT_RATIOS)
        self.feature_maps = size_cfg.FEATURE_MAPS
        self.variance = size_cfg.VARIANCE or [0.1]
        self.min_sizes = size_cfg.MIN_SIZES
        self.use_max_sizes = size_cfg.USE_MAX_SIZE
        if self.use_max_sizes:
            self.max_sizes = size_cfg.MAX_SIZES
        self.steps = size_cfg.STEPS
        self.aspect_ratios = size_cfg.ASPECT_RATIOS
        self.clip = size_cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            grid_h, grid_w = f[1], f[0]
            for i in range(grid_h):
                for j in range(grid_w):
                    f_k_h = self.img_wh[1] / self.steps[k][1]
                    f_k_w = self.img_wh[0] / self.steps[k][0]
                    cx = (j + 0.5) / f_k_w
                    cy = (i + 0.5) / f_k_h
                    s_k_h = self.min_sizes[k] / self.img_wh[1]
                    s_k_w = self.min_sizes[k] / self.img_wh[0]
                    mean += [cx, cy, s_k_w, s_k_h]
                    if self.use_max_sizes:
                        s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] / self.img_wh[0]))
                        s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] / self.img_wh[1]))
                        mean += [cx, cy, s_k_prime_w, s_k_prime_h]
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        if len(parts) == 1:
            return globals()[parts[0]]
        module_name = 'models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        None
        raise


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def _init_modules(self):
        self.arm_loc.apply(weights_init)
        self.arm_conf.apply(weights_init)
        if self.cfg.MODEL.REFINE:
            self.odm_loc.apply(weights_init)
            self.odm_conf.apply(weights_init)
        if self.cfg.MODEL.LOAD_PRETRAINED_WEIGHTS:
            weights = torch.load(self.cfg.MODEL.PRETRAIN_WEIGHTS)
            None
            if self.cfg.MODEL.TYPE.split('_')[-1] == 'vgg':
                self.extractor.vgg.load_state_dict(weights)
            else:
                self.extractor.load_state_dict(weights, strict=False)

    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.prior_layer = PriorLayer(cfg)
        self.priorbox = PriorBox(cfg)
        self.priors = self.priorbox.forward()
        self.extractor = get_func(cfg.MODEL.CONV_BODY)(self.size, cfg.TRAIN.CHANNEL_SIZE)
        if cfg.MODEL.REFINE:
            self.odm_channels = size_cfg.ODM_CHANNELS
            self.arm_num_classes = 2
            self.odm_loc = nn.ModuleList()
            self.odm_conf = nn.ModuleList()
        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()
        self.arm_channels = size_cfg.ARM_CHANNELS
        self.num_anchors = size_cfg.NUM_ANCHORS
        self.input_fixed = size_cfg.INPUT_FIXED
        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()
        for i in range(len(self.arm_channels)):
            if cfg.MODEL.REFINE:
                self.arm_loc += [nn.Conv2d(self.arm_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
                self.arm_conf += [nn.Conv2d(self.arm_channels[i], self.num_anchors[i] * self.arm_num_classes, kernel_size=3, padding=1)]
                self.odm_loc += [nn.Conv2d(self.odm_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
                self.odm_conf += [nn.Conv2d(self.odm_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]
            else:
                self.arm_loc += [nn.Conv2d(self.arm_channels[i], self.num_anchors[i] * 4, kernel_size=3, padding=1)]
                self.arm_conf += [nn.Conv2d(self.arm_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]
        if cfg.TRAIN.TRAIN_ON:
            self._init_modules()

    def forward(self, x):
        arm_loc = list()
        arm_conf = list()
        if self.cfg.MODEL.REFINE:
            odm_loc = list()
            odm_conf = list()
            arm_xs, odm_xs = self.extractor(x)
            for x, l, c in zip(odm_xs, self.odm_loc, self.odm_conf):
                odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
            odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        else:
            arm_xs = self.extractor(x)
        img_wh = x.size(3), x.size(2)
        feature_maps_wh = [(t.size(3), t.size(2)) for t in arm_xs]
        for x, l, c in zip(arm_xs, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        if self.cfg.MODEL.REFINE:
            output = arm_loc.view(arm_loc.size(0), -1, 4), arm_conf.view(arm_conf.size(0), -1, self.arm_num_classes), odm_loc.view(odm_loc.size(0), -1, 4), odm_conf.view(odm_conf.size(0), -1, self.num_classes), self.priors if self.input_fixed else self.prior_layer(img_wh, feature_maps_wh)
        else:
            output = arm_loc.view(arm_loc.size(0), -1, 4), arm_conf.view(arm_conf.size(0), -1, self.num_classes), self.priors if self.input_fixed else self.prior_layer(img_wh, feature_maps_wh)
        return output


def latent_layers(fpn_num):
    layers = []
    for i in range(fpn_num):
        layers += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)]
    return layers


def trans_layers(block, fpn_num):
    layers = list()
    for i in range(fpn_num):
        layers += [nn.Sequential(nn.Conv2d(block[i], 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))]
    return layers


def up_layers(fpn_num):
    layers = []
    for i in range(fpn_num - 1):
        layers += [nn.Upsample(scale_factor=2, mode='bilinear')]
    return layers


class FpnAdapter(nn.Module):

    def __init__(self, block, fpn_num):
        super(FpnAdapter, self).__init__()
        self.trans_layers = nn.ModuleList(trans_layers(block, fpn_num))
        self.up_layers = nn.ModuleList(up_layers(fpn_num))
        self.latent_layers = nn.ModuleList(latent_layers(fpn_num))
        self._init_modules()

    def _init_modules(self):
        self.trans_layers.apply(weights_init)
        self.latent_layers.apply(weights_init)

    def forward(self, x):
        trans_layers_list = list()
        fpn_out = list()
        for p, t in zip(x, self.trans_layers):
            trans_layers_list.append(t(p))
        last = F.relu(self.latent_layers[-1](trans_layers_list[-1]), inplace=True)
        fpn_out.append(last)
        _up = self.up_layers[-1](last)
        for i in range(len(trans_layers_list) - 2, -1, -1):
            q = F.relu(trans_layers_list[i] + _up, inplace=True)
            q = F.relu(self.latent_layers[i](q), inplace=True)
            fpn_out.append(q)
            if i > 0:
                _up = self.up_layers[i - 1](q)
        fpn_out = fpn_out[::-1]
        return fpn_out


class ConvPool(nn.Module):

    def __init__(self, inplane, plane):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv2d(inplane, plane, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._init_modules()

    def _init_modules(self):
        self.conv.apply(weights_init)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        return x, out


class ConvUpsample(nn.Module):

    def __init__(self, inplace, plane):
        super(ConvUpsample, self).__init__()
        self.conv = nn.Conv2d(inplace, plane, kernel_size=1, stride=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.smooth_conv = nn.Conv2d(plane, plane, kernel_size=1, stride=1)
        self._init_modules()

    def _init_modules(self):
        self.conv.apply(weights_init)
        self.smooth_conv.apply(weights_init)

    def forward(self, x):
        out = self.conv(x)
        out = self.up_sample(out)
        out = self.smooth_conv(out)
        return x, out


class ConvPoolUpsample(nn.Module):

    def __init__(self, inplace, plane):
        super(ConvPoolUpsample, self).__init__()
        self.up_conv = nn.Conv2d(inplace, plane, kernel_size=1, stride=1)
        self.pool_conv = nn.Conv2d(inplace, plane, kernel_size=1, stride=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.smooth_conv = nn.Conv2d(plane, plane, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._init_modules()

    def _init_modules(self):
        self.up_conv.apply(weights_init)
        self.smooth_conv.apply(weights_init)
        self.pool_conv.apply(weights_init)

    def forward(self, x):
        up_out = self.up_conv(x)
        pool_out = self.pool_conv(x)
        up_out = self.up_sample(up_out)
        up_out = self.smooth_conv(up_out)
        pool_out = self.pool(pool_out)
        return x, pool_out, up_out


def weave_concat_layers(block, weave_num, channel):
    layers = list()
    for i in range(weave_num):
        if i == 0 or i == weave_num - 1:
            add_channel = channel
        else:
            add_channel = channel * 2
        layers += [nn.Conv2d(block[i] + add_channel, 256, kernel_size=1, stride=1)]
    return layers


def weave_layers(block, weave_num):
    layers = list()
    add_channel = 32
    for i in range(weave_num):
        if i == 0:
            layers += [ConvPool(block[i], add_channel)]
        elif i == weave_num - 1:
            layers += [ConvUpsample(block[i], add_channel)]
        else:
            layers += [ConvPoolUpsample(block[i], add_channel)]
    return layers


class WeaveAdapter(nn.Module):

    def __init__(self, block, weave_num):
        super(WeaveAdapter, self).__init__()
        self.trans_layers = nn.ModuleList(trans_layers(block, weave_num))
        self.weave_layers = nn.ModuleList(weave_layers([256, 256, 256, 256], weave_num))
        self.weave_concat_layers = nn.ModuleList(weave_concat_layers([256, 256, 256, 256], weave_num, 48))
        self.weave_num = weave_num
        self._init_modules()

    def _init_modules(self):
        self.trans_layers.apply(weights_init)
        self.weave_concat_layers.apply(weights_init)

    def forward(self, x):
        trans_layers_list = list()
        weave_out = list()
        for p, t in zip(x, self.trans_layers):
            trans_layers_list.append(t(p))
        weave_list = list()
        for t, w in zip(trans_layers_list, self.weave_layers):
            weave_list.append(w(t))
        for i in range(self.weave_num):
            if i == 0:
                weave = torch.cat((weave_list[i][0], weave_list[i + 1][-1]), 1)
            elif i == self.weave_num - 1:
                weave = torch.cat((weave_list[i][0], weave_list[i - 1][1]), 1)
            else:
                weave = torch.cat((weave_list[i][0], weave_list[i - 1][1], weave_list[i + 1][-1]), 1)
            weave = F.relu(self.weave_concat_layers[i](weave), inplace=True)
            weave_out.append(weave)
        return weave_out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class RefineResnet(nn.Module):

    def __init__(self, block, num_blocks, size):
        super(RefineResnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras(str(size), self.inchannel))
        self.smooth1 = nn.Conv2d(self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self.fpn = FpnAdapter([512, 1024, 512, 256], 4)
        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.smooth1.apply(weights_init)

    def forward(self, x):
        odm_sources = list()
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        arm_sources = [c3, c4, c5_]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                arm_sources.append(x)
        odm_sources = self.fpn(arm_sources)
        return arm_sources, odm_sources


class SSDResnet(nn.Module):

    def __init__(self, block, num_blocks, size):
        super(SSDResnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras(str(size), self.inchannel))
        self.smooth1 = nn.Conv2d(self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.smooth1.apply(weights_init)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        sources = [c3, c4, c5_]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


class WeaveResnet(nn.Module):

    def __init__(self, block, num_blocks, size):
        super(WeaveResnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras(str(size), self.inchannel))
        self.smooth1 = nn.Conv2d(self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self.weave = WeaveAdapter([512, 1024, 512, 256], 4)
        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.smooth1.apply(weights_init)

    def forward(self, x):
        odm_sources = list()
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        arm_sources = [c3, c4, c5_]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                arm_sources.append(x)
        odm_sources = self.weave(arm_sources)
        return arm_sources, odm_sources


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBN,
     lambda: ([], {'ch_in': 4, 'ch_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvPool,
     lambda: ([], {'inplane': 4, 'plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvPoolUpsample,
     lambda: ([], {'inplace': 4, 'plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvUpsample,
     lambda: ([], {'inplace': 4, 'plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarknetBlock,
     lambda: ([], {'ch_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLossSigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBottleneck,
     lambda: ([], {'inplanes': 4, 'outplanes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeaveBlock,
     lambda: ([], {'raw_channel': 4, 'weave_add_channel': [4, 4, 4, 4], 'dense_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yqyao_SSD_Pytorch(_paritybench_base):
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

