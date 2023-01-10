import sys
_module = sys.modules[__name__]
del sys
infer = _module
lib = _module
config = _module
defaults = _module
paths_catalog = _module
data_loader = _module
build = _module
collate_batch = _module
datasets = _module
carfusion = _module
coco = _module
concat_dataset = _module
keypoint = _module
list_dataset = _module
voc = _module
samplers = _module
distributed = _module
grouped_batch_sampler = _module
iteration_based_batch_sampler = _module
transforms = _module
transforms = _module
detector = _module
detectors = _module
generalized_rcnn = _module
registry = _module
graph_head = _module
graph_head = _module
inference = _module
loss = _module
modules = _module
roi_graph_feature_extractors = _module
roi_graph_predictors = _module
utils = _module
roi_heads = _module
predictor = _module
trainer = _module
trainer_cometml = _module
train_net = _module

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


import matplotlib.path as mplPath


import torch


import numpy as np


import copy


import logging


import torch.utils.data


import torchvision


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


import math


import torch.distributed as dist


from torch.utils.data.sampler import Sampler


import itertools


from torch.utils.data.sampler import BatchSampler


import random


from torchvision.transforms import functional as F


from torch import nn


from torch.nn import functional as F


from sklearn.decomposition import PCA


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.utils.data.dataset import TensorDataset


from torch.utils.data import DataLoader


from torch.distributions import Normal


import matplotlib.pyplot as plt


from torchvision import transforms as T


import time


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.GRAPH_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.graph.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        if self.cfg.MODEL.GRAPH_ON:
            graph_features = features
            if self.training and self.cfg.MODEL.ROI_GRAPH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                graph_features = x
            x, detections, loss_graph = self.graph(graph_features, detections, targets)
            losses.update(loss_graph)
        elif self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            if self.training and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                keypoint_features = x
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


def heatmaps_to_graph(heatmaps):
    count = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    width = heatmaps.shape[2]
    heatmaps = heatmaps.view(count, num_keypoints, width * width)
    values, index = torch.max(heatmaps, 2)
    index_y = index.div(width)
    index_x = index - index_y.mul(width)
    index_y = index_y.type(torch.FloatTensor).view(count, num_keypoints, 1)
    index_x = index_x.type(torch.FloatTensor).view(count, num_keypoints, 1)
    index_y = (index_y - width / 2) / (width / 2)
    index_x = (index_x - width / 2) / (width / 2)
    values = values.type(torch.FloatTensor).view(count, num_keypoints, 1)
    graph_features = torch.cat((index_x, index_y, values), 2)
    return graph_features


def make_roi_graph_feature_extractor(cfg):
    func = registry.ROI_GRAPH_FEATURE_EXTRACTORS[cfg.MODEL.ROI_GRAPH_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points[..., 0] >= boxes[:, 0, None]) & (points[..., 0] <= boxes[:, 2, None])
    y_within = (points[..., 1] >= boxes[:, 1, None]) & (points[..., 1] <= boxes[:, 3, None])
    return x_within & y_within


def keypoints_scaled(keypoints, rois, heatmap_size, bb_pad):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long(), rois.new().long(), rois.new().long()
    width = rois[:, 2] - rois[:, 0]
    height = rois[:, 3] - rois[:, 1]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / width
    scale_y = heatmap_size / height
    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]
    x_float = (x - offset_x) * scale_x
    x = x_float.floor().long()
    y_float = (y - offset_y) * scale_y
    y = y_float.floor().long()
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1
    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 1
    invis = keypoints[..., 2] == 1
    valid_vis = (valid_loc & vis).long()
    valid_invis = (valid_loc & invis).long()
    x = (x_float - heatmap_size / 2) / (heatmap_size / 2)
    y = (y_float - heatmap_size / 2) / (heatmap_size / 2)
    squashed = torch.cat([x.view(-1, x.shape[0], x.shape[1]), y.view(-1, y.shape[0], y.shape[1])])
    return squashed, valid_vis, valid_invis


def keypoints_to_squash(keypoints, proposals, discretization_size, bb_pad):
    proposals = proposals.convert('xyxy')
    return keypoints_scaled(keypoints.keypoints, proposals.bbox, discretization_size, bb_pad)


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


class KeypointRCNNLossComputation(object):

    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size, cfg):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size
        self.bb_pad = cfg.MODEL.ROI_GRAPH_HEAD.BB_PAD
        off_diag = np.ones([cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES, cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES]) - np.eye(cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES)
        self.idx = torch.LongTensor(np.where(off_diag)[1].reshape(cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES, cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES - 1))

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(['labels', 'keypoints'])
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field('matched_idxs', matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            matched_idxs = matched_targets.get_field('matched_idxs')
            labels_per_image = matched_targets.get_field('labels')
            labels_per_image = labels_per_image
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            keypoints_per_image = matched_targets.get_field('keypoints')
            within_box = _within_box(keypoints_per_image.keypoints, matched_targets.bbox)
            vis_kp = keypoints_per_image.keypoints[..., 2] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0
            labels_per_image[~is_visible] = -1
            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)
        return labels, keypoints

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        labels, keypoints = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        proposals = list(proposals)
        for labels_per_image, keypoints_per_image, proposals_per_image in zip(labels, keypoints, proposals):
            proposals_per_image.add_field('labels', labels_per_image)
            proposals_per_image.add_field('keypoints', keypoints_per_image)
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        self._proposals = proposals
        return proposals

    def process_keypoints(self, proposals):
        for inc, proposals_per_image in enumerate(proposals):
            kp = proposals_per_image.get_field('keypoints')
            keypoint_squashed = kp.keypoints
            keypoints_per_image, valid_vis, valid_invis = keypoints_to_squash(kp, proposals_per_image, self.discretization_size, self.bb_pad)
            if keypoints_per_image.shape[0] != 0:
                keypoints_per_image = keypoints_per_image.permute(1, 2, 0)
                if inc == 0:
                    keypoints_gt = keypoints_per_image
                    vis_all = valid_vis
                    invis_all = valid_invis
                else:
                    keypoints_gt = torch.cat((keypoints_gt, keypoints_per_image))
                    vis_all = torch.cat((vis_all, valid_vis))
                    invis_all = torch.cat((invis_all, valid_invis))
        return keypoints_gt, vis_all, invis_all

    def loss_kgnn2d(self, keypoints_gt, valid_points, keypoints_logits):
        keypoints_gt = keypoints_gt.type(torch.FloatTensor) * valid_points.unsqueeze(2).type(torch.FloatTensor)
        keypoints_logits = keypoints_logits.type(torch.FloatTensor) * valid_points.unsqueeze(2).type(torch.FloatTensor)
        keypoints_gt = keypoints_gt
        keypoints_logits = keypoints_logits
        loss_occ = nll_gaussian(keypoints_gt[:, :, 0:2], keypoints_logits[:, :, 0:2], 0.1)
        return loss_occ

    def loss_edges(self, valid_points, edges):
        relations = torch.zeros(valid_points.shape[0], valid_points.shape[1] * (valid_points.shape[1] - 1))
        for count, vis in enumerate(valid_points):
            vis = vis.view(-1, 1)
            vis = vis * vis.t()
            vis = torch.gather(vis, 1, self.idx)
            relations[count] = vis.view(-1)
        relations = relations.type(torch.LongTensor)
        loss_edges = F.cross_entropy(edges.view(-1, 2), relations.view(-1))
        return loss_edges

    def loss_edges_old(self, valid_points, edges):
        relations = torch.zeros(valid_points.shape[0], valid_points.shape[1] * (valid_points.shape[1] - 1))
        count = 0
        for vis in valid_points:
            adj_mat = torch.zeros(relations.shape[1])
            loop = 0
            for i in range(vis.shape[0]):
                for j in range(vis.shape[0]):
                    if i == j:
                        continue
                    if vis[i] == 1 and vis[j] == 1:
                        adj_mat[loop] = 0
                    else:
                        adj_mat[loop] = 1
                    loop = loop + 1
            None
            relations[count] = adj_mat
        relations = relations.type(torch.LongTensor)
        loss_edges = F.cross_entropy(edges.view(-1, 2), relations.view(-1))
        return loss_edges

    def loss_kgnn3d(self, keypoint_kgnn2d, valid, projected_points):
        keypoint_kgnn2d = keypoint_kgnn2d.type(torch.FloatTensor)
        projected_points = projected_points.type(torch.FloatTensor)
        keypoint_kgnn2d = keypoint_kgnn2d * valid.unsqueeze(2).type(torch.FloatTensor)
        projected_points = projected_points * valid.unsqueeze(2).type(torch.FloatTensor)
        loss_kgnn3d = nll_gaussian(keypoint_kgnn2d[:, :, 0:2], projected_points[:, :, 0:2], 100)
        if torch.isnan(loss_kgnn3d):
            sys.exit('kgnn3d error exploded')
        return loss_kgnn3d


def make_roi_graph_loss_evaluator(cfg):
    matcher = Matcher(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, allow_low_quality_matches=False)
    fg_bg_sampler = BalancedPositiveNegativeSampler(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION)
    resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
    loss_evaluator = KeypointRCNNLossComputation(matcher, fg_bg_sampler, resolution, cfg)
    return loss_evaluator


FLIP_LEFT_RIGHT = 0


class Keypoints(object):

    def __init__(self, keypoints, size, mode=None):
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        self.keypoints = keypoints
        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError('Only FLIP_LEFT_RIGHT implemented')
        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0
        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints, self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


class PersonKeypoints(Keypoints):
    NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    FLIP_MAP = {'left_eye': 'right_eye', 'left_ear': 'right_ear', 'left_shoulder': 'right_shoulder', 'left_elbow': 'right_elbow', 'left_wrist': 'right_wrist', 'left_hip': 'right_hip', 'left_knee': 'right_knee', 'left_ankle': 'right_ankle'}


class KeypointGraphPostProcessor(nn.Module):

    def __init__(self, keypointer=None):
        super(KeypointGraphPostProcessor, self).__init__()
        self.keypointer = keypointer

    def forward(self, x, edges, boxes):
        graph_prob = x
        scores = None
        if self.keypointer:
            graph_prob, scores = self.keypointer(x, edges, boxes)
        assert len(boxes) == 1, 'Only non-batched inference supported for now'
        boxes_per_image = [box.bbox.size(0) for box in boxes]
        graph_prob = graph_prob.split(boxes_per_image, dim=0)
        scores = scores.split(boxes_per_image, dim=0)
        results = []
        for prob, box, score in zip(graph_prob, boxes, scores):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            prob = PersonKeypoints(prob, box.size)
            prob.add_field('logits', score)
            bbox.add_field('keypoints', prob)
            results.append(bbox)
        return results


def edges_to_vis(edges, n_kps):
    array = np.zeros([edges.shape[0], n_kps])
    array = torch.FloatTensor(array)
    array = Variable(array)
    for a in range(n_kps):
        value = torch.sum(edges[:, a * (n_kps - 1):(a + 1) * (n_kps - 1)], dim=1) / 2
        value[value > 0] = 1
        array[:, a] = value
    return array


def graphs_to_keypoints(graphs, edges, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)
    min_size = 0
    num_keypoints = graphs.shape[1]
    xy_preds = np.zeros((len(rois), 3, num_keypoints), dtype=np.float32)
    end_scores = np.zeros((len(rois), num_keypoints), dtype=np.float32)
    _, edge_value = edges.max(-1)
    visibilty_pred = edges_to_vis(edge_value, num_keypoints)
    end_scores = visibilty_pred.numpy() * 100
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        x_int = np.transpose(graphs[i, :, 0] + 1) * (roi_map_width / 2)
        y_int = (graphs[i, :, 1] + 1) * (roi_map_height / 2)
        x = (x_int + 0.5) * width_correction
        y = (y_int + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
    return np.transpose(xy_preds, [0, 2, 1]), end_scores


class Keypointer(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, graphs, edges, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == 1
        result, scores = graphs_to_keypoints(graphs.cpu().numpy(), edges.cpu(), boxes[0].bbox.cpu().numpy())
        return torch.from_numpy(result), torch.as_tensor(scores, device=graphs.device)


def make_roi_graph_post_processor(cfg):
    keypointer = Keypointer()
    keypoint_post_processor = KeypointGraphPostProcessor(keypointer)
    return keypoint_post_processor


class ROIGraphHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIGraphHead, self).__init__()
        self.cfg = cfg.clone()
        self.predictor_heatmap = make_roi_keypoint_predictor(cfg, cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS[-1])
        self.feature_extractor = make_roi_graph_feature_extractor(cfg)
        self.feature_extractor_heatmap = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.post_processor = make_roi_graph_post_processor(cfg)
        self.post_processor_heatmap = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator_heatmap = make_roi_keypoint_loss_evaluator(cfg)
        self.loss_evaluator = make_roi_graph_loss_evaluator(cfg)
        self.edges = cfg.MODEL.ROI_GRAPH_HEAD.EDGES
        self.KGNN2D = cfg.MODEL.ROI_GRAPH_HEAD.KGNN2D
        self.KGNN3D = cfg.MODEL.ROI_GRAPH_HEAD.KGNN3D

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator_heatmap.subsample(proposals, targets)
        x = self.feature_extractor_heatmap(features, proposals)
        kp_logits = self.predictor_heatmap(x)
        if x.shape[0] == 0:
            return torch.zeros((0, x.shape[2], 3)), proposals, {}
        graph_features = heatmaps_to_graph(kp_logits)
        for inc, proposals_per_image in enumerate(proposals):
            proposals_per_image = proposals_per_image.convert('xyxy')
            width = proposals_per_image.bbox[:, 2] - proposals_per_image.bbox[:, 0]
            height = proposals_per_image.bbox[:, 3] - proposals_per_image.bbox[:, 1]
            if inc == 0:
                ratio = width / height
            else:
                ratio = torch.cat((ratio, width / height))
        edge_logits, KGNN2D, KGNN3D = self.feature_extractor(graph_features, ratio)
        if not self.training:
            output = graph_features
            if self.edges == True:
                result = self.post_processor(graph_features, edge_logits, proposals)
                output = graph_features
            if self.KGNN2D == True:
                result = self.post_processor(KGNN2D, edge_logits, proposals)
                output = KGNN2D
            if self.KGNN3D == True:
                result = self.post_processor(KGNN3D, edge_logits, proposals)
                output = KGNN3D
            return output, result, {}
        keypoints_gt, valid_vis_all, valid_invis_all = self.loss_evaluator.process_keypoints(proposals)
        valid_all = valid_vis_all + valid_invis_all
        loss_kp = self.loss_evaluator_heatmap(proposals, kp_logits)
        if self.edges == True:
            loss_edges = self.loss_evaluator.loss_edges(valid_vis_all, edge_logits)
            loss_dict_all = dict(loss_edges=loss_edges, loss_kp=loss_kp)
        if self.KGNN2D == True:
            loss_trifocal = self.loss_evaluator.loss_kgnn2d(keypoints_gt, valid_invis_all, KGNN2D)
            loss_dict_all = dict(loss_edges=loss_edges, loss_kp=loss_kp, loss_trifocal=loss_trifocal)
        if self.KGNN3D == True:
            valid_all = (valid_vis_all + valid_invis_all) * 0 + 1
            valid_all[:, -1] = valid_all[:, -1] * 0
            valid_all[:, 8] = valid_all[:, 8] * 0
            loss_kgnn3d = self.loss_evaluator.loss_kgnn3d(KGNN2D, valid_all, KGNN3D)
            loss_dict_all = dict(loss_edges=loss_edges, loss_kp=loss_kp, loss_trifocal=loss_trifocal, loss_kgnn3d=loss_kgnn3d)
        return KGNN2D, proposals, loss_dict_all


def build_roi_graph_head(cfg, in_channels):
    return ROIGraphHead(cfg, in_channels)


def build_roi_heads(cfg, in_channels):
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(('mask', build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(('keypoint', build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.GRAPH_ON:
        roi_heads.append(('graph', build_roi_graph_head(cfg, in_channels)))
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            x = features
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return result


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


class CNN(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class GraphEncoder3D(nn.Module):

    def __init__(self, n_in, n_hid, n_out, n_kps, do_prob=0.0, factor=True):
        super(GraphEncoder3D, self).__init__()
        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            None
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            None
        self.fc_out = nn.Linear(n_hid * n_kps, n_out)
        self.flatten = Flatten()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.flatten(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.flatten(x)
        return self.fc_out(x)


class GraphEncoder(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super(GraphEncoder, self).__init__()
        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            None
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            None
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
        return self.fc_out(x)


class GraphDecoder(nn.Module):

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid, do_prob=0.0, skip_first=False):
        super(GraphDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first
        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        None
        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send, single_timestep_rel_type):
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, i:i + 1]
            all_msgs += msg
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        last_pred = inputs[:, :, :]
        curr_rel_type = rel_type[:, :, :]
        preds = []
        last_pred = self.single_step_forward(last_pred, rel_rec, rel_send, curr_rel_type)
        preds.append(last_pred)
        sizes = [preds[0].size(0), preds[0].size(1), preds[0].size(2)]
        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output
        for i in range(len(preds)):
            output[:, :, :] = preds[i]
        pred_all = output[:, :, :]
        return pred_all


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0
    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def quat_rotate(X, q):
    """Rotate points by quaternions.
    Args:
        X: B X N X 3 points
        q: B X 4 quaternions
    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x
    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)
    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


def orthographic_proj_withz(X, cam, ratio, offset_z=0.0):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)
    proj = scale * X_rot
    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    ratio = ratio.repeat(14, 1).permute(1, 0).contiguous().view(-1, 14, 1)
    proj_xy = proj_xy * torch.cat((ratio * 0 + 1, ratio), 2)
    return torch.cat((proj_xy, proj_z), 2)


def pca_computation(path):
    pca_3d = np.load(path)
    Xtrain = np.zeros((pca_3d.shape[0], 18))
    for inc, pca_inc in enumerate(pca_3d):
        imp_points = pca_inc[::2, 3:6]
        imp_points = imp_points.flatten()
        Xtrain[inc, :] = imp_points
    pca = PCA(n_components=5)
    pca.fit(Xtrain)
    U, S, VT = np.linalg.svd(Xtrain - Xtrain.mean(0))
    X_train_pca = pca.transform(Xtrain)
    mean_shape = torch.FloatTensor(pca.mean_)
    pca_component = torch.FloatTensor(pca.components_)
    mean_shape, pca_component = mean_shape, pca_component
    return mean_shape, pca_component


class graphRCNNFeatureExtractor(nn.Module):

    def __init__(self, cfg):
        super(graphRCNNFeatureExtractor, self).__init__()
        self.encoder = GraphEncoder(cfg.MODEL.ROI_GRAPH_HEAD.DIMS, cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_HIDDEN, cfg.MODEL.ROI_GRAPH_HEAD.EDGE_TYPES, cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_DROPOUT, cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_FACTOR)
        self.decoder = GraphDecoder(n_in_node=cfg.MODEL.ROI_GRAPH_HEAD.DIMS, edge_types=cfg.MODEL.ROI_GRAPH_HEAD.EDGE_TYPES, msg_hid=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_HIDDEN, msg_out=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_HIDDEN, n_hid=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_HIDDEN, do_prob=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_DROPOUT, skip_first=cfg.MODEL.ROI_GRAPH_HEAD.SKIP_FIRST)
        self.encoder_rt = GraphEncoder3D(cfg.MODEL.ROI_GRAPH_HEAD.DIMS, cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_HIDDEN, cfg.MODEL.ROI_GRAPH_HEAD.PARAMS_3D, cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES, cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_DROPOUT, cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_FACTOR)
        self.off_diag = np.ones([cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES, cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES]) - np.eye(cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES)
        self.rel_rec = np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(self.rel_rec)
        self.rel_send = torch.FloatTensor(self.rel_send)
        self.encoder = self.encoder
        self.encoder_rt = self.encoder_rt
        self.decoder = self.decoder
        self.rel_rec = self.rel_rec
        self.rel_send = self.rel_send
        self.mean_shape, self.pca_component = pca_computation('data/pca_3d_cad.npy')

    def forward(self, x, ratio):
        logits = self.encoder(x, self.rel_rec, self.rel_send)
        edges = my_softmax(logits, -1)
        KGNN2D = self.decoder(x, edges, self.rel_rec, self.rel_send, 0)
        rt = self.encoder_rt(x, self.rel_rec, self.rel_send)
        shape_basis = rt[:, 7:]
        shape = self.mean_shape
        shape = shape.view(-1, 6, 3)
        shape = torch.cat((shape, shape), 2)
        shape[:, :, 5] = -shape[:, :, 5]
        shape = shape.view(-1, 12, 3)
        idx = 8
        b = torch.zeros(shape.shape[0], 1, 3)
        b = b
        shape = torch.cat([shape[:, :idx], b, shape[:, idx:]], 1)
        shape = torch.cat([shape, b], 1)
        projected_points = orthographic_proj_withz(shape, rt[:, 0:7], ratio)
        return logits, KGNN2D, projected_points


class GraphRCNNPredictor(nn.Module):

    def __init__(self, cfg, in_channels):
        super(GraphRCNNPredictor, self).__init__()

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GraphDecoder,
     lambda: ([], {'n_in_node': 4, 'edge_types': 4, 'msg_hid': 4, 'msg_out': 4, 'n_hid': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (GraphEncoder,
     lambda: ([], {'n_in': 4, 'n_hid': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (GraphEncoder3D,
     lambda: ([], {'n_in': 4, 'n_hid': 4, 'n_out': 4, 'n_kps': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (GraphRCNNPredictor,
     lambda: ([], {'cfg': _mock_config(), 'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLP,
     lambda: ([], {'n_in': 4, 'n_hid': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_dineshreddy91_Occlusion_Net(_paritybench_base):
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

