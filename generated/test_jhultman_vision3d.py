import sys
_module = sys.modules[__name__]
del sys
pvrcnn = _module
core = _module
anchor_generator = _module
bev_drawer = _module
box_encode = _module
config = _module
geometry = _module
preprocess = _module
proposal_targets = _module
refinement_targets = _module
viz_utils = _module
dataset = _module
augmentation = _module
kitti_dataset = _module
kitti_utils = _module
detector = _module
layers = _module
model = _module
proposal = _module
refinement = _module
roi_grid_pool = _module
second = _module
sparse_cnn = _module
inference = _module
ops = _module
focal_loss = _module
iou_nms = _module
matcher = _module
train = _module
setup = _module

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


import torch


from torch import nn


import math


import numpy as np


from typing import List


from collections import defaultdict


from itertools import compress


from copy import deepcopy


from torch.utils.data import Dataset


from torch.nn import functional as F


from functools import partial


import torch.nn.functional as F


from torch.nn.modules.batchnorm import _BatchNorm


import itertools


import matplotlib.pyplot as plt


from torchvision.ops import boxes as box_ops


from torchvision.ops import nms


from torch.utils.data import DataLoader


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


def _linspace_midpoint(x0, x1, nx):
    """
    Mimics np.linspace with endpoint=False except
    samples fall in middle of bin instead of end.
    """
    dx = (x1 - x0) / nx
    x = torch.linspace(x0, x1 - dx, nx) + dx / 2
    return x


def meshgrid_midpoint(*arrays):
    """Customized meshgrid to use the above."""
    spaces = [_linspace_midpoint(*x) for x in arrays]
    grid = torch.stack(torch.meshgrid(spaces), -1)
    return grid


def torchify_anchor_attributes(cfg):
    attr = {}
    for key in ['wlh', 'center_z', 'yaw']:
        vals = [torch.tensor(anchor[key]) for anchor in cfg.ANCHORS]
        attr[key] = torch.stack(vals).float()
    return dict(attr)


class AnchorGenerator(nn.Module):
    """
    TODO: Add comment justifying unorthodox dimension ordering.
    """

    def __init__(self, cfg):
        super(AnchorGenerator, self).__init__()
        self.cfg = cfg
        self.anchor_attributes = torchify_anchor_attributes(cfg)
        self.anchors = self.make_anchors()

    def compute_grid_params(self):
        pixel_size = torch.tensor(self.cfg.VOXEL_SIZE[:2]) * self.cfg.STRIDES[-1]
        lower, upper = torch.tensor(self.cfg.GRID_BOUNDS).view(2, 3)[:, :2]
        grid_shape = ((upper - lower) / pixel_size).long()
        return lower, upper, grid_shape

    def make_anchor_sizes(self, nx, ny):
        num_yaw = self.anchor_attributes['yaw'].shape[1]
        sizes = self.anchor_attributes['wlh'][None, None, None]
        sizes = sizes.expand(nx, ny, num_yaw, -1, -1)
        return sizes

    def make_anchor_centers(self, meshgrid_params):
        num_yaw = self.anchor_attributes['yaw'].shape[1]
        anchor_z = self.anchor_attributes['center_z']
        centers = meshgrid_midpoint(*meshgrid_params)[:, :, (None)]
        centers = centers.expand(-1, -1, num_yaw, self.cfg.NUM_CLASSES, -1)
        centers[:, :, :, (torch.arange(self.cfg.NUM_CLASSES)), (2)] = anchor_z
        return centers

    def make_anchor_angles(self, nx, ny):
        yaw = self.anchor_attributes['yaw'].T[None, None, ..., None]
        yaw = yaw.expand(nx, ny, -1, -1, -1)
        return yaw

    def make_anchors(self):
        z0, z1, nz = 1, 1, 1
        (x0, y0), (x1, y1), (nx, ny) = self.compute_grid_params()
        centers = self.make_anchor_centers([(x0, x1, nx), (y0, y1, ny), (z0, z1, nz)])
        sizes = self.make_anchor_sizes(nx, ny)
        angles = self.make_anchor_angles(nx, ny)
        anchors = torch.cat((centers, sizes, angles), dim=-1)
        anchors = anchors.permute(3, 2, 1, 0, 4).contiguous()
        return anchors


class Preprocessor(nn.Module):

    def __init__(self, cfg):
        super(Preprocessor, self).__init__()
        self.voxel_generator = self.build_voxel_generator(cfg)
        self.cfg = cfg

    def build_voxel_generator(self, cfg):
        voxel_generator = spconv.utils.VoxelGenerator(voxel_size=cfg.VOXEL_SIZE, point_cloud_range=cfg.GRID_BOUNDS, max_voxels=cfg.MAX_VOXELS, max_num_points=cfg.MAX_OCCUPANCY)
        return voxel_generator

    def generate_batch_voxels(self, points):
        """Voxelize points and prefix coordinates with batch index."""
        features, coordinates, occupancy = [], [], []
        for i, p in enumerate(points):
            f, c, o = self.voxel_generator.generate(p)
            c = np.pad(c, ((0, 0), (1, 0)), constant_values=i)
            features += [f]
            coordinates += [c]
            occupancy += [o]
        return map(np.concatenate, (features, coordinates, occupancy))

    def pad_for_batch(self, points: List) ->np.ndarray:
        """Pad with subsampled points to form dense minibatch.
        :return np.ndarray of shape (B, N, C)"""
        num_points = np.r_[[p.shape[0] for p in points]]
        pad = num_points.max() - num_points
        points_batch = []
        for points_i, pad_i in zip(points, pad):
            idx = np.random.choice(points_i.shape[0], pad_i)
            points_batch += [np.concatenate((points_i, points_i[idx]))]
        points = np.stack(points_batch, axis=0)
        return points

    def forward(self, item):
        """
        Compute batch input from points.
        :points_in length B list of np.ndarrays of shape (Np, 4)
        :points_out FloatTensor of shape (B, Np, 4)
        :features FloatTensor of shape (B * Nv, 1)
        :coordinates IntTensor of shape (B * Nv, 4)
        :occupancy LongTensor of shape (B * Nv, 4)
        """
        features, coordinates, occupancy = self.generate_batch_voxels(item['points'])
        points = self.pad_for_batch(item['points'])
        keys = ['points', 'features', 'coordinates', 'occupancy', 'batch_size']
        vals = map(torch.from_numpy, (points, features, coordinates, occupancy))
        item.update(dict(zip(keys, list(vals) + [len(points)])))
        return item


class TrainPreprocessor(Preprocessor):

    def collate_mapping(self, key, val):
        if key in ['G_cls', 'G_reg', 'M_cls', 'M_reg']:
            return torch.stack(val)
        return val

    def collate(self, items):
        """Form batch item from list of items."""
        batch_item = defaultdict(list)
        for item in items:
            for key, val in item.items():
                batch_item[key] += [val]
        for key, val in batch_item.items():
            batch_item[key] = self.collate_mapping(key, val)
        return self(dict(batch_item))


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool=False):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float('inf'))
        thresholds.append(float('inf'))
        assert all(low <= high for low, high in zip(thresholds[:-1], thresholds[1:]))
        assert all(l in [-1, 0, 1] for l in labels)
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            default_match_labels = match_quality_matrix.new_full((match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8)
            return default_matches, default_match_labels
        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)
        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, (None)])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, (1)]
        match_labels[pred_inds_to_update] = 1


def _anchor_diagonal(A_wlh):
    """Reference: VoxelNet."""
    A_wl, A_h = A_wlh.split([2, 1], -1)
    A_norm = A_wl.norm(dim=-1, keepdim=True).expand_as(A_wl)
    A_norm = torch.cat((A_norm, A_h), dim=-1)
    return A_norm


def encode(boxes, anchors):
    """Both inputs of shape (*, 7)."""
    G_xyz, G_wlh, G_yaw = boxes.split([3, 3, 1], -1)
    A_xyz, A_wlh, A_yaw = anchors.split([3, 3, 1], -1)
    A_norm = _anchor_diagonal(A_wlh)
    deltas = torch.cat(((G_xyz - A_xyz) / A_norm, (G_wlh / A_wlh).log(), (G_yaw - A_yaw) % math.pi), dim=-1)
    return deltas


class ProposalTargetAssigner(nn.Module):
    """
    Match ground truth boxes to anchors by IOU.
    TODO: Make this run faster if possible.
    """

    def __init__(self, cfg):
        super(ProposalTargetAssigner, self).__init__()
        self.cfg = cfg
        self.anchors = AnchorGenerator(cfg).anchors
        self.matchers = self.build_matchers(cfg)

    def build_matchers(self, cfg):
        matchers = []
        for anchor in cfg.ANCHORS:
            matchers += [Matcher(anchor['iou_thresh'], [0, -1, +1], cfg.ALLOW_LOW_QUALITY_MATCHES)]
        return matchers

    def compute_iou(self, boxes, anchors):
        matrix = box_iou_rotated(boxes[:, ([0, 1, 3, 4, 6])], anchors[:, ([0, 1, 3, 4, 6])])
        return matrix

    def get_cls_targets(self, G_cls):
        """
        Clamps ignore to 0 and represents with binary mask.
        Note: allows anchor to be matched to multiple classes.
        """
        M_cls = G_cls.ne(-1)
        G_cls = G_cls.clamp_(min=0)
        return G_cls, M_cls

    def get_reg_targets(self, boxes, box_idx, G_cls):
        """Standard VoxelNet-style box encoding."""
        M_reg = G_cls == 1
        G_reg = encode(boxes[box_idx[M_reg]], self.anchors[M_reg])
        M_reg = M_reg.unsqueeze(-1)
        G_reg = torch.zeros_like(self.anchors).masked_scatter_(M_reg, G_reg)
        return G_reg, M_reg

    def match_class_i(self, boxes, class_idx, full_idx, i):
        class_mask = class_idx == i
        anchors = self.anchors[i].view(-1, self.cfg.BOX_DOF)
        iou = self.compute_iou(boxes[class_mask], anchors)
        matches, labels = self.matchers[i](iou)
        if class_mask.any():
            matches = full_idx[class_mask][matches]
        return matches, labels

    def apply_ignore_mask(self, matches, labels, box_ignore):
        """Ignore anchors matched to boxes[i] if box_ignore[i].
        E.g., boxes containing too few lidar points."""
        labels[box_ignore[matches] & (labels != -1)] = -1

    def match_all_classes(self, boxes, class_idx, box_ignore):
        """Match boxes to anchors based on IOU."""
        full_idx = torch.arange(boxes.shape[0], device=boxes.device)
        classes = range(self.cfg.NUM_CLASSES)
        matches, labels = zip(*[self.match_class_i(boxes, class_idx, full_idx, i) for i in classes])
        matches = torch.stack(matches).view(self.anchors.shape[:-1])
        labels = torch.stack(labels).view(self.anchors.shape[:-1])
        return matches, labels

    def to_device(self, item):
        """Move items to anchors.device for fast rotated IOU."""
        keys = ['boxes', 'class_idx', 'box_ignore']
        items = [item[k] for k in keys]
        return items

    def forward(self, item):
        boxes, class_idx, box_ignore = self.to_device(item)
        box_idx, G_cls = self.match_all_classes(boxes, class_idx, box_ignore)
        G_cls, M_cls = self.get_cls_targets(G_cls)
        G_reg, M_reg = self.get_reg_targets(boxes, box_idx, G_cls)
        item.update(dict(G_cls=G_cls, G_reg=G_reg, M_cls=M_cls, M_reg=M_reg))


class RefinementTargetAssigner(nn.Module):
    """
    TODO: Remove batch support to simplify implementation.
    TODO: Complete rewrite.
    """

    def __init__(self, cfg):
        super(RefinementTargetAssigner, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.NUM_CLASSES
        anchor_sizes = [anchor['wlh'] for anchor in cfg.ANCHORS]
        anchor_radii = [anchor['radius'] for anchor in cfg.ANCHORS]
        self.anchor_sizes = torch.tensor(anchor_sizes).float()
        self.anchor_radii = torch.tensor(anchor_radii).float()

    def batch_correspondence_mask(self, box_counts, device):
        """
        Trick to ensure boxes not matched to wrong batch index.
        """
        num_boxes, batch_size = sum(box_counts), len(box_counts)
        box_inds = torch.arange(num_boxes)
        box_batch_inds = torch.repeat_interleave(torch.arange(batch_size), torch.LongTensor(box_counts))
        mask = torch.full((batch_size, 1, num_boxes), False, dtype=torch.bool, device=device)
        mask[(box_batch_inds), :, (box_inds)] = True
        return mask

    def fill_negatives(self, targets_cls):
        """TODO: REFINEMENT_NUM_NEGATIVES needs rethinking."""
        M = self.cfg.TRAIN.REFINEMENT_NUM_NEGATIVES
        B, N, _ = targets_cls.shape
        inds = torch.randint(N, (B, M), dtype=torch.long)
        targets_cls[:, (inds), (-2)] = 1
        targets_cls[:, (inds), (-1)] = 0

    def fill_positives(self, targets_cls, inds):
        i, j, k = inds
        targets_cls[i, j, k] = 1
        targets_cls[(i), (j), -2:] = 0

    def fill_ambiguous(self, targets_cls):
        """Disables positives matched to multiple classes."""
        ambiguous = targets_cls.int().sum(2) > 1
        targets_cls[ambiguous][:, :-1] = 0
        targets_cls[ambiguous][:, (-1)] = 1

    def make_cls_targets(self, inds, shape, device):
        """
        Note that some negatives will be overwritten by positives.
        Last two indices are background and ignore, respectively.
        Uses one-hot encoding.
        """
        B, N, _ = shape
        targets_cls = torch.zeros((B, N, self.num_classes + 2), dtype=torch.long, device=device)
        targets_cls[..., -1] = 1
        self.fill_negatives(targets_cls)
        self.fill_positives(targets_cls, inds)
        self.fill_ambiguous(targets_cls)
        return targets_cls

    def make_reg_targets(self, inds, boxes, keypoints, anchor_sizes):
        i, j, k = inds
        B, N, _ = keypoints.shape
        targets_reg = torch.zeros((B, N, self.num_classes, 7), dtype=torch.float32, device=keypoints.device)
        box_centers, box_sizes, box_angles = boxes.split([3, 3, 1], dim=-1)
        targets_reg[(i), (j), (k), 0:3] = box_centers[k] - keypoints[i, j]
        targets_reg[(i), (j), (k), 3:6] = (box_sizes[k] - anchor_sizes[k]) / anchor_sizes[k]
        targets_reg[(i), (j), (k), 6:7] = box_angles[k]
        return targets_reg

    def match_keypoints(self, boxes, keypoints, anchor_radii, class_ids, box_counts):
        """Find keypoints within spherical radius of ground truth center."""
        box_centers, box_sizes, box_angles = boxes.split([3, 3, 1], dim=-1)
        distances = torch.norm(keypoints[:, :, (None), :] - box_centers, dim=-1)
        in_radius = distances < anchor_radii[class_ids]
        in_radius &= self.batch_correspondence_mask(box_counts, keypoints.device)
        return in_radius.nonzero().t()

    def get_targets(self, item):
        box_counts = [b.shape[0] for b in item['boxes']]
        boxes = torch.cat(item['boxes'], dim=0)
        class_ids = torch.cat(item['class_ids'], dim=0)
        keypoints = item['keypoints']
        device = keypoints.device
        anchor_sizes = self.anchor_sizes
        anchor_radii = self.anchor_radii
        i, j, k = self.match_keypoints(boxes, keypoints, anchor_radii, class_ids, box_counts)
        inds = i, j, class_ids[k]
        targets_cls = self.make_cls_targets(inds, keypoints.shape, device)
        targets_reg = self.make_reg_targets(inds, boxes, keypoints, anchor_sizes)
        return targets_cls, targets_reg

    def forward(self, item):
        raise NotImplementedError


class VoxelFeatureExtractor(nn.Module):
    """Computes mean of non-zero points within voxel."""

    def forward(self, feature, occupancy):
        """
        :feature FloatTensor of shape (N, K, C)
        :return FloatTensor of shape (N, C)
        """
        denominator = occupancy.type_as(feature).view(-1, 1)
        feature = (feature.sum(1) / denominator).contiguous()
        return feature


class BEVFeatureGatherer(nn.Module):
    """Gather BEV features at keypoints using bilinear interpolation."""

    def __init__(self, cfg, voxel_offset, base_voxel_size):
        super(BEVFeatureGatherer, self).__init__()
        self.cfg = cfg
        self.pixel_offset = voxel_offset[:2]
        self.base_pixel_size = base_voxel_size[:2]

    def normalize_indices(self, indices, H, W):
        """
        F.grid_sample expects normalized indices on (-1, +1).
        Note: We swap H and W because spconv transposes the feature map.
        """
        image_dims = indices.new_tensor([W - 1, H - 1])
        indices = torch.min(torch.clamp(indices, min=0), image_dims)
        indices = 2 * (indices / (image_dims - 1)) - 1
        return indices

    def compute_bev_indices(self, keypoint_xyz, H, W):
        """Convert xyz coordinates to fractional BEV indices."""
        indices = keypoint_xyz[:, (None), :, :2] - self.pixel_offset
        indices = indices / (self.base_pixel_size * self.cfg.STRIDES[-1])
        indices = self.normalize_indices(indices, H, W).flip(3)
        return indices

    def forward(self, feature_map, keypoint_xyz):
        N, C, H, W = feature_map.shape
        indices = self.compute_bev_indices(keypoint_xyz, H, W)
        features = F.grid_sample(feature_map, indices, align_corners=True).squeeze(2)
        return features


class MLP(nn.Sequential):

    def __init__(self, channels, bias=False, bn=False, relu=True):
        super(MLP, self).__init__()
        bias, bn, relu = map(partial(self._repeat, n=len(channels)), (bias, bn, relu))
        for i in range(len(channels) - 1):
            self.add_module(f'linear_{i}', nn.Linear(channels[i], channels[i + 1], bias=bias[i]))
            nn.init.normal_(self[-1].weight, std=0.01)
            if bias[i]:
                nn.init.constant_(self[-1].bias, 0)
            if bn[i]:
                self.add_module(f'batchnorm_{i}', nn.BatchNorm1d(channels[i + 1]))
                nn.init.constant_(self[-1].weight, 1)
                nn.init.constant_(self[-1].bias, 0)
            if relu[i]:
                self.add_module(f'relu_{i}', nn.ReLU(inplace=True))

    def _repeat(self, module, n):
        if not isinstance(module, (tuple, list)):
            module = [module] * (n - 1)
        return module


def compute_grid_shape(cfg):
    voxel_size = np.r_[cfg.VOXEL_SIZE]
    lower, upper = np.reshape(cfg.GRID_BOUNDS, (2, 3))
    grid_shape = (upper - lower) / voxel_size + [0, 0, 1]
    grid_shape = np.int32(grid_shape)[::-1].tolist()
    return grid_shape


def random_choice(x, n, dim=0):
    """Emulate numpy.random.choice."""
    assert dim == 0, 'Currently support only dim 0.'
    inds = torch.randint(0, x.size(dim), (n,), device=x.device)
    return x[inds]


class SparseCNNBase(nn.Module):
    """
    block      shape    stride
    0    [ 4, 8y, 8x, 41]    1
    1    [32, 4y, 4x, 21]    2
    2    [64, 2y, 2x, 11]    4
    3    [64, 1y, 1x,  5]    8
    4    [64, 1y, 1x,  2]    8
    """

    def __init__(self, cfg):
        """grid_shape given in ZYX order."""
        super(SparseCNNBase, self).__init__()
        self.cfg = cfg
        self.grid_shape = compute_grid_shape(cfg)
        self.base_voxel_size = torch.FloatTensor(cfg.VOXEL_SIZE)
        self.voxel_offset = torch.FloatTensor(cfg.GRID_BOUNDS[:3])
        self.make_blocks(cfg)

    def make_blocks(self, cfg):
        """Subclasses must implement this method."""
        raise NotImplementedError

    def maybe_bias_init(self, module, val):
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, val)

    def kaiming_init(self, module):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out', nonlinearity='relu')
        self.maybe_bias_init(module, 0)

    def batchnorm_init(self, module):
        nn.init.constant_(module.weight, 1)
        self.maybe_bias_init(module, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                self.batchnorm_init(m)

    def to_global(self, stride, volume):
        """
        Convert integer voxel indices to metric coordinates.
        Indices are reversed ijk -> kji to maintain correspondence with xyz.
        Sparse voxels are padded with subsamples to allow batch PointNet processing.
        :voxel_size length-3 tensor describing size of atomic voxel, accounting for stride.
        :voxel_offset length-3 tensor describing coordinate offset of voxel grid.
        """
        index = torch.flip(volume.indices, (1,))
        voxel_size = self.base_voxel_size * stride
        xyz = index[(...), 0:3].float() * voxel_size
        xyz = xyz + self.voxel_offset
        xyz = self.pad_batch(xyz, index[..., -1], volume.batch_size)
        feature = self.pad_batch(volume.features, index[..., -1], volume.batch_size)
        return xyz, feature

    def compute_pad_amounts(self, batch_index, batch_size):
        """Compute padding needed to form dense minibatch."""
        helper_index = torch.arange(batch_size + 1, device=batch_index.device)
        helper_index = helper_index.unsqueeze(0).contiguous().int()
        batch_index = batch_index.unsqueeze(0).contiguous().int()
        start_index = searchsorted(batch_index, helper_index).squeeze(0)
        batch_count = start_index[1:] - start_index[:-1]
        pad = list((batch_count.max() - batch_count).cpu().numpy())
        batch_count = list(batch_count.cpu().numpy())
        return batch_count, pad

    def pad_batch(self, x, batch_index, batch_size):
        """Pad sparse tensor with subsamples to form dense minibatch."""
        if batch_size == 1:
            return x.unsqueeze(0)
        batch_count, pad = self.compute_pad_amounts(batch_index, batch_size)
        chunks = x.split(batch_count)
        pad_values = [random_choice(c, n) for c, n in zip(chunks, pad)]
        chunks = [torch.cat((c, p)) for c, p in zip(chunks, pad_values)]
        return torch.stack(chunks)

    def to_bev(self, volume):
        """Collapse z-dimension to form BEV feature map."""
        volume = volume.dense()
        N, C, D, H, W = volume.shape
        bev = volume.view(N, C * D, H, W)
        return bev

    def forward(self, features, coordinates, batch_size):
        x0 = spconv.SparseConvTensor(features, coordinates.int(), self.grid_shape, batch_size)
        x1 = self.blocks[0](x0)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        x4 = self.blocks[3](x3)
        x4 = self.to_bev(x4)
        args = zip(self.cfg.STRIDES, (x0, x1, x2, x3))
        x = list(itertools.starmap(self.to_global, args))
        return x, x4


def make_sparse_conv_layer(C_in, C_out, *args, **kwargs):
    layer = spconv.SparseSequential(spconv.SparseConv3d(C_in, C_out, *args, **kwargs, bias=False), nn.BatchNorm1d(C_out, eps=0.001, momentum=0.01), nn.ReLU())
    return layer


def make_subm_layer(C_in, C_out, *args, **kwargs):
    layer = spconv.SparseSequential(spconv.SubMConv3d(C_in, C_out, 3, *args, **kwargs, bias=False), nn.BatchNorm1d(C_out, eps=0.001, momentum=0.01), nn.ReLU())
    return layer


class SpMiddleFHD(SparseCNNBase):

    def make_blocks(self, cfg):
        blocks = []
        blocks += [spconv.SparseSequential(make_subm_layer(cfg.C_IN, 16, 3, indice_key='subm0'), make_subm_layer(16, 16, 3, indice_key='subm0'), make_sparse_conv_layer(16, 32, 3, 2, padding=1))]
        blocks += [spconv.SparseSequential(make_subm_layer(32, 32, 3, indice_key='subm1'), make_subm_layer(32, 32, 3, indice_key='subm1'), make_sparse_conv_layer(32, 64, 3, 2, padding=1))]
        blocks += [spconv.SparseSequential(make_subm_layer(64, 64, 3, indice_key='subm2'), make_subm_layer(64, 64, 3, indice_key='subm2'), make_subm_layer(64, 64, 3, indice_key='subm2'), make_sparse_conv_layer(64, 64, 3, 2, padding=[0, 1, 1]))]
        blocks += [spconv.SparseSequential(make_subm_layer(64, 64, 3, indice_key='subm3'), make_subm_layer(64, 64, 3, indice_key='subm3'), make_subm_layer(64, 64, 3, indice_key='subm3'), make_sparse_conv_layer(64, 64, (3, 1, 1), (2, 1, 1)))]
        self.blocks = spconv.SparseSequential(*blocks)


class SpMiddleFHDLite(SparseCNNBase):

    def make_blocks(self, cfg):
        self.blocks = spconv.SparseSequential(make_sparse_conv_layer(cfg.C_IN, 32, 3, 2, padding=1), make_sparse_conv_layer(32, 64, 3, 2, padding=1), make_sparse_conv_layer(64, 64, 3, 2, padding=[0, 1, 1]), make_sparse_conv_layer(64, 64, (3, 1, 1), (2, 1, 1)))


CNN_FACTORY = dict(SpMiddleFHD=SpMiddleFHD, SpMiddleFHDLite=SpMiddleFHDLite)


def nms_rotated(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).
    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.
    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.
    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.
    As an extreme example, consider a single character v and the square box around it.
    If the angle is 0 degree, the object (text) would be read as 'v';
    If the angle is 90 degrees, the object (text) would become '>';
    If the angle is 180 degrees, the object (text) would become '^';
    If the angle is 270/-90 degrees, the object (text) would become '<'
    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.
    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)
    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)
    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.
    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    return _C.nms_rotated(boxes, scores, iou_threshold)


def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold
    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = (torch.max(boxes[:, (0)], boxes[:, (1)]) + torch.max(boxes[:, (2)], boxes[:, (3)]) / 2).max()
    min_coordinate = (torch.min(boxes[:, (0)], boxes[:, (1)]) - torch.min(boxes[:, (2)], boxes[:, (3)]) / 2).min()
    offsets = idxs * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, (None)]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep


def decode(deltas, anchors):
    """Both inputs of shape (*, 7)."""
    P_xyz, P_wlh, P_yaw = deltas.split([3, 3, 1], -1)
    A_xyz, A_wlh, A_yaw = anchors.split([3, 3, 1], -1)
    A_norm = _anchor_diagonal(A_wlh)
    boxes = torch.cat((P_xyz * A_norm + A_xyz, P_wlh.exp() * A_wlh, P_yaw + A_yaw), dim=-1)
    return boxes


class ProposalLayer(nn.Module):
    """
    Use BEV feature map to generate 3D box proposals.
    TODO: Fix long variable names, ugly line wraps.
    """

    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.cfg = cfg
        self.conv_cls = nn.Conv2d(cfg.PROPOSAL.C_IN, cfg.NUM_CLASSES * cfg.NUM_YAW, 1)
        self.conv_reg = nn.Conv2d(cfg.PROPOSAL.C_IN, cfg.NUM_CLASSES * cfg.NUM_YAW * cfg.BOX_DOF, 1)
        self.TOPK, self.DOF = cfg.PROPOSAL.TOPK, cfg.BOX_DOF
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.conv_cls.bias, -math.log(1 - 0.01) / 0.01)
        nn.init.constant_(self.conv_reg.bias, 0)
        for m in (self.conv_cls.weight, self.conv_reg.weight):
            nn.init.normal_(m, std=0.01)

    def _generate_group_idx(self, B, n_cls):
        """Compute unique group_idx based on (batch_idx, class_idx) tuples."""
        batch_idx = torch.arange(B)[:, (None)].expand(-1, n_cls)
        class_idx = torch.arange(n_cls)[(None), :].expand(B, -1)
        group_idx = class_idx + n_cls * batch_idx
        b, c, g = [x[..., None].expand(-1, -1, self.TOPK).reshape(-1) for x in (batch_idx, class_idx, group_idx)]
        return b, c, g

    def _above_score_thresh(self, scores, class_idx):
        """Classes may have different score thresholds."""
        thresh = scores.new_tensor([a['score_thresh'] for a in self.cfg.ANCHORS])
        mask = scores > thresh[class_idx]
        return mask

    def _multiclass_batch_nms(self, boxes, scores):
        """Only boxes with same group_idx are jointly considered in nms"""
        B, n_cls = scores.shape[:2]
        scores = scores.view(-1)
        boxes = boxes.view(-1, self.DOF)
        bev_boxes = boxes[:, ([0, 1, 3, 4, 6])]
        batch_idx, class_idx, group_idx = self._generate_group_idx(B, n_cls)
        idx = batched_nms_rotated(bev_boxes, scores, group_idx, iou_threshold=0.01)
        boxes, batch_idx, class_idx, scores = [x[idx] for x in (boxes, batch_idx, class_idx, scores)]
        mask = self._above_score_thresh(scores, class_idx)
        out = [x[mask] for x in (boxes, batch_idx, class_idx, scores)]
        return out

    def _decode(self, reg_map, anchors, anchor_idx):
        """Expands anchors in batch dimension and calls decode."""
        B, n_cls = reg_map.shape[:2]
        anchor_idx = anchor_idx[..., None].expand(-1, -1, -1, self.DOF)
        deltas = reg_map.reshape(B, n_cls, -1, self.cfg.BOX_DOF).gather(2, anchor_idx)
        anchors = anchors.view(1, n_cls, -1, self.cfg.BOX_DOF).expand(B, -1, -1, -1).gather(2, anchor_idx)
        boxes = decode(deltas, anchors)
        return boxes

    def inference(self, feature_map, anchors):
        """:return (boxes, batch_idx, class_idx, scores)"""
        cls_map, reg_map = self(feature_map)
        score_map = cls_map.sigmoid_()
        B, n_cls = score_map.shape[:2]
        scores, anchor_idx = score_map.view(B, n_cls, -1).topk(self.TOPK, -1)
        boxes = self._decode(reg_map, anchors, anchor_idx)
        out = self._multiclass_batch_nms(boxes, scores)
        return out

    def reshape_cls(self, cls_map):
        B, _, ny, nx = cls_map.shape
        shape = B, self.cfg.NUM_CLASSES, self.cfg.NUM_YAW, ny, nx
        cls_map = cls_map.view(shape)
        return cls_map

    def reshape_reg(self, reg_map):
        B, _, ny, nx = reg_map.shape
        shape = B, self.cfg.NUM_CLASSES, self.cfg.BOX_DOF, -1, ny, nx
        reg_map = reg_map.view(shape).permute(0, 1, 3, 4, 5, 2)
        return reg_map

    def forward(self, feature_map):
        cls_map = self.reshape_cls(self.conv_cls(feature_map))
        reg_map = self.reshape_reg(self.conv_reg(feature_map))
        return cls_map, reg_map


class RefinementLayer(nn.Module):
    """
    Uses pooled features to refine proposals.

    TODO: Pass class predictions from proposals since this
        module only predicts confidence.
    TODO: Implement RefinementLoss.
    TODO: Decide if decode box predictions / apply box
        deltas here or elsewhere.
    """

    def __init__(self, cfg):
        super(RefinementLayer, self).__init__()
        self.mlp = self.build_mlp(cfg)
        self.cfg = cfg

    def build_mlp(self, cfg):
        """
        TODO: Check if should use bias.
        """
        channels = cfg.REFINEMENT.MLPS + [cfg.BOX_DOF + 1]
        mlp = MLP(channels, bias=True, bn=False, relu=[True, False])
        return mlp

    def apply_refinements(self, box_deltas, boxes):
        raise NotImplementedError

    def inference(self, points, features, boxes):
        box_deltas, scores = self(points, features, boxes)
        boxes = self.apply_refinements(box_deltas, boxes)
        scores = scores.sigmoid()
        positive = 1 - scores[(...), -1:]
        _, indices = torch.topk(positive, k=self.cfg.PROPOSAL.TOPK, dim=1)
        indices = indices.expand(-1, -1, self.cfg.NUM_CLASSES)
        box_indices = indices[..., None].expand(-1, -1, -1, self.cfg.BOX_DOF)
        scores = scores.gather(1, indices)
        boxes = boxes.gather(1, box_indices)
        return boxes, scores, indices

    def forward(self, points, features, boxes):
        refinements = self.mlp(features.permute(0, 2, 1))
        box_deltas, scores = refinements.split(1)
        return box_deltas, scores


class RoiGridPool(nn.Module):
    """
    Pools features from within proposals.
    TODO: I think must be misunderstanding dimensions claimed in paper.
        If sample 216 gridpoints in each proposal, and keypoint features
        are of dim 256, and gridpoint features are vectorized before linear layer,
        causes 216 * 256 * 256 parameters in reduction...
    TODO: Document input and output sizes.
    """

    def __init__(self, cfg):
        super(RoiGridPool, self).__init__()
        self.pnet = self.build_pointnet(cfg)
        self.reduction = MLP(cfg.GRIDPOOL.MLPS_REDUCTION)
        self.cfg = cfg

    def build_pointnet(self, cfg):
        """Copy channel list because PointNet modifies it in-place."""
        pnet = PointnetSAModuleMSG(npoint=-1, radii=cfg.GRIDPOOL.RADII_PN, nsamples=cfg.SAMPLES_PN, mlps=deepcopy(cfg.GRIDPOOL.MLPS_PN), use_xyz=True)
        return pnet

    def rotate_z(self, points, theta):
        """
        Rotate points by theta around z-axis.
        :points (b, n, m, 3)
        :theta (b, n)
        :return (b, n, m, 3)
        """
        b, n, m, _ = points.shape
        theta = theta.unsqueeze(-1).expand(-1, -1, m)
        xy, z = torch.split(points, [2, 1], dim=-1)
        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack((c, -s, s, c), dim=-1).view(b, n, m, 2, 2)
        xy = torch.matmul(R, xy.unsqueeze(-1))
        xyz = torch.cat((xy.squeeze(-1), z), dim=-1)
        return xyz

    def sample_gridpoints(self, boxes):
        """
        Sample axis-aligned points, then rotate.
        :return (b, n, ng, 3)
        """
        b, n, _ = boxes.shape
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        gridpoints = boxes[:, :, (None), 3:6] * (torch.rand((b, n, m, 3), device=boxes.device) - 0.5)
        gridpoints = boxes[:, :, (None), 0:3] + self.rotate_z(gridpoints, boxes[..., -1])
        return gridpoints

    def forward(self, proposals, keypoint_xyz, keypoint_features):
        b, n, _ = proposals.shape
        m = self.cfg.GRIDPOOL.NUM_GRIDPOINTS
        gridpoints = self.sample_gridpoints(proposals).view(b, -1, 3)
        features = self.pnet(keypoint_xyz, keypoint_features, gridpoints)[1]
        features = features.view(b, -1, n, m).permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        features = self.reduction(features)
        return features


class PV_RCNN(nn.Module):
    """
    TODO: Improve docstrings.
    TODO: Some docstrings may claim incorrect dimensions.
    TODO: Figure out clean way to handle proposals_only forward.
    """

    def __init__(self, cfg):
        super(PV_RCNN, self).__init__()
        self.pnets = self.build_pointnets(cfg)
        self.roi_grid_pool = RoiGridPool(cfg)
        self.vfe = VoxelFeatureExtractor()
        self.cnn = CNN_FACTORY[cfg.CNN](cfg)
        self.bev = BEVFeatureGatherer(cfg, self.cnn.voxel_offset, self.cnn.base_voxel_size)
        self.proposal_layer = ProposalLayer(cfg)
        self.refinement_layer = RefinementLayer(cfg)
        self.cfg = cfg

    def build_pointnets(self, cfg):
        """Copy list because PointNet modifies it in-place."""
        pnets = []
        for i, mlps in enumerate(cfg.PSA.MLPS):
            pnets += [PointnetSAModuleMSG(npoint=-1, radii=cfg.PSA.RADII[i], nsamples=cfg.SAMPLES_PN, mlps=deepcopy(mlps), use_xyz=True)]
        return nn.Sequential(*pnets)

    def sample_keypoints(self, points):
        """
        fps expects points shape (B, N, 3)
        fps returns indices shape (B, K)
        gather expects features shape (B, C, N)
        """
        points = points[(...), :3].contiguous()
        indices = furthest_point_sample(points, self.cfg.NUM_KEYPOINTS)
        keypoints = gather_operation(points.transpose(1, 2).contiguous(), indices)
        keypoints = keypoints.transpose(1, 2).contiguous()
        return keypoints

    def _pointnets(self, cnn_out, keypoint_xyz):
        """xyz (B, N, 3) | features (B, N, C) | new_xyz (B, M, C) | return (B, M, Co)"""
        pnet_out = []
        for (voxel_xyz, voxel_features), pnet in zip(cnn_out, self.pnets):
            voxel_xyz = voxel_xyz.contiguous()
            voxel_features = voxel_features.transpose(1, 2).contiguous()
            out = pnet(voxel_xyz, voxel_features, keypoint_xyz)[1]
            pnet_out += [out]
        return pnet_out

    def point_feature_extract(self, item, cnn_features, bev_map):
        points_split = torch.split(item['points'], [3, 1], dim=-1)
        cnn_features = [points_split] + cnn_features
        point_features = self._pointnets(cnn_features, item['keypoints'])
        bev_features = self.bev(bev_map, item['keypoints'])
        point_features = torch.cat(point_features + [bev_features], dim=1)
        return point_features

    def proposal(self, item):
        item['keypoints'] = self.sample_keypoints(item['points'])
        features = self.vfe(item['features'], item['occupancy'])
        cnn_features, bev_map = self.cnn(features, item['coordinates'], item['batch_size'])
        scores, boxes = self.proposal_layer(bev_map)
        item.update(dict(P_cls=scores, P_reg=boxes))
        return item

    def forward(self, item):
        raise NotImplementedError


def sigmoid_focal_loss(inputs, targets, alpha: float=0.25, gamma: float=2, reduction: str='none'):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The prediction logits for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


class ProposalLoss(nn.Module):
    """
    Notation: (P_i, G_i, M_i) ~ (predicted, ground truth, mask).
    Loss is averaged by number of positive examples.
    TODO: Replace with compiled cuda focal loss.
    """

    def __init__(self, cfg):
        super(ProposalLoss, self).__init__()
        self.cfg = cfg

    def masked_sum(self, loss, mask):
        """Mask assumed to be binary."""
        mask = mask.type_as(loss)
        loss = (loss * mask).sum()
        return loss

    def reg_loss(self, P_reg, G_reg, M_reg):
        """Loss applied at all positive sites."""
        P_xyz, P_wlh, P_yaw = P_reg.split([3, 3, 1], dim=-1)
        G_xyz, G_wlh, G_yaw = G_reg.split([3, 3, 1], dim=-1)
        loss_xyz = F.smooth_l1_loss(P_xyz, G_xyz, reduction='none')
        loss_wlh = F.smooth_l1_loss(P_wlh, G_wlh, reduction='none')
        loss_yaw = F.smooth_l1_loss(P_yaw, G_yaw, reduction='none') / math.pi
        loss = self.masked_sum(loss_xyz + loss_wlh + loss_yaw, M_reg)
        return loss

    def cls_loss(self, P_cls, G_cls, M_cls):
        """Loss is applied at all non-ignore sites. Assumes logit scores."""
        loss = sigmoid_focal_loss(P_cls, G_cls.float(), reduction='none')
        loss = self.masked_sum(loss, M_cls)
        return loss

    def forward(self, item):
        keys = ['G_cls', 'M_cls', 'P_cls', 'G_reg', 'M_reg', 'P_reg']
        G_cls, M_cls, P_cls, G_reg, M_reg, P_reg = map(item.get, keys)
        normalizer = M_reg.type_as(P_reg).sum().clamp_(min=1)
        cls_loss = self.cls_loss(P_cls, G_cls, M_cls) / normalizer
        reg_loss = self.reg_loss(P_reg, G_reg, M_reg) / normalizer
        loss = cls_loss + self.cfg.TRAIN.LAMBDA * reg_loss
        losses = dict(cls_loss=cls_loss, reg_loss=reg_loss, loss=loss)
        return losses


class Middle(SpMiddleFHD):
    """Skips conversion to metric coordinates."""

    def forward(self, features, coordinates, batch_size):
        x = spconv.SparseConvTensor(features, coordinates.int(), self.grid_shape, batch_size)
        x = self.to_bev(self.blocks(x))
        return x


class RPN(nn.Module):
    """OneStage RPN from SECOND."""

    def __init__(self, C_in=128, C_up=128, C_down=128, blocks=5):
        super(RPN, self).__init__()
        self.down_block, C_in = self._make_down_block(C_in, C_down, blocks)
        self.up_block = self._make_up_block(C_in, C_up)
        self._init_weights()

    def _make_down_block(self, inplanes, planes, num_blocks, stride=1):
        block = [nn.ZeroPad2d(1), nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False), nn.BatchNorm2d(planes, eps=0.001, momentum=0.01), nn.ReLU()]
        for j in range(num_blocks):
            block += [nn.Conv2d(planes, planes, 3, padding=1, bias=False), nn.BatchNorm2d(planes, eps=0.001, momentum=0.01), nn.ReLU()]
        return nn.Sequential(*block), planes

    def _make_up_block(self, inplanes, planes, stride=1):
        block = nn.Sequential(nn.Conv2d(inplanes, planes, stride, stride=stride, bias=False), nn.BatchNorm2d(planes, eps=0.001, momentum=0.01), nn.ReLU())
        return block

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, _BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.down_block(x)
        x = self.up_block(x)
        return x


class Second(nn.Module):

    def __init__(self, cfg):
        super(Second, self).__init__()
        self.vfe = VoxelFeatureExtractor()
        self.cnn = Middle(cfg)
        self.rpn = RPN()
        self.head = ProposalLayer(cfg)
        self.cfg = cfg

    def feature_extract(self, item):
        features = self.vfe(item['features'], item['occupancy'])
        features = self.cnn(features, item['coordinates'], item['batch_size'])
        features = self.rpn(features)
        return features

    def forward(self, item):
        features = self.feature_extract(item)
        scores, boxes = self.head(features)
        item.update(dict(P_cls=scores, P_reg=boxes))
        return item

    def inference(self, item):
        features = self.feature_extract(item)
        out = self.head.inference(features, item['anchors'])
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4, 4])], {}),
     True),
    (VoxelFeatureExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_jhultman_vision3d(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

