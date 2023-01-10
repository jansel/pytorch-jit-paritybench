import sys
_module = sys.modules[__name__]
del sys
Sim3DR = _module
_init_paths = _module
lighting = _module
setup = _module
config = _module
convert_json_list_to_lmdb = _module
data_loader_lmdb = _module
data_loader_lmdb_augmenter = _module
early_stop = _module
evaluate_wider = _module
generalized_rcnn = _module
img2pose = _module
losses = _module
model_loader = _module
models = _module
rpn = _module
run_face_alignment = _module
train = _module
train_logger = _module
annotate_dataset = _module
augmentation = _module
dist = _module
face_align = _module
image_operations = _module
json_loader = _module
json_loader_300wlp = _module
pose_operations = _module
renderer = _module

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


import numpy as np


import torch


from torch.nn import MSELoss


from torch.utils.data import BatchSampler


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


from torchvision import transforms


import warnings


from collections import OrderedDict


from torch import Tensor


from torch import nn


from torch.jit.annotations import Dict


from torch.jit.annotations import List


from torch.jit.annotations import Optional


from torch.jit.annotations import Tuple


from torch.nn import DataParallel


from torch.nn import Module


from torch.nn.parallel import DistributedDataParallel


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


from itertools import chain


from itertools import repeat


import torch.nn.functional as F


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import torchvision.models.detection._utils as det_utils


from torchvision.models.detection.faster_rcnn import TwoMLPHead


from torchvision.models.detection.roi_heads import RoIHeads


from torchvision.models.detection.transform import GeneralizedRCNNTransform


from torchvision.ops import MultiScaleRoIAlign


from torchvision.ops import boxes as box_ops


import torchvision


from torch.nn import functional as F


from torchvision.models.detection import _utils as det_utils


from torchvision.models.detection.image_list import ImageList


import random


from torch import optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.tensorboard import SummaryWriter


import torch.distributed as dist


import pandas as pd


from torchvision.datasets import ImageFolder


from scipy.spatial.transform import Rotation


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN
            and computes detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs
            to feed into the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections, evaluating):
        if evaluating:
            return losses
        return detections

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        if self.training or targets is not None:
            assert targets is not None
            for target in targets:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError('Expected target boxes to be a tensorof shape [N, 4], got {:}.'.format(boxes.shape))
                else:
                    raise ValueError('Expected target boxes to be of type Tensor, got {:}.'.format(type(boxes)))
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target['boxes']
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError('All bounding boxes should have positive height and width. Found invaid box {} for target at index {}.'.format(degen_bb, target_idx))
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('RCNN always returns a (Losses, Detections) tuple in scripting')
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections, targets is not None)


class WrappedModel(Module):

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, images, targets=None):
        return self.module(images, targets)


class FastRCNNDoFPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNDoFPredictor, self).__init__()
        hidden_layer = 256
        self.dof_pred = nn.Sequential(nn.Linear(in_channels, hidden_layer), nn.BatchNorm1d(hidden_layer), nn.ReLU(), nn.Linear(hidden_layer, num_classes * 6))

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        dof = self.dof_pred(x)
        return dof


class FastRCNNClassPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNClassPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        return scores


class AnchorGenerator(nn.Module):
    __annotations__ = {'cell_anchors': Optional[List[torch.Tensor]], '_cache': Dict[str, List[torch.Tensor]]}
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device='cpu'):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return
        cell_anchors = [self.generate_anchors(sizes, aspect_ratios, dtype, device) for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [(len(s) * len(a)) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device), torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()
        return anchors


def quat_to_rotation_mat_tensor(quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w
    matrix = torch.zeros(3, 3)
    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)
    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = -x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)
    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = -x2 - y2 + z2 + w2
    return matrix


def from_rotvec_tensor(rotvec):
    norm = torch.norm(rotvec)
    small_angle = norm <= 0.001
    scale = 0
    if small_angle:
        scale = 0.5 - norm ** 2 / 48 + norm ** 4 / 3840
    else:
        scale = torch.sin(norm / 2) / norm
    quat = torch.zeros(4)
    quat[0:3] = scale * rotvec
    quat[3] = torch.cos(norm / 2)
    return quat_to_rotation_mat_tensor(quat)


def transform_points_tensor(points, pose):
    return torch.matmul(points, from_rotvec_tensor(pose[:3]).T) + pose[3:]


def plot_3d_landmark_torch(verts, campose, intrinsics):
    lm_3d_trans = transform_points_tensor(verts, campose)
    lms_3d_trans_proj = torch.matmul(intrinsics, lm_3d_trans.T).T
    lms_projected = lms_3d_trans_proj[:, :2] / lms_3d_trans_proj[:, 2].repeat(2, 1).T
    return lms_projected


def bbox_is_dict(bbox):
    if not isinstance(bbox, dict):
        temp_bbox = {}
        temp_bbox['left'] = bbox[0]
        temp_bbox['top'] = bbox[1]
        temp_bbox['right'] = bbox[2]
        temp_bbox['bottom'] = bbox[3]
        bbox = temp_bbox
    return bbox


def get_bbox_intrinsics(image_intrinsics, bbox):
    bbox_center_x = bbox['left'] + (bbox['right'] - bbox['left']) // 2
    bbox_center_y = bbox['top'] + (bbox['bottom'] - bbox['top']) // 2
    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y
    return bbox_intrinsics


def pose_full_image_to_bbox(pose, image_intrinsics, bbox):
    bbox = bbox_is_dict(bbox)
    rvec = pose[:3].copy()
    tvec = pose[3:].copy()
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)
    focal_length = image_intrinsics[0, 0]
    bbox_width = bbox['right'] - bbox['left']
    bbox_height = bbox['bottom'] - bbox['top']
    bbox_size = bbox_width + bbox_height
    projected_point = image_intrinsics.dot(tvec.T)
    tvec = projected_point.dot(np.linalg.inv(bbox_intrinsics.T))
    tvec[2] /= focal_length / bbox_size
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    projected_point = image_intrinsics.dot(rmat)
    rmat = np.linalg.inv(bbox_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()
    return np.concatenate([rvec, tvec])


def fastrcnn_loss(class_logits, class_labels, dof_regression, labels, dof_regression_targets, proposals, image_shapes, pose_mean=None, pose_stddev=None, threed_points=None):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        dof_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        dof_loss (Tensor)
        points_loss (Tensor)
    """
    img_size = [(boxes_in_image.shape[0], image_shapes[i]) for i, boxes_in_image in enumerate(proposals)]
    img_size = list(chain.from_iterable(repeat(j, i) for i, j in img_size))
    labels = torch.cat(labels, dim=0)
    class_labels = torch.cat(class_labels, dim=0)
    dof_regression_targets = torch.cat(dof_regression_targets, dim=0)
    proposals = torch.cat(proposals, dim=0)
    classification_loss = F.cross_entropy(class_logits, class_labels)
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = dof_regression.shape[0]
    dof_regression = dof_regression.reshape(N, -1, 6)
    dof_regression = dof_regression[sampled_pos_inds_subset, labels_pos]
    prop_regression = proposals[sampled_pos_inds_subset]
    dof_regression_targets = dof_regression_targets[sampled_pos_inds_subset]
    all_target_calibration_points = None
    all_pred_calibration_points = None
    for i in range(prop_regression.shape[0]):
        h, w = img_size[i]
        global_intrinsics = torch.Tensor([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        threed_points = threed_points
        h = prop_regression[i, 3] - prop_regression[i, 1]
        w = prop_regression[i, 2] - prop_regression[i, 0]
        local_intrinsics = torch.Tensor([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        local_dof_regression = dof_regression[i, :] * pose_stddev + pose_mean
        pred_calibration_points = plot_3d_landmark_torch(threed_points, local_dof_regression.float(), local_intrinsics).unsqueeze(0)
        dof_regression_targets[i, :] = torch.from_numpy(pose_full_image_to_bbox(dof_regression_targets[i, :].cpu().numpy(), global_intrinsics.cpu().numpy(), prop_regression[i, :].cpu().numpy()))
        target_calibration_points = plot_3d_landmark_torch(threed_points, dof_regression_targets[i, :], local_intrinsics).unsqueeze(0)
        if all_target_calibration_points is None:
            all_target_calibration_points = target_calibration_points
        else:
            all_target_calibration_points = torch.cat((all_target_calibration_points, target_calibration_points))
        if all_pred_calibration_points is None:
            all_pred_calibration_points = pred_calibration_points
        else:
            all_pred_calibration_points = torch.cat((all_pred_calibration_points, pred_calibration_points))
        if pose_mean is not None:
            dof_regression_targets[i, :] = (dof_regression_targets[i, :] - pose_mean) / pose_stddev
    points_loss = F.l1_loss(all_target_calibration_points, all_pred_calibration_points)
    dof_loss = F.mse_loss(dof_regression, dof_regression_targets, reduction='sum') / dof_regression.shape[0]
    return classification_loss, dof_loss, points_loss


def expand_bbox_rectangle(w, h, bbox_x_factor=2.0, bbox_y_factor=2.0, lms=None, expand_forehead=0.3, roll=0):
    min_pt_x = np.min(lms[:, 0], axis=0)
    max_pt_x = np.max(lms[:, 0], axis=0)
    min_pt_y = np.min(lms[:, 1], axis=0)
    max_pt_y = np.max(lms[:, 1], axis=0)
    bbox_size_x = int(np.max(max_pt_x - min_pt_x) * bbox_x_factor)
    center_pt_x = 0.5 * min_pt_x + 0.5 * max_pt_x
    bbox_size_y = int(np.max(max_pt_y - min_pt_y) * bbox_y_factor)
    center_pt_y = 0.5 * min_pt_y + 0.5 * max_pt_y
    bbox_min_x, bbox_max_x = center_pt_x - bbox_size_x * 0.5, center_pt_x + bbox_size_x * 0.5
    bbox_min_y, bbox_max_y = center_pt_y - bbox_size_y * 0.5, center_pt_y + bbox_size_y * 0.5
    if abs(roll) > 2.5:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_max_y += expand_forehead_size
    elif roll > 1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_max_x += expand_forehead_size
    elif roll < -1:
        expand_forehead_size = expand_forehead * np.max(max_pt_x - min_pt_x)
        bbox_min_x -= expand_forehead_size
    else:
        expand_forehead_size = expand_forehead * np.max(max_pt_y - min_pt_y)
        bbox_min_y -= expand_forehead_size
    bbox_min_x = bbox_min_x.astype(np.int32)
    bbox_max_x = bbox_max_x.astype(np.int32)
    bbox_min_y = bbox_min_y.astype(np.int32)
    bbox_max_y = bbox_max_y.astype(np.int32)
    padding_left = abs(min(bbox_min_x, 0))
    padding_top = abs(min(bbox_min_y, 0))
    padding_right = max(bbox_max_x - w, 0)
    padding_bottom = max(bbox_max_y - h, 0)
    crop_left = 0 if padding_left > 0 else bbox_min_x
    crop_top = 0 if padding_top > 0 else bbox_min_y
    crop_right = w if padding_right > 0 else bbox_max_x
    crop_bottom = h if padding_bottom > 0 else bbox_max_y
    return np.array([crop_left, crop_top, crop_right, crop_bottom])


def transform_points(points, pose):
    return points.dot(Rotation.from_rotvec(pose[:3]).as_matrix().T) + pose[3:]


def plot_3d_landmark(verts, campose, intrinsics):
    lm_3d_trans = transform_points(verts, campose)
    lms_3d_trans_proj = intrinsics.dot(lm_3d_trans.T).T
    lms_projected = lms_3d_trans_proj[:, :2] / np.tile(lms_3d_trans_proj[:, 2], (2, 1)).T
    return lms_projected, lms_3d_trans_proj


def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    bbox = bbox_is_dict(bbox)
    rvec = pose[:3].copy()
    tvec = pose[3:].copy()
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)
    focal_length = image_intrinsics[0, 0]
    bbox_width = bbox['right'] - bbox['left']
    bbox_height = bbox['bottom'] - bbox['top']
    bbox_size = bbox_width + bbox_height
    tvec[2] *= focal_length / bbox_size
    projected_point = bbox_intrinsics.dot(tvec.T)
    tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    projected_point = bbox_intrinsics.dot(rmat)
    rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()
    return np.concatenate([rvec, tvec])


def transform_pose_global_project_bbox(boxes, dofs, pose_mean, pose_stddev, image_shape, threed_68_points=None, bbox_x_factor=1.1, bbox_y_factor=1.1, expand_forehead=0.3):
    if len(dofs) == 0:
        return boxes, dofs
    device = dofs.device
    boxes = boxes.cpu().numpy()
    dofs = dofs.cpu().numpy()
    threed_68_points = threed_68_points.numpy()
    h, w = image_shape
    global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
    if threed_68_points is not None:
        threed_68_points = threed_68_points
    pose_mean = pose_mean.numpy()
    pose_stddev = pose_stddev.numpy()
    dof_mean = pose_mean
    dof_std = pose_stddev
    dofs = dofs * dof_std + dof_mean
    projected_boxes = []
    global_dofs = []
    for i in range(dofs.shape[0]):
        global_dof = pose_bbox_to_full_image(dofs[i], global_intrinsics, boxes[i])
        global_dofs.append(global_dof)
        if threed_68_points is not None:
            projected_lms, _ = plot_3d_landmark(threed_68_points, global_dof, global_intrinsics)
            projected_bbox = expand_bbox_rectangle(w, h, bbox_x_factor=bbox_x_factor, bbox_y_factor=bbox_y_factor, lms=projected_lms, roll=global_dof[2], expand_forehead=expand_forehead)
        else:
            projected_bbox = boxes[i]
        projected_boxes.append(projected_bbox)
    global_dofs = torch.from_numpy(np.asarray(global_dofs)).float()
    projected_boxes = torch.from_numpy(np.asarray(projected_boxes)).float()
    return projected_boxes, global_dofs


class DOFRoIHeads(RoIHeads):

    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, out_channels, mask_roi_pool=None, mask_head=None, mask_predictor=None, keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None, pose_mean=None, pose_stddev=None, threed_68_points=None, threed_5_points=None, bbox_x_factor=1.1, bbox_y_factor=1.1, expand_forehead=0.3):
        super(RoIHeads, self).__init__()
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        if bbox_reg_weights is None:
            bbox_reg_weights = 10.0, 10.0, 5.0, 5.0
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        num_classes = 2
        self.class_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        self.class_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        self.class_predictor = FastRCNNClassPredictor(representation_size, num_classes)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor
        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor
        self.pose_mean = pose_mean
        self.pose_stddev = pose_stddev
        self.threed_68_points = threed_68_points
        self.threed_5_points = threed_5_points
        self.bbox_x_factor = bbox_x_factor
        self.bbox_y_factor = bbox_y_factor
        self.expand_forehead = expand_forehead

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        gt_dofs = [t['dofs'] for t in targets]
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        matched_gt_dofs = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            gt_boxes_in_image = gt_boxes[img_id]
            gt_dofs_in_image = gt_dofs[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            if gt_dofs_in_image.numel() == 0:
                gt_dofs_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            matched_gt_dofs.append(gt_dofs_in_image[matched_idxs[img_id]])
        dof_regression_targets = matched_gt_dofs
        box_regression_targets = matched_gt_boxes
        return proposals, matched_idxs, labels, dof_regression_targets, box_regression_targets

    def decode(self, rel_codes, boxes):
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        pred_boxes = self.decode_single(rel_codes.reshape(box_sum, -1), concat_boxes)
        return pred_boxes.reshape(box_sum, -1, 6)

    def postprocess_detections(self, class_logits, dof_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = torch.cat(proposals, dim=0)
        N = dof_regression.shape[0]
        pred_boxes = pred_boxes.reshape(N, -1, 4)
        pred_dofs = dof_regression.reshape(N, -1, 6)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_dofs_list = pred_dofs.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_dofs = []
        for boxes, dofs, scores, image_shape in zip(pred_boxes_list, pred_dofs_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            dofs = dofs[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            boxes = boxes.reshape(-1, 4)
            dofs = dofs.reshape(-1, 6)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, dofs, scores, labels = boxes[inds], dofs[inds], scores[inds], labels[inds]
            keep = box_ops.remove_small_boxes(boxes, min_size=0.01)
            boxes, dofs, scores, labels = boxes[keep], dofs[keep], scores[keep], labels[keep]
            boxes, dofs = transform_pose_global_project_bbox(boxes, dofs, self.pose_mean, self.pose_stddev, image_shape, self.threed_68_points, bbox_x_factor=self.bbox_x_factor, bbox_y_factor=self.bbox_y_factor, expand_forehead=self.expand_forehead)
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            boxes, dofs, scores, labels = boxes[keep], dofs[keep], scores[keep], labels[keep]
            keep = keep[:self.detections_per_img]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_dofs.append(dofs)
        return all_boxes, all_dofs, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = torch.float, torch.double, torch.half
                assert t['boxes'].dtype in floating_point_types, 'target boxes must of float type'
                assert t['labels'].dtype == torch.int64, 'target labels must of int64 type'
        if self.training or targets is not None:
            proposals, matched_idxs, labels, regression_targets, regression_targets_box = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        if self.training or targets is not None:
            num_images = len(proposals)
            dof_proposals = []
            dof_regression_targets = []
            box_regression_targets = []
            dof_labels = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                dof_proposals.append(proposals[img_id][pos])
                dof_regression_targets.append(regression_targets[img_id][pos])
                box_regression_targets.append(regression_targets_box[img_id][pos])
                dof_labels.append(labels[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
            box_features = self.box_roi_pool(features, dof_proposals, image_shapes)
            box_features = self.box_head(box_features)
            dof_regression = self.box_predictor(box_features)
            class_features = self.class_roi_pool(features, proposals, image_shapes)
            class_features = self.class_head(class_features)
            class_logits = self.class_predictor(class_features)
            result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        else:
            num_images = len(proposals)
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            dof_regression = self.box_predictor(box_features)
            class_features = self.class_roi_pool(features, proposals, image_shapes)
            class_features = self.class_head(class_features)
            class_logits = self.class_predictor(class_features)
            result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training or targets is not None:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_dof_reg, loss_points = fastrcnn_loss(class_logits, labels, dof_regression, dof_labels, dof_regression_targets, dof_proposals, image_shapes, self.pose_mean, self.pose_stddev, self.threed_5_points)
            losses = {'loss_classifier': loss_classifier, 'loss_dof_reg': loss_dof_reg, 'loss_points': loss_points}
        else:
            boxes, dofs, scores, labels = self.postprocess_detections(class_logits, dof_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({'boxes': boxes[i], 'labels': labels[i], 'scores': scores[i], 'dofs': dofs[i]})
        return result, losses


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat((torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype), num_anchors), 0))
    return num_anchors, pre_nms_top_n


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors
            for a set of feature maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so
            that they can be considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so
            that they can be considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during
            training of the RPN for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch
            during training of the RPN
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS.
            It should contain two fields: training and testing, to allow for different
            values depending on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS.
            It should contain two fields: training and testing, to allow for different
            values depending on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {'box_coder': det_utils.BoxCoder, 'proposal_matcher': det_utils.Matcher, 'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler, 'pre_nms_top_n': Dict[str, int], 'post_nms_top_n': Dict[str, int]}

    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 0.001

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        Arguments:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = det_utils.smooth_l1_loss(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1 / 9, size_average=False) / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [(s[0] * s[1] * s[2]) for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        losses = {}
        if self.training or targets is not None:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {'loss_objectness': loss_objectness, 'loss_rpn_box_reg': loss_rpn_box_reg}
        return boxes, losses


class FasterDoFRCNN(GeneralizedRCNN):

    def __init__(self, backbone, num_classes=None, min_size=800, max_size=1333, image_mean=None, image_std=None, rpn_anchor_generator=None, rpn_head=None, rpn_pre_nms_top_n_train=6000, rpn_pre_nms_top_n_test=6000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.4, rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, box_roi_pool=None, box_head=None, box_predictor=None, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=1000, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None, pose_mean=None, pose_stddev=None, threed_68_points=None, threed_5_points=None, bbox_x_factor=1.1, bbox_y_factor=1.1, expand_forehead=0.3):
        if not hasattr(backbone, 'out_channels'):
            raise ValueError('backbone should contain an attribute out_channels specifying the number of output channels (assumed to be the same for all the levels)')
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError('num_classes should be None when box_predictor is specified')
        elif box_predictor is None:
            raise ValueError('num_classes should not be None when box_predictor is not specified')
        out_channels = backbone.out_channels
        if rpn_anchor_generator is None:
            anchor_sizes = (16,), (32,), (64,), (128,), (256,), (512,)
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = {'training': rpn_pre_nms_top_n_train, 'testing': rpn_pre_nms_top_n_test}
        rpn_post_nms_top_n = {'training': rpn_post_nms_top_n_train, 'testing': rpn_post_nms_top_n_test}
        rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh, rpn_batch_size_per_image, rpn_positive_fraction, rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNDoFPredictor(representation_size, num_classes)
        roi_heads = DOFRoIHeads(box_roi_pool, box_head, box_predictor, box_fg_iou_thresh, box_bg_iou_thresh, box_batch_size_per_image, box_positive_fraction, bbox_reg_weights, box_score_thresh, box_nms_thresh, box_detections_per_img, out_channels, pose_mean=pose_mean, pose_stddev=pose_stddev, threed_68_points=threed_68_points, threed_5_points=threed_5_points, bbox_x_factor=bbox_x_factor, bbox_y_factor=bbox_y_factor, expand_forehead=expand_forehead)
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super(FasterDoFRCNN, self).__init__(backbone, rpn, roi_heads, transform)

    def set_max_min_size(self, max_size, min_size):
        self.min_size = min_size,
        self.max_size = max_size
        self.transform.min_size = self.min_size
        self.transform.max_size = self.max_size


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FastRCNNClassPredictor,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (FastRCNNDoFPredictor,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (RPNHead,
     lambda: ([], {'in_channels': 4, 'num_anchors': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_vitoralbiero_img2pose(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

