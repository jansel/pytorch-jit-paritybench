import sys
_module = sys.modules[__name__]
del sys
infer = _module
utils = _module
coco_eval = _module
coco_utils = _module
config = _module
core = _module
_utils = _module
backbone_utils = _module
faster_rcnn = _module
generalized_rcnn = _module
image_list = _module
keypoint_rcnn = _module
mask_rcnn = _module
roi_heads = _module
rpn = _module
transform = _module
datasets = _module
coco = _module
layout = _module
pen = _module
publaynet = _module
tb_detection = _module
teenet = _module
engine = _module
evaluate = _module
group_by_aspect_ratio = _module
infer = _module
main_publaynet = _module
models = _module
scripts = _module
run_sagemaker = _module
start_training = _module
wait_sagemaker = _module
transforms = _module
utils = _module

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


import random


import torch


import torchvision


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from torchvision.transforms import transforms


import numpy as np


import copy


import time


import torch._six


from collections import defaultdict


import torch.utils.data


import math


from torch.jit.annotations import List


from torch.jit.annotations import Tuple


from torch import Tensor


from collections import OrderedDict


from torch import nn


from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


from torch.jit.annotations import Dict


from torchvision.ops import misc as misc_nn_ops


from torchvision.models import resnet


import torch.nn.functional as F


from torchvision.ops import MultiScaleRoIAlign


import warnings


from torch.jit.annotations import Optional


from torchvision.ops import boxes as box_ops


from torchvision.ops import roi_align


from torch.nn import functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision import utils


from torch.utils.data.sampler import Sampler


import logging


from torchvision.datasets.coco import CocoDetection


import re


import torchvision.models.detection.mask_rcnn


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data.distributed


from torch.utils.tensorboard import SummaryWriter


import torchvision.datasets as datasets


from torch.utils.data import BatchSampler


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import MultiStepLR


from torch.nn.parallel import DistributedDataParallel


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import BatchSampler


from torch.utils.model_zoo import tqdm


from torchvision.models.detection.rpn import AnchorGenerator


from torchvision.models.detection import MaskRCNN


from torchvision.models.detection import maskrcnn_resnet50_fpn


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.transforms import functional as F


from collections import deque


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {'return_layers': Dict[str, str]}

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses
        return detections

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed')
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
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
            return self.eager_outputs(losses, detections)


class KeypointRCNNHeads(nn.Sequential):

    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for l in layers:
            d.append(misc_nn_ops.Conv2d(next_feature, l, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = l
        super(KeypointRCNNHeads, self).__init__(*d)
        for m in self.children():
            if isinstance(m, misc_nn_ops.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class KeypointRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = misc_nn_ops.ConvTranspose2d(input_features, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1)
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = misc_nn_ops.interpolate(x, scale_factor=float(self.up_scale), mode='bilinear', align_corners=False)
        return x


class MaskRCNNHeads(nn.Sequential):

    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = misc_nn_ops.Conv2d(next_feature, layer_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class MaskRCNNPredictor(nn.Sequential):

    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([('conv5_mask', misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)), ('relu', nn.ReLU(inplace=True)), ('mask_fcn_logits', misc_nn_ops.Conv2d(dim_reduced, num_classes, 1, 1, 0))]))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = F.smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos], regression_targets[sampled_pos_inds_subset], reduction='sum')
    box_loss = box_loss / labels.numel()
    return classification_loss, box_loss


def _onnx_heatmaps_to_keypoints(maps, maps_i, roi_map_width, roi_map_height, widths_i, heights_i, offset_x_i, offset_y_i):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)
    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height
    roi_map = torch.nn.functional.interpolate(maps_i[None], size=(int(roi_map_height), int(roi_map_width)), mode='bicubic', align_corners=False)[0]
    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)
    x_int = pos % w
    y_int = (pos - x_int) / w
    x = (torch.tensor(0.5, dtype=torch.float32) + x_int) * width_correction
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int) * height_correction
    xy_preds_i_0 = x + offset_x_i
    xy_preds_i_1 = y + offset_y_i
    xy_preds_i_2 = torch.ones(xy_preds_i_1.shape, dtype=torch.float32)
    xy_preds_i = torch.stack([xy_preds_i_0, xy_preds_i_1, xy_preds_i_2], 0)
    end_scores_i = roi_map.index_select(1, y_int).index_select(2, x_int)[:num_keypoints, 0, 0]
    return xy_preds_i, end_scores_i


@torch.jit.script
def _onnx_heatmaps_to_keypoints_loop(maps, rois, widths_ceil, heights_ceil, widths, heights, offset_x, offset_y, num_keypoints):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(maps, maps[i], widths_ceil[i], heights_ceil[i], widths[i], heights[i], offset_x[i], offset_y[i])
        xy_preds = torch.cat((xy_preds, xy_preds_i.unsqueeze(0)), 0)
        end_scores = torch.cat((end_scores, end_scores_i.unsqueeze(0)), 0)
    return xy_preds, end_scores


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()
    num_keypoints = maps.shape[1]
    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(maps, rois, widths_ceil, heights_ceil, widths, heights, offset_x, offset_y, torch.scalar_tensor(num_keypoints, dtype=torch.int64))
        return xy_preds.permute(0, 2, 1), end_scores
    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = torch.nn.functional.interpolate(maps[i][None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[0]
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)
        x_int = pos % w
        y_int = (pos - x_int) // w
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints), y_int, x_int]
    return xy_preds.permute(0, 2, 1), end_scores


def keypointrcnn_inference(x, boxes):
    kp_probs = []
    kp_scores = []
    boxes_per_image = [box.size(0) for box in boxes]
    if len(boxes_per_image) == 1:
        kp_prob, scores = heatmaps_to_keypoints(x, boxes[0])
        return [kp_prob], [scores]
    x2 = x.split(boxes_per_image, dim=0)
    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)
    return kp_probs, kp_scores


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])
    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]
    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    x[x_boundary_inds] = torch.tensor(heatmap_size - 1)
    y[y_boundary_inds] = torch.tensor(heatmap_size - 1)
    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()
    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid
    return heatmaps, valid


def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0)
    valid = torch.nonzero(valid).squeeze(1)
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0
    keypoint_logits = keypoint_logits.view(N * K, H * W)
    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


def maskrcnn_inference(x, labels):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()
    num_masks = x.shape[0]
    boxes_per_image = [len(l) for l in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    if len(boxes_per_image) == 1:
        mask_prob_list = [mask_prob]
    else:
        mask_prob_list = mask_prob.split(boxes_per_image, dim=0)
    return mask_prob_list


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None]
    return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """
    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets)
    return mask_loss


class AnchorGenerator(nn.Module):
    __annotations__ = {'cell_anchors': Optional[List[torch.Tensor]], '_cache': Dict[str, List[torch.Tensor]]}
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

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
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device), torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
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
        for l in self.children():
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

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
    pre_nms_top_n = torch.min(torch.cat((torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype), num_anchors), 0).to(torch.int32))
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


def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)
    w = box[2] - box[0] + one
    h = box[3] - box[1] + one
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))
    mask = torch.nn.functional.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))
    unpaded_im_mask = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0]:x_1 - box[0]]
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0, unpaded_im_mask, zeros_y1), 0)[0:im_h, :]
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0, concat_0, zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit.script
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


def _onnx_expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5
    w_half = w_half * scale
    h_half = h_half * scale
    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


def expand_boxes(boxes, scale):
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
    h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
    x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
    y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    return torch.tensor(M + 2 * padding) / torch.tensor(M)


def expand_masks(mask, padding):
    M = mask.shape[-1]
    if torch._C._get_tracing_state():
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w):
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = misc_nn_ops.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]
    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - box[1]:y_1 - box[1], x_0 - box[0]:x_1 - box[0]]
    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale)
    im_h, im_w = img_shape
    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(masks, boxes, torch.scalar_tensor(im_h, dtype=torch.int64), torch.scalar_tensor(im_w, dtype=torch.int64))[:, None]
    res = [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


def resize_boxes(boxes, original_size, new_size):
    ratios = [(torch.tensor(s, dtype=torch.float32, device=boxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)) for s, s_orig in zip(new_size, original_size)]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def resize_keypoints(keypoints, original_size, new_size):
    ratios = [(torch.tensor(s, dtype=torch.float32, device=keypoints.device) / torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)) for s, s_orig in zip(new_size, original_size)]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = min_size,
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            if image.dim() != 3:
                raise ValueError('images is expected to be a list of 3d tensors of shape [C, H, W], got {}'.format(image.shape))
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, l):
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(l))).item())
        return l[index]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        if target is None:
            return image, target
        bbox = target['boxes']
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target['boxes'] = bbox
        if 'masks' in target:
            mask = target['masks']
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target['masks'] = mask
        if 'keypoints' in target:
            keypoints = target['keypoints']
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target['keypoints'] = keypoints
        return image, target

    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32))
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = torch.ceil(max_size[1].to(torch.float32) / stride) * stride
        max_size[2] = torch.ceil(max_size[2].to(torch.float32) / stride) * stride
        max_size = tuple(max_size)
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)
        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        if torchvision._is_tracing():
            return self._onnx_batch_images(images, size_divisible)
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes
            if 'masks' in pred:
                masks = pred['masks']
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]['masks'] = masks
            if 'keypoints' in pred:
                keypoints = pred['keypoints']
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]['keypoints'] = keypoints
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += '{0}Normalize(mean={1}, std={2})'.format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size, self.max_size)
        format_string += '\n)'
        return format_string


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FastRCNNPredictor,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_phamquiluan_PubLayNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

