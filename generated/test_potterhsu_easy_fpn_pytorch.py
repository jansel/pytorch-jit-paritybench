import sys
_module = sys.modules[__name__]
del sys
base = _module
resnet101 = _module
resnet18 = _module
resnet50 = _module
bbox = _module
config = _module
eval_config = _module
train_config = _module
base = _module
coco2017 = _module
coco2017_animal = _module
coco2017_car = _module
coco2017_person = _module
voc2007 = _module
voc2007_cat_dog = _module
eval = _module
evaluator = _module
infer = _module
logger = _module
model = _module
build = _module
nms = _module
test_nms = _module
realtime = _module
build = _module
crop_and_resize = _module
roi_align = _module
wrapper = _module
region_proposal_network = _module
train = _module
voc_eval = _module

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


from typing import Tuple


from typing import Type


from typing import NamedTuple


from torch import nn


import torchvision


from typing import Callable


from torch import Tensor


import torch


from enum import Enum


from typing import List


import torch.utils.data.dataset


from torchvision.transforms import transforms


import random


from typing import Dict


from torchvision.datasets import CocoDetection


import numpy as np


import torch.utils.data


from torch.utils.data import DataLoader


from typing import Union


from torch.nn import functional as F


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import time


import math


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Function


import uuid


from collections import deque


from typing import Optional


from torch import optim


from torch.optim.lr_scheduler import MultiStepLR


class BBox(object):

    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) ->str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(self.left, self.top, self.right, self.bottom)

    def tolist(self):
        return [self.left, self.top, self.right, self.bottom]

    @staticmethod
    def to_center_base(bboxes: Tensor):
        return torch.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2, bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], dim=1)

    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) ->Tensor:
        return torch.stack([center_based_bboxes[:, 0] - center_based_bboxes[:, 2] / 2, center_based_bboxes[:, 1] - center_based_bboxes[:, 3] / 2, center_based_bboxes[:, 0] + center_based_bboxes[:, 2] / 2, center_based_bboxes[:, 1] + center_based_bboxes[:, 3] / 2], dim=1)

    @staticmethod
    def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) ->Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
        transformers = torch.stack([(center_based_dst_bboxes[:, 0] - center_based_src_bboxes[:, 0]) / center_based_dst_bboxes[:, 2], (center_based_dst_bboxes[:, 1] - center_based_src_bboxes[:, 1]) / center_based_dst_bboxes[:, 3], torch.log(center_based_dst_bboxes[:, 2] / center_based_src_bboxes[:, 2]), torch.log(center_based_dst_bboxes[:, 3] / center_based_src_bboxes[:, 3])], dim=1)
        return transformers

    @staticmethod
    def apply_transformer(src_bboxes: Tensor, transformers: Tensor) ->Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = torch.stack([transformers[:, 0] * center_based_src_bboxes[:, 2] + center_based_src_bboxes[:, 0], transformers[:, 1] * center_based_src_bboxes[:, 3] + center_based_src_bboxes[:, 1], torch.exp(transformers[:, 2]) * center_based_src_bboxes[:, 2], torch.exp(transformers[:, 3]) * center_based_src_bboxes[:, 3]], dim=1)
        dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
        return dst_bboxes

    @staticmethod
    def iou(source: Tensor, other: Tensor) ->Tensor:
        source = source.repeat(other.shape[0], 1, 1).permute(1, 0, 2)
        other = other.repeat(source.shape[0], 1, 1)
        source_area = (source[:, :, 2] - source[:, :, 0]) * (source[:, :, 3] - source[:, :, 1])
        other_area = (other[:, :, 2] - other[:, :, 0]) * (other[:, :, 3] - other[:, :, 1])
        intersection_left = torch.max(source[:, :, 0], other[:, :, 0])
        intersection_top = torch.max(source[:, :, 1], other[:, :, 1])
        intersection_right = torch.min(source[:, :, 2], other[:, :, 2])
        intersection_bottom = torch.min(source[:, :, 3], other[:, :, 3])
        intersection_width = torch.clamp(intersection_right - intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom - intersection_top, min=0)
        intersection_area = intersection_width * intersection_height
        return intersection_area / (source_area + other_area - intersection_area)

    @staticmethod
    def inside(source: Tensor, other: Tensor) ->bool:
        source = source.repeat(other.shape[0], 1, 1).permute(1, 0, 2)
        other = other.repeat(source.shape[0], 1, 1)
        return (source[:, :, 0] >= other[:, :, 0]) * (source[:, :, 1] >= other[:, :, 1]) * (source[:, :, 2] <= other[:, :, 2]) * (source[:, :, 3] <= other[:, :, 3])

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) ->Tensor:
        return torch.stack([torch.clamp(bboxes[:, 0], min=left, max=right), torch.clamp(bboxes[:, 1], min=top, max=bottom), torch.clamp(bboxes[:, 2], min=left, max=right), torch.clamp(bboxes[:, 3], min=top, max=bottom)], dim=1)


class NMS(object):

    @staticmethod
    def suppress(sorted_bboxes: Tensor, threshold: float) ->Tensor:
        kept_indices = torch.tensor([], dtype=torch.long)
        nms.suppress(sorted_bboxes.contiguous(), threshold, kept_indices)
        return kept_indices


class RegionProposalNetwork(nn.Module):

    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int, int]], anchor_scales: List[int], pre_nms_top_n: int, post_nms_top_n: int):
        super().__init__()
        self._features = nn.Sequential(nn.Conv2d(in_channels=num_features_out, out_channels=512, kernel_size=3, padding=1), nn.ReLU())
        self._anchor_ratios = anchor_ratios
        self._anchor_scales = anchor_scales
        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_scales = len(self._anchor_scales)
        num_anchors = num_anchor_ratios * num_anchor_scales
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self._objectness = nn.Conv2d(in_channels=512, out_channels=num_anchors * 2, kernel_size=1)
        self._transformer = nn.Conv2d(in_channels=512, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor, image_width: int, image_height: int) ->Tuple[Tensor, Tensor]:
        features = self._features(features)
        objectnesses = self._objectness(features)
        transformers = self._transformer(features)
        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        transformers = transformers.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        return objectnesses, transformers

    def sample(self, anchor_bboxes: Tensor, gt_bboxes: Tensor, image_width: int, image_height: int) ->Tuple[Tensor, Tensor, Tensor, Tensor]:
        sample_fg_indices = torch.arange(end=len(anchor_bboxes), dtype=torch.long)
        sample_selected_indices = torch.arange(end=len(anchor_bboxes), dtype=torch.long)
        anchor_bboxes = anchor_bboxes.cpu()
        gt_bboxes = gt_bboxes.cpu()
        boundary = torch.tensor(BBox(0, 0, image_width, image_height).tolist(), dtype=torch.float)
        inside_indices = BBox.inside(anchor_bboxes, boundary.unsqueeze(dim=0)).squeeze().nonzero().view(-1)
        anchor_bboxes = anchor_bboxes[inside_indices]
        sample_fg_indices = sample_fg_indices[inside_indices]
        sample_selected_indices = sample_selected_indices[inside_indices]
        labels = torch.ones(len(anchor_bboxes), dtype=torch.long) * -1
        ious = BBox.iou(anchor_bboxes, gt_bboxes)
        anchor_max_ious, anchor_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)
        anchor_additions = (ious == gt_max_ious).nonzero()[:, 0]
        labels[anchor_max_ious < 0.3] = 0
        labels[anchor_additions] = 1
        labels[anchor_max_ious >= 0.7] = 1
        fg_indices = (labels == 1).nonzero().view(-1)
        bg_indices = (labels == 0).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 - len(fg_indices)]]
        selected_indices = torch.cat([fg_indices, bg_indices])
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        gt_anchor_objectnesses = labels[selected_indices]
        gt_bboxes = gt_bboxes[anchor_assignments[fg_indices]]
        anchor_bboxes = anchor_bboxes[fg_indices]
        gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)
        gt_anchor_objectnesses = gt_anchor_objectnesses
        gt_anchor_transformers = gt_anchor_transformers
        sample_fg_indices = sample_fg_indices[fg_indices]
        sample_selected_indices = sample_selected_indices[selected_indices]
        return sample_fg_indices, sample_selected_indices, gt_anchor_objectnesses, gt_anchor_transformers

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor, gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor) ->Tuple[Tensor, Tensor]:
        cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=gt_anchor_objectnesses)
        smooth_l1_loss = F.smooth_l1_loss(input=anchor_transformers, target=gt_anchor_transformers, reduction='sum')
        smooth_l1_loss /= len(gt_anchor_transformers)
        return cross_entropy, smooth_l1_loss

    def generate_anchors(self, image_width: int, image_height: int, num_x_anchors: int, num_y_anchors: int, anchor_size: int) ->Tensor:
        center_ys = np.linspace(start=0, stop=image_height, num=num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=num_x_anchors + 2)[1:-1]
        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, 0] / ratios[:, 1]
        scales = np.array(self._anchor_scales)
        center_ys, center_xs, ratios, scales = np.meshgrid(center_ys, center_xs, ratios, scales, indexing='ij')
        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        scales = scales.reshape(-1)
        widths = anchor_size * scales * np.sqrt(1 / ratios)
        heights = anchor_size * scales * np.sqrt(ratios)
        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths, heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(center_based_anchor_bboxes).float()
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)
        return anchor_bboxes

    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses: Tensor, transformers: Tensor, image_width: int, image_height: int) ->Tensor:
        proposal_score = objectnesses[:, 1]
        _, sorted_indices = torch.sort(proposal_score, dim=0, descending=True)
        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_bboxes = anchor_bboxes[sorted_indices]
        proposal_bboxes = BBox.apply_transformer(sorted_anchor_bboxes, sorted_transformers.detach())
        proposal_bboxes = BBox.clip(proposal_bboxes, 0, 0, image_width, image_height)
        proposal_bboxes = proposal_bboxes[:self._pre_nms_top_n]
        kept_indices = NMS.suppress(proposal_bboxes, threshold=0.7)
        proposal_bboxes = proposal_bboxes[kept_indices]
        proposal_bboxes = proposal_bboxes[:self._post_nms_top_n]
        return proposal_bboxes


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)
        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(image, boxes, box_ind, self.extrapolation_value, self.crop_height, self.crop_width, crops)
        else:
            _backend.crop_and_resize_forward(image, boxes, box_ind, self.extrapolation_value, self.crop_height, self.crop_width, crops)
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)
        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)
        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(grad_outputs, boxes, box_ind, grad_image)
        else:
            _backend.crop_and_resize_backward(grad_outputs, boxes, box_ind, grad_image)
        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(image, boxes, box_ind)


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]
        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)
            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self.crop_height - 1) / float(image_height - 1)
            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)
        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction(self.crop_height, self.crop_width, self.extrapolation_value)(featuremap, boxes, box_ind)

