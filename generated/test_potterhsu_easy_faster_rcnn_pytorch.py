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
functional = _module
lr_scheduler = _module
infer = _module
infer_stream = _module
infer_websocket = _module
logger = _module
model = _module
pooler = _module
region_proposal_network = _module
nms = _module
roi_align = _module
setup = _module
test_nms = _module
train = _module
voc_eval = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from typing import Tuple


from typing import Type


from torch import nn


import random


from enum import Enum


from typing import List


from typing import Iterator


import torch.utils.data.dataset


import torch.utils.data.sampler


from torch import Tensor


from torch.nn import functional as F


from typing import Union


from typing import Optional


import torch


from torch.optim import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


import numpy as np


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import time


import uuid


from collections import deque


import torch.nn as nn


from torch import optim


from torch.utils.data import DataLoader


class BBox(object):

    def __init__(self, left: float, top: float, right: float, bottom: float):
        super().__init__()
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) ->str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}]'.format(self.
            left, self.top, self.right, self.bottom)

    def tolist(self) ->List[float]:
        return [self.left, self.top, self.right, self.bottom]

    @staticmethod
    def to_center_base(bboxes: Tensor) ->Tensor:
        return torch.stack([(bboxes[..., 0] + bboxes[..., 2]) / 2, (bboxes[
            ..., 1] + bboxes[..., 3]) / 2, bboxes[..., 2] - bboxes[..., 0],
            bboxes[..., 3] - bboxes[..., 1]], dim=-1)

    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) ->Tensor:
        return torch.stack([center_based_bboxes[..., 0] - 
            center_based_bboxes[..., 2] / 2, center_based_bboxes[..., 1] - 
            center_based_bboxes[..., 3] / 2, center_based_bboxes[..., 0] + 
            center_based_bboxes[..., 2] / 2, center_based_bboxes[..., 1] + 
            center_based_bboxes[..., 3] / 2], dim=-1)

    @staticmethod
    def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) ->Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
        transformers = torch.stack([(center_based_dst_bboxes[..., 0] -
            center_based_src_bboxes[..., 0]) / center_based_src_bboxes[...,
            2], (center_based_dst_bboxes[..., 1] - center_based_src_bboxes[
            ..., 1]) / center_based_src_bboxes[..., 3], torch.log(
            center_based_dst_bboxes[..., 2] / center_based_src_bboxes[..., 
            2]), torch.log(center_based_dst_bboxes[..., 3] /
            center_based_src_bboxes[..., 3])], dim=-1)
        return transformers

    @staticmethod
    def apply_transformer(src_bboxes: Tensor, transformers: Tensor) ->Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = torch.stack([transformers[..., 0] *
            center_based_src_bboxes[..., 2] + center_based_src_bboxes[..., 
            0], transformers[..., 1] * center_based_src_bboxes[..., 3] +
            center_based_src_bboxes[..., 1], torch.exp(transformers[..., 2]
            ) * center_based_src_bboxes[..., 2], torch.exp(transformers[...,
            3]) * center_based_src_bboxes[..., 3]], dim=-1)
        dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
        return dst_bboxes

    @staticmethod
    def iou(source: Tensor, other: Tensor) ->Tensor:
        source, other = source.unsqueeze(dim=-2).repeat(1, 1, other.shape[-
            2], 1), other.unsqueeze(dim=-3).repeat(1, source.shape[-2], 1, 1)
        source_area = (source[..., 2] - source[..., 0]) * (source[..., 3] -
            source[..., 1])
        other_area = (other[..., 2] - other[..., 0]) * (other[..., 3] -
            other[..., 1])
        intersection_left = torch.max(source[..., 0], other[..., 0])
        intersection_top = torch.max(source[..., 1], other[..., 1])
        intersection_right = torch.min(source[..., 2], other[..., 2])
        intersection_bottom = torch.min(source[..., 3], other[..., 3])
        intersection_width = torch.clamp(intersection_right -
            intersection_left, min=0)
        intersection_height = torch.clamp(intersection_bottom -
            intersection_top, min=0)
        intersection_area = intersection_width * intersection_height
        return intersection_area / (source_area + other_area -
            intersection_area)

    @staticmethod
    def inside(bboxes: Tensor, left: float, top: float, right: float,
        bottom: float) ->Tensor:
        return (bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) * (bboxes
            [..., 2] <= right) * (bboxes[..., 3] <= bottom)

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom:
        float) ->Tensor:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)
        return bboxes


class Pooler(object):


    class Mode(Enum):
        POOLING = 'pooling'
        ALIGN = 'align'
    OPTIONS = ['pooling', 'align']

    @staticmethod
    def apply(features: Tensor, proposal_bboxes: Tensor,
        proposal_batch_indices: Tensor, mode: Mode) ->Tensor:
        _, _, feature_map_height, feature_map_width = features.shape
        scale = 1 / 16
        output_size = 7 * 2, 7 * 2
        if mode == Pooler.Mode.POOLING:
            pool = []
            for proposal_bbox, proposal_batch_index in zip(proposal_bboxes,
                proposal_batch_indices):
                start_x = max(min(round(proposal_bbox[0].item() * scale), 
                    feature_map_width - 1), 0)
                start_y = max(min(round(proposal_bbox[1].item() * scale), 
                    feature_map_height - 1), 0)
                end_x = max(min(round(proposal_bbox[2].item() * scale) + 1,
                    feature_map_width), 1)
                end_y = max(min(round(proposal_bbox[3].item() * scale) + 1,
                    feature_map_height), 1)
                roi_feature_map = features[(proposal_batch_index), :,
                    start_y:end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(input=roi_feature_map,
                    output_size=output_size))
            pool = torch.stack(pool, dim=0)
        elif mode == Pooler.Mode.ALIGN:
            pool = ROIAlign(output_size, spatial_scale=scale, sampling_ratio=0
                )(features, torch.cat([proposal_batch_indices.view(-1, 1).
                float(), proposal_bboxes], dim=1))
        else:
            raise ValueError
        pool = F.max_pool2d(input=pool, kernel_size=2, stride=2)
        return pool


def beta_smooth_l1_loss(input: Tensor, target: Tensor, beta: float) ->Tensor:
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    loss = loss.sum() / (input.numel() + 1e-08)
    return loss


class RegionProposalNetwork(nn.Module):

    def __init__(self, num_features_out: int, anchor_ratios: List[Tuple[int,
        int]], anchor_sizes: List[int], pre_nms_top_n: int, post_nms_top_n:
        int, anchor_smooth_l1_loss_beta: float):
        super().__init__()
        self._features = nn.Sequential(nn.Conv2d(in_channels=
            num_features_out, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU())
        self._anchor_ratios = anchor_ratios
        self._anchor_sizes = anchor_sizes
        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_sizes = len(self._anchor_sizes)
        num_anchors = num_anchor_ratios * num_anchor_sizes
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta
        self._anchor_objectness = nn.Conv2d(in_channels=512, out_channels=
            num_anchors * 2, kernel_size=1)
        self._anchor_transformer = nn.Conv2d(in_channels=512, out_channels=
            num_anchors * 4, kernel_size=1)

    def forward(self, features: Tensor, anchor_bboxes: Optional[Tensor]=
        None, gt_bboxes_batch: Optional[Tensor]=None, image_width: Optional
        [int]=None, image_height: Optional[int]=None) ->Union[Tuple[Tensor,
        Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        batch_size = features.shape[0]
        features = self._features(features)
        anchor_objectnesses = self._anchor_objectness(features)
        anchor_transformers = self._anchor_transformer(features)
        anchor_objectnesses = anchor_objectnesses.permute(0, 2, 3, 1
            ).contiguous().view(batch_size, -1, 2)
        anchor_transformers = anchor_transformers.permute(0, 2, 3, 1
            ).contiguous().view(batch_size, -1, 4)
        if not self.training:
            return anchor_objectnesses, anchor_transformers
        else:
            inside_indices = BBox.inside(anchor_bboxes, left=0, top=0,
                right=image_width, bottom=image_height).nonzero().unbind(dim=1)
            inside_anchor_bboxes = anchor_bboxes[inside_indices].view(
                batch_size, -1, anchor_bboxes.shape[2])
            inside_anchor_objectnesses = anchor_objectnesses[inside_indices
                ].view(batch_size, -1, anchor_objectnesses.shape[2])
            inside_anchor_transformers = anchor_transformers[inside_indices
                ].view(batch_size, -1, anchor_transformers.shape[2])
            labels = torch.full((batch_size, inside_anchor_bboxes.shape[1]),
                -1, dtype=torch.long, device=inside_anchor_bboxes.device)
            ious = BBox.iou(inside_anchor_bboxes, gt_bboxes_batch)
            anchor_max_ious, anchor_assignments = ious.max(dim=2)
            gt_max_ious, gt_assignments = ious.max(dim=1)
            anchor_additions = ((ious > 0) & (ious == gt_max_ious.unsqueeze
                (dim=1))).nonzero()[:, :2].unbind(dim=1)
            labels[anchor_max_ious < 0.3] = 0
            labels[anchor_additions] = 1
            labels[anchor_max_ious >= 0.7] = 1
            fg_indices = (labels == 1).nonzero()
            bg_indices = (labels == 0).nonzero()
            fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(
                len(fg_indices), 128 * batch_size)]]
            bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 *
                batch_size - len(fg_indices)]]
            selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
            selected_indices = selected_indices[torch.randperm(len(
                selected_indices))].unbind(dim=1)
            inside_anchor_bboxes = inside_anchor_bboxes[selected_indices]
            gt_bboxes = gt_bboxes_batch[selected_indices[0],
                anchor_assignments[selected_indices]]
            gt_anchor_objectnesses = labels[selected_indices]
            gt_anchor_transformers = BBox.calc_transformer(inside_anchor_bboxes
                , gt_bboxes)
            batch_indices = selected_indices[0]
            anchor_objectness_losses, anchor_transformer_losses = self.loss(
                inside_anchor_objectnesses[selected_indices],
                inside_anchor_transformers[selected_indices],
                gt_anchor_objectnesses, gt_anchor_transformers, batch_size,
                batch_indices)
            return (anchor_objectnesses, anchor_transformers,
                anchor_objectness_losses, anchor_transformer_losses)

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
        gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor,
        batch_size: int, batch_indices: Tensor) ->Tuple[Tensor, Tensor]:
        cross_entropies = torch.empty(batch_size, dtype=torch.float, device
            =anchor_objectnesses.device)
        smooth_l1_losses = torch.empty(batch_size, dtype=torch.float,
            device=anchor_transformers.device)
        for batch_index in range(batch_size):
            selected_indices = (batch_indices == batch_index).nonzero().view(-1
                )
            cross_entropy = F.cross_entropy(input=anchor_objectnesses[
                selected_indices], target=gt_anchor_objectnesses[
                selected_indices])
            fg_indices = gt_anchor_objectnesses[selected_indices].nonzero(
                ).view(-1)
            smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[
                selected_indices][fg_indices], target=
                gt_anchor_transformers[selected_indices][fg_indices], beta=
                self._anchor_smooth_l1_loss_beta)
            cross_entropies[batch_index] = cross_entropy
            smooth_l1_losses[batch_index] = smooth_l1_loss
        return cross_entropies, smooth_l1_losses

    def generate_anchors(self, image_width: int, image_height: int,
        num_x_anchors: int, num_y_anchors: int) ->Tensor:
        center_ys = np.linspace(start=0, stop=image_height, num=
            num_y_anchors + 2)[1:-1]
        center_xs = np.linspace(start=0, stop=image_width, num=
            num_x_anchors + 2)[1:-1]
        ratios = np.array(self._anchor_ratios)
        ratios = ratios[:, (0)] / ratios[:, (1)]
        sizes = np.array(self._anchor_sizes)
        center_ys, center_xs, ratios, sizes = np.meshgrid(center_ys,
            center_xs, ratios, sizes, indexing='ij')
        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        sizes = sizes.reshape(-1)
        widths = sizes * np.sqrt(1 / ratios)
        heights = sizes * np.sqrt(ratios)
        center_based_anchor_bboxes = np.stack((center_xs, center_ys, widths,
            heights), axis=1)
        center_based_anchor_bboxes = torch.from_numpy(
            center_based_anchor_bboxes).float()
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)
        return anchor_bboxes

    def generate_proposals(self, anchor_bboxes: Tensor, objectnesses:
        Tensor, transformers: Tensor, image_width: int, image_height: int
        ) ->Tensor:
        batch_size = anchor_bboxes.shape[0]
        proposal_bboxes = BBox.apply_transformer(anchor_bboxes, transformers)
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=
            image_width, bottom=image_height)
        proposal_probs = F.softmax(objectnesses[:, :, (1)], dim=-1)
        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)
        nms_proposal_bboxes_batch = []
        for batch_index in range(batch_size):
            sorted_bboxes = proposal_bboxes[batch_index][sorted_indices[
                batch_index]][:self._pre_nms_top_n]
            sorted_probs = proposal_probs[batch_index][sorted_indices[
                batch_index]][:self._pre_nms_top_n]
            threshold = 0.7
            kept_indices = nms(sorted_bboxes, sorted_probs, threshold)
            nms_bboxes = sorted_bboxes[kept_indices][:self._post_nms_top_n]
            nms_proposal_bboxes_batch.append(nms_bboxes)
        max_nms_proposal_bboxes_length = max([len(it) for it in
            nms_proposal_bboxes_batch])
        padded_proposal_bboxes = []
        for nms_proposal_bboxes in nms_proposal_bboxes_batch:
            padded_proposal_bboxes.append(torch.cat([nms_proposal_bboxes,
                torch.zeros(max_nms_proposal_bboxes_length - len(
                nms_proposal_bboxes), 4).to(nms_proposal_bboxes)]))
        padded_proposal_bboxes = torch.stack(padded_proposal_bboxes, dim=0)
        return padded_proposal_bboxes


class _ROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(input, roi, spatial_scale,
            output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(grad_output, rois, spatial_scale,
            output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):

    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale,
            self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ')'
        return tmpstr


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_potterhsu_easy_faster_rcnn_pytorch(_paritybench_base):
    pass
