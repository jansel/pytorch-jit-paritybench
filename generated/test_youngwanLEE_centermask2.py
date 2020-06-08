import sys
_module = sys.modules[__name__]
del sys
centermask = _module
checkpoint = _module
adet_checkpoint = _module
config = _module
defaults = _module
evaluation = _module
coco_evaluation = _module
layers = _module
conv_with_kaiming_uniform = _module
deform_conv = _module
iou_loss = _module
ml_nms = _module
wrappers = _module
modeling = _module
backbone = _module
fpn = _module
mobilenet = _module
vovnet = _module
center_heads = _module
keypoint_head = _module
mask_head = _module
maskiou_head = _module
pooler = _module
proposal_utils = _module
sam = _module
fcos = _module
fcos = _module
fcos_outputs = _module
comm = _module
measures = _module
prepare_panoptic_fpn = _module
demo = _module
predictor = _module
train_net = _module

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


import torch


from torch import nn


import torch.nn.functional as F


from torch.nn import BatchNorm2d


from collections import OrderedDict


import torch.nn as nn


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


from torch.nn import functional as F


import copy


import math


import logging


from torch.nn.parallel import DistributedDataParallel


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class IOULoss(nn.Module):

    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, (0)]
        pred_top = pred[:, (1)]
        pred_right = pred[:, (2)]
        pred_bottom = pred[:, (3)]
        target_left = target[:, (0)]
        target_top = target[:, (1)]
        target_right = target[:, (2)]
        target_bottom = target[:, (3)]
        target_aera = (target_left + target_right) * (target_top +
            target_bottom)
        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right,
            target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
            pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class MaxPool2d(torch.nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.MaxPool2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._make_iteratable()

    def forward(self, x):
        if x.numel() == 0:
            output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // s + 1) for
                i, p, di, k, s in zip(x.shape[-2:], self.padding, self.
                dilation, self.kernel_size, self.stride)]
            output_shape = [x.shape[0], x.shape[1]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            return empty
        x = super().forward(x)
        return x

    def _make_iteratable(self):
        if not isinstance(self.padding, list):
            self.padding = [self.padding, self.padding]
        if not isinstance(self.dilation, list):
            self.dilation = [self.dilation, self.dilation]
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size, self.kernel_size]
        if not isinstance(self.stride, list):
            self.stride = [self.stride, self.stride]


class Linear(torch.nn.Linear):
    """
    A wrapper around :class:`torch.nn.Linear` to support empty inputs and more features.
    Because of https://github.com/pytorch/pytorch/issues/34202
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.numel() == 0:
            output_shape = [x.shape[0], self.weight.shape[0]]
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty
        x = super().forward(x)
        return x


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_features='res5'):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features='res5'):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(Conv2d(hidden_dim, hidden_dim, 3,
                stride, 1, groups=hidden_dim, bias=False),
                FrozenBatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(Conv2d(inp, hidden_dim, 1, 1, 0, bias
                =False), FrozenBatchNorm2d(hidden_dim), nn.ReLU6(inplace=
                True), Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=
                hidden_dim, bias=False), FrozenBatchNorm2d(hidden_dim), nn.
                ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=
                False), FrozenBatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


_NORM = False


class DFConv3x3(nn.Module):

    def __init__(self, in_channels, out_channels, module_name, postfix,
        dilation=1, groups=1, with_modulated_dcn=None, deformable_groups=1):
        super(DFConv3x3, self).__init__()
        self.module_names = []
        self.with_modulated_dcn = with_modulated_dcn
        if self.with_modulated_dcn:
            deform_conv_op = ModulatedDeformConv
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18
        unit_name = f'{module_name}_{postfix}/conv_offset'
        self.module_names.append(unit_name)
        self.add_module(unit_name, Conv2d(in_channels, offset_channels *
            deformable_groups, kernel_size=3, stride=1, padding=1 *
            dilation, dilation=dilation))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        unit_name = f'{module_name}_{postfix}/conv'
        self.module_names.append(unit_name)
        self.add_module(f'{module_name}_{postfix}/conv', deform_conv_op(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1 *
            dilation, bias=False, groups=groups, dilation=1,
            deformable_groups=deformable_groups))
        unit_name = f'{module_name}_{postfix}/norm'
        self.module_names.append(unit_name)
        self.add_module(unit_name, get_norm(_NORM, out_channels))

    def forward(self, x):
        if self.with_modulated_dcn:
            offset_mask = getattr(self, self.module_names[0])(x)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = getattr(self, self.module_names[1])(x, offset, mask)
        else:
            offset = getattr(self, self.module_names[0])(x)
            out = getattr(self, self.module_names[1])(x, offset)
        return F.relu_(getattr(self, self.module_names[2])(out))


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1,
    groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels,
        out_channels, kernel_size=kernel_size, stride=stride, padding=
        padding, groups=groups, bias=False)), (
        f'{module_name}_{postfix}/norm', get_norm(_NORM, out_channels)), (
        f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1,
    groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels,
        out_channels, kernel_size=kernel_size, stride=stride, padding=
        padding, groups=groups, bias=False)), (
        f'{module_name}_{postfix}/norm', get_norm(_NORM, out_channels)), (
        f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1,
    kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [('{}_{}/dw_conv3x3'.format(module_name, postfix), nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        padding=padding, groups=out_channels, bias=False)), (
        '{}_{}/pw_conv1x1'.format(module_name, postfix), nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0,
        groups=1, bias=False)), ('{}_{}/pw_norm'.format(module_name,
        postfix), get_norm(_NORM, out_channels)), ('{}_{}/pw_relu'.format(
        module_name, postfix), nn.ReLU(inplace=True))]


class _OSA_module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block,
        module_name, SE=False, identity=False, depthwise=False, dcn_config={}):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(OrderedDict(conv1x1(
                in_channel, stage_ch, '{}_reduction'.format(module_name), '0'))
                )
        with_dcn = dcn_config.get('stage_with_dcn', False)
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(
                    stage_ch, stage_ch, module_name, i))))
            elif with_dcn:
                deformable_groups = dcn_config.get('deformable_groups', 1)
                with_modulated_dcn = dcn_config.get('with_modulated_dcn', False
                    )
                self.layers.append(DFConv3x3(in_channel, stage_ch,
                    module_name, i, with_modulated_dcn=with_modulated_dcn,
                    deformable_groups=deformable_groups))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(
                    in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel,
            concat_ch, module_name, 'concat')))
        self.ese = eSEModule(concat_ch)

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat
        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage,
        layer_per_block, stage_num, SE=False, depthwise=False, dcn_config={}):
        super(_OSA_stage, self).__init__()
        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2,
                ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch,
            layer_per_block, module_name, SE, depthwise=depthwise,
            dcn_config=dcn_config))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch,
                concat_ch, layer_per_block, module_name, SE, identity=True,
                depthwise=depthwise, dcn_config=dcn_config))


def add_ground_truth_to_proposals_single_image(targets_i, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with targets and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.scores.device
    proposals.proposal_boxes = proposals.pred_boxes
    proposals.remove('pred_boxes')
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(targets_i), device=device)
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = targets_i.gt_boxes
    gt_proposal.scores = gt_logits
    gt_proposal.pred_classes = targets_i.gt_classes
    gt_proposal.locations = torch.ones((len(targets_i), 2), device=device)
    new_proposals = Instances.cat([proposals, gt_proposal])
    return new_proposals


def add_ground_truth_to_proposals(targets, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        targets(list[Instances]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert targets is not None
    assert len(proposals) == len(targets)
    if len(proposals) == 0:
        return proposals
    return [add_ground_truth_to_proposals_single_image(tagets_i,
        proposals_i) for tagets_i, proposals_i in zip(targets, proposals)]


def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, score).
            The scores are larger than 0.
    """
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)
    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(),
        bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, ([0, 1, 3])].split(
        num_instances_per_image, dim=0)
    for keypoint_results_per_image, instances_per_image in zip(keypoint_results
        , pred_instances):
        instances_per_image.pred_keypoints = keypoint_results_per_image


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []
    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len)
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar('kpts_num_skipped_batches', _TOTAL_SKIPPED,
            smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0
    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    keypoint_loss = F.cross_entropy(pred_keypoint_logits[valid],
        keypoint_targets[valid], reduction='sum')
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer
    return keypoint_loss


def _img_area(instance):
    device = instance.pred_classes.device
    image_size = instance.image_size
    area = torch.as_tensor(image_size[0] * image_size[1], dtype=torch.float,
        device=device)
    tmp = torch.zeros((len(instance.pred_classes), 1), dtype=torch.float,
        device=device)
    return (area + tmp).squeeze(1)


def assign_boxes_to_levels_by_ratio(instances, min_level, max_level,
    is_train=False):
    """
    Map each box in `instances` to a feature map level index by adaptive ROI mapping function 
    in CenterMask paper and return the assignment
    vector.

    Args:
        instances (list[Instances]): the per-image instances to train/predict masks.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    if is_train:
        box_lists = [x.proposal_boxes for x in instances]
    else:
        box_lists = [x.pred_boxes for x in instances]
    box_areas = cat([boxes.area() for boxes in box_lists])
    img_areas = cat([_img_area(instance_i) for instance_i in instances])
    level_assignments = torch.ceil(max_level - torch.log2(img_areas /
        box_areas + eps))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=
        max_level)
    return level_assignments.to(torch.int64) - min_level


def assign_boxes_to_levels(box_lists, min_level, max_level,
    canonical_box_size, canonical_level):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes /
        canonical_box_size + eps))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=
        max_level)
    return level_assignments.to(torch.int64) - min_level


def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full((len(box_tensor), 1), batch_index,
            dtype=box_tensor.dtype, device=box_tensor.device)
        return cat((repeated_index, box_tensor), dim=1)
    pooler_fmt_boxes = cat([fmt_box_list(box_list.tensor, i) for i,
        box_list in enumerate(box_lists)], dim=0)
    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(self, output_size, scales, sampling_ratio, pooler_type,
        canonical_box_size=224, canonical_level=4, assign_crit='area'):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = output_size, output_size
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1
            ], int)
        self.output_size = output_size
        if pooler_type == 'ROIAlign':
            self.level_poolers = nn.ModuleList(ROIAlign(output_size,
                spatial_scale=scale, sampling_ratio=sampling_ratio, aligned
                =False) for scale in scales)
        elif pooler_type == 'ROIAlignV2':
            self.level_poolers = nn.ModuleList(ROIAlign(output_size,
                spatial_scale=scale, sampling_ratio=sampling_ratio, aligned
                =True) for scale in scales)
        elif pooler_type == 'ROIPool':
            self.level_poolers = nn.ModuleList(RoIPool(output_size,
                spatial_scale=scale) for scale in scales)
        elif pooler_type == 'ROIAlignRotated':
            self.level_poolers = nn.ModuleList(ROIAlignRotated(output_size,
                spatial_scale=scale, sampling_ratio=sampling_ratio) for
                scale in scales)
        else:
            raise ValueError('Unknown pooler type: {}'.format(pooler_type))
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)), 'Featuremap stride is not power of 2!'
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert len(scales
            ) == self.max_level - self.min_level + 1, '[ROIPooler] Sizes of input featuremaps do not form a pyramid!'
        assert 0 < self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        self.assign_crit = assign_crit

    def forward(self, x, instances, is_train=False):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
            is_train (True/False)

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        if is_train:
            box_lists = [x.proposal_boxes for x in instances]
        else:
            box_lists = [x.pred_boxes for x in instances]
        num_level_assignments = len(self.level_poolers)
        assert isinstance(x, list) and isinstance(box_lists, list
            ), 'Arguments to pooler must be lists'
        assert len(x
            ) == num_level_assignments, 'unequal value, num_level_assignments={}, but x is list of {} Tensors'.format(
            num_level_assignments, len(x))
        assert len(box_lists) == x[0].size(0
            ), 'unequal value, x[0] batch dim 0 is {}, but box_list has length {}'.format(
            x[0].size(0), len(box_lists))
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        if self.assign_crit == 'ratio':
            level_assignments = assign_boxes_to_levels_by_ratio(instances,
                self.min_level, self.max_level, is_train)
        else:
            level_assignments = assign_boxes_to_levels(box_lists, self.
                min_level, self.max_level, self.canonical_box_size, self.
                canonical_level)
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size,
            output_size), dtype=dtype, device=device)
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
        return output


def Max(x):
    """
    A wrapper around torch.max in Spatial Attention Module (SAM) to support empty inputs and more features.
    """
    if x.numel() == 0:
        output_shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        empty = _NewEmptyTensorOp.apply(x, output_shape)
        return empty
    return torch.max(x, dim=1, keepdim=True)[0]


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        weight_init.c2_msra_fill(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, ([0, 2])]
    top_bottom = reg_targets[:, ([1, 3])]
    ctrness = left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def fcos_losses(labels, reg_targets, logits_pred, reg_pred, ctrness_pred,
    focal_loss_alpha, focal_loss_gamma, iou_loss):
    num_classes = logits_pred.size(1)
    labels = labels.flatten()
    pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
    num_pos_local = pos_inds.numel()
    num_gpus = get_world_size()
    total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
    num_pos_avg = max(total_num_pos / num_gpus, 1.0)
    class_target = torch.zeros_like(logits_pred)
    class_target[pos_inds, labels[pos_inds]] = 1
    class_loss = sigmoid_focal_loss_jit(logits_pred, class_target, alpha=
        focal_loss_alpha, gamma=focal_loss_gamma, reduction='sum'
        ) / num_pos_avg
    reg_pred = reg_pred[pos_inds]
    reg_targets = reg_targets[pos_inds]
    ctrness_pred = ctrness_pred[pos_inds]
    ctrness_targets = compute_ctrness_targets(reg_targets)
    ctrness_targets_sum = ctrness_targets.sum()
    ctrness_norm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-06
        )
    reg_loss = iou_loss(reg_pred, reg_targets, ctrness_targets) / ctrness_norm
    ctrness_loss = F.binary_cross_entropy_with_logits(ctrness_pred,
        ctrness_targets, reduction='sum') / num_pos_avg
    losses = {'loss_fcos_cls': class_loss, 'loss_fcos_loc': reg_loss,
        'loss_fcos_ctr': ctrness_loss}
    return losses, {}


def ml_nms(boxlist, nms_thresh, max_proposals=-1, score_field='scores',
    label_field='labels'):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist


INF = 100000000


class FCOSOutputs(object):

    def __init__(self, images, locations, logits_pred, reg_pred,
        ctrness_pred, focal_loss_alpha, focal_loss_gamma, iou_loss,
        center_sample, sizes_of_interest, strides, radius, num_classes,
        pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n,
        thresh_with_ctr, gt_instances=None):
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        self.ctrness_pred = ctrness_pred
        self.locations = locations
        self.gt_instances = gt_instances
        self.num_feature_maps = len(logits_pred)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr

    def _transpose(self, training_targets, num_loc_list):
        """
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        """
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(training_targets[im_i],
                num_loc_list, dim=0)
        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(torch.cat(targets_per_level, dim=0))
        return targets_level_first

    def _get_ground_truth(self):
        num_loc_list = [len(loc) for loc in self.locations]
        self.num_loc_list = num_loc_list
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.
                sizes_of_interest[l])
            loc_to_size_range.append(loc_to_size_range_per_level[None].
                expand(num_loc_list[l], -1))
        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)
        training_targets = self.compute_targets_for_locations(locations,
            self.gt_instances, loc_to_size_range)
        training_targets = {k: self._transpose(v, num_loc_list) for k, v in
            training_targets.items()}
        reg_targets = training_targets['reg_targets']
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
        return training_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys,
        radius=1):
        num_gts = gt.shape[0]
        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            center_gt[beg:end, :, (0)] = torch.where(xmin > gt[beg:end, :,
                (0)], xmin, gt[beg:end, :, (0)])
            center_gt[beg:end, :, (1)] = torch.where(ymin > gt[beg:end, :,
                (1)], ymin, gt[beg:end, :, (1)])
            center_gt[beg:end, :, (2)] = torch.where(xmax > gt[beg:end, :,
                (2)], gt[beg:end, :, (2)], xmax)
            center_gt[beg:end, :, (3)] = torch.where(ymax > gt[beg:end, :,
                (3)], gt[beg:end, :, (3)], ymax)
            beg = end
        left = loc_xs[:, (None)] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, (None)]
        top = loc_ys[:, (None)] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, (None)]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges):
        labels = []
        reg_targets = []
        xs, ys = locations[:, (0)], locations[:, (1)]
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) +
                    self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue
            area = targets_per_im.gt_boxes.area()
            l = xs[:, (None)] - bboxes[:, (0)][None]
            t = ys[:, (None)] - bboxes[:, (1)][None]
            r = bboxes[:, (2)][None] - xs[:, (None)]
            b = bboxes[:, (3)][None] - ys[:, (None)]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            if self.center_sample:
                is_in_boxes = self.get_sample_region(bboxes, self.strides,
                    self.num_loc_list, xs, ys, radius=self.radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            is_cared_in_the_level = (max_reg_targets_per_im >= size_ranges[
                :, ([0])]) & (max_reg_targets_per_im <= size_ranges[:, ([1])])
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            locations_to_min_area, locations_to_gt_inds = (locations_to_gt_area
                .min(dim=1))
            reg_targets_per_im = reg_targets_per_im[range(len(locations)),
                locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        return {'labels': labels, 'reg_targets': reg_targets}

    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """
        training_targets = self._get_ground_truth()
        labels, reg_targets = training_targets['labels'], training_targets[
            'reg_targets']
        logits_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, self.
            num_classes) for x in self.logits_pred], dim=0)
        reg_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in self.
            reg_pred], dim=0)
        ctrness_pred = cat([x.reshape(-1) for x in self.ctrness_pred], dim=0)
        labels = cat([x.reshape(-1) for x in labels], dim=0)
        reg_targets = cat([x.reshape(-1, 4) for x in reg_targets], dim=0)
        return fcos_losses(labels, reg_targets, logits_pred, reg_pred,
            ctrness_pred, self.focal_loss_alpha, self.focal_loss_gamma,
            self.iou_loss)

    def predict_proposals(self):
        sampled_boxes = []
        bundle = (self.locations, self.logits_pred, self.reg_pred, self.
            ctrness_pred, self.strides)
        for i, (l, o, r, c, s) in enumerate(zip(*bundle)):
            r = r * s
            sampled_boxes.append(self.forward_for_single_feature_map(l, o,
                r, c, self.image_sizes))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def forward_for_single_feature_map(self, locations, box_cls, reg_pred,
        ctrness, image_sizes):
        N, C, H, W = box_cls.shape
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness = ctrness.reshape(N, -1).sigmoid()
        if self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, (None)]
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        if not self.thresh_with_ctr:
            box_cls = box_cls * ctrness[:, :, (None)]
        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, (0)]
            per_class = per_candidate_nonzeros[:, (1)]
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n
                    , sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            detections = torch.stack([per_locations[:, (0)] -
                per_box_regression[:, (0)], per_locations[:, (1)] -
                per_box_regression[:, (1)], per_locations[:, (0)] +
                per_box_regression[:, (2)], per_locations[:, (1)] +
                per_box_regression[:, (3)]], dim=1)
            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            results.append(boxlist)
        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(cls_scores.cpu(), 
                    number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_youngwanLEE_centermask2(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(IOULoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(MaxPool2d(*[], **{'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(Linear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Hsigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(eSEModule(*[], **{'channel': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Scale(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
