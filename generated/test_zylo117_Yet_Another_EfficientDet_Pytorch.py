import sys
_module = sys.modules[__name__]
del sys
backbone = _module
coco_eval = _module
config = _module
dataset = _module
loss = _module
model = _module
utils = _module
efficientdet_test = _module
efficientdet_test_videos = _module
efficientnet = _module
model = _module
utils = _module
utils_extra = _module
train = _module
sync_batchnorm = _module
batchnorm = _module
batchnorm_reimpl = _module
comm = _module
replicate = _module
unittest = _module
utils = _module

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


import math


import torch


from torch import nn


import torch.nn as nn


import numpy as np


import itertools


from torch.nn import functional as F


import re


import collections


from functools import partial


from torch.utils import model_zoo


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as init


import functools


from torch.nn.parallel.data_parallel import DataParallel


from typing import Union


import uuid


from torch.nn.init import _calculate_fan_in_and_fan_out


from torch.nn.init import _no_grad_normal_


class EfficientDetBackbone(nn.Module):

    def __init__(self, num_classes=80, compound_coef=0, load_weights=False,
        **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7),
            (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0
            ), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {(0): [40, 112, 320], (1): [40, 112, 320], (2):
            [48, 120, 352], (3): [48, 136, 384], (4): [56, 160, 448], (5):
            [64, 176, 512], (6): [72, 200, 576], (7): [72, 200, 576]}
        num_anchors = len(self.aspect_ratios) * self.num_scales
        self.bifpn = nn.Sequential(*[BiFPN(self.fpn_num_filters[self.
            compound_coef], conv_channel_coef[compound_coef], True if _ == 
            0 else False, attention=True if compound_coef < 6 else False) for
            _ in range(self.fpn_cell_repeats[compound_coef])])
        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.
            compound_coef], num_anchors=num_anchors, num_layers=self.
            box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.
            compound_coef], num_anchors=num_anchors, num_classes=
            num_classes, num_layers=self.box_class_repeats[self.compound_coef])
        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef
            ], **kwargs)
        self.backbone_net = EfficientNet(self.backbone_compound_coef[
            compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]
        _, p3, p4, p5 = self.backbone_net(inputs)
        features = p3, p4, p5
        features = self.bifpn(features)
        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)
        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            None
        except RuntimeError as e:
            None


def calc_iou(a, b):
    area = (b[:, (2)] - b[:, (0)]) * (b[:, (3)] - b[:, (1)])
    iw = torch.min(torch.unsqueeze(a[:, (3)], dim=1), b[:, (2)]) - torch.max(
        torch.unsqueeze(a[:, (1)], 1), b[:, (0)])
    ih = torch.min(torch.unsqueeze(a[:, (2)], dim=1), b[:, (3)]) - torch.max(
        torch.unsqueeze(a[:, (0)], 1), b[:, (1)])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, (2)] - a[:, (0)]) * (a[:, (3)] - a[:, (1)]),
        dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-08)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score), (x1, y1 +
                10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def postprocess(x, anchors, regression, classification, regressBoxes,
    clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, (0)]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({'rois': np.array(()), 'class_ids': np.array(()),
                'scores': np.array(())})
            continue
        classification_per = classification[i, scores_over_thresh[(i), :], ...
            ].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh
            [(i), :], ...]
        scores_per = scores[i, scores_over_thresh[(i), :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:,
            (0)], classes_, iou_threshold=iou_threshold)
        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[(anchors_nms_idx), :]
            out.append({'rois': boxes_.cpu().numpy(), 'class_ids': classes_
                .cpu().numpy(), 'scores': scores_.cpu().numpy()})
        else:
            out.append({'rois': np.array(()), 'class_ids': np.array(()),
                'scores': np.array(())})
    return out


class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations,
        **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchor = anchors[(0), :, :]
        dtype = anchors.dtype
        anchor_widths = anchor[:, (3)] - anchor[:, (1)]
        anchor_heights = anchor[:, (2)] - anchor[:, (0)]
        anchor_ctr_x = anchor[:, (1)] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, (0)] + 0.5 * anchor_heights
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            regression = regressions[(j), :, :]
            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, (4)] != -1]
            classification = torch.clamp(classification, 0.0001, 1.0 - 0.0001)
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor
                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma
                        )
                    bce = -torch.log(1.0 - classification)
                    cls_loss = focal_weight * bce
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())
                else:
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma
                        )
                    bce = -torch.log(1.0 - classification)
                    cls_loss = focal_weight * bce
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())
                continue
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets
            targets[(torch.lt(IoU_max, 0.4)), :] = 0
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[(IoU_argmax), :]
            targets[(positive_indices), :] = 0
            targets[positive_indices, assigned_annotations[positive_indices,
                4].long()] = 1
            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor
            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor,
                1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 -
                classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) *
                torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)
            classification_losses.append(cls_loss.sum() / torch.clamp(
                num_positive_anchors.to(dtype), min=1.0))
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[(
                    positive_indices), :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                gt_widths = assigned_annotations[:, (2)
                    ] - assigned_annotations[:, (0)]
                gt_heights = assigned_annotations[:, (3)
                    ] - assigned_annotations[:, (1)]
                gt_ctr_x = assigned_annotations[:, (0)] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, (1)] + 0.5 * gt_heights
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                targets = torch.stack((targets_dy, targets_dx, targets_dh,
                    targets_dw))
                targets = targets.t()
                regression_diff = torch.abs(targets - regression[(
                    positive_indices), :])
                regression_loss = torch.where(torch.le(regression_diff, 1.0 /
                    9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), 
                    regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            elif torch.cuda.is_available():
                regression_losses.append(torch.tensor(0).to(dtype))
            else:
                regression_losses.append(torch.tensor(0).to(dtype))
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(), torch.stack([anchors[0]] *
                imgs.shape[0], 0).detach(), regressions.detach(),
                classifications.detach(), regressBoxes, clipBoxes, 0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) *
                255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)
        return torch.stack(classification_losses).mean(dim=0, keepdim=True
            ), torch.stack(regression_losses).mean(dim=0, keepdim=True)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True,
        activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels,
            in_channels, kernel_size=3, stride=1, groups=in_channels, bias=
            False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels,
            out_channels, kernel_size=1, stride=1)
        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=
                0.01, eps=0.001)
        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False,
        epsilon=0.0001, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=
            onnx_export)
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(Conv2dStaticSamePadding(
                conv_channels[2], num_channels, 1), nn.BatchNorm2d(
                num_channels, momentum=0.01, eps=0.001))
            self.p4_down_channel = nn.Sequential(Conv2dStaticSamePadding(
                conv_channels[1], num_channels, 1), nn.BatchNorm2d(
                num_channels, momentum=0.01, eps=0.001))
            self.p3_down_channel = nn.Sequential(Conv2dStaticSamePadding(
                conv_channels[0], num_channels, 1), nn.BatchNorm2d(
                num_channels, momentum=0.01, eps=0.001))
            self.p5_to_p6 = nn.Sequential(Conv2dStaticSamePadding(
                conv_channels[2], num_channels, 1), nn.BatchNorm2d(
                num_channels, momentum=0.01, eps=0.001),
                MaxPool2dStaticSamePadding(3, 2))
            self.p6_to_p7 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))
            self.p4_down_channel_2 = nn.Sequential(Conv2dStaticSamePadding(
                conv_channels[1], num_channels, 1), nn.BatchNorm2d(
                num_channels, momentum=0.01, eps=0.001))
            self.p5_down_channel_2 = nn.Sequential(Conv2dStaticSamePadding(
                conv_channels[2], num_channels, 1), nn.BatchNorm2d(
                num_channels, momentum=0.01, eps=0.001))
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32),
            requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32),
            requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32),
            requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32),
            requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32),
            requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32),
            requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32),
            requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32),
            requires_grad=True)
        self.p7_w2_relu = nn.ReLU()
        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = (self.
                _forward_fast_attention(inputs))
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)
        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] *
            self.p6_upsample(p7_in)))
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] *
            self.p5_upsample(p6_up)))
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] *
            self.p4_upsample(p5_up)))
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] *
            self.p3_upsample(p4_up)))
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(self.swish(weight[0] * p4_in + weight[1] *
            p4_up + weight[2] * self.p4_downsample(p3_out)))
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] *
            p5_up + weight[2] * self.p5_downsample(p4_out)))
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(self.swish(weight[0] * p6_in + weight[1] *
            p6_up + weight[2] * self.p6_downsample(p5_out)))
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] *
            self.p7_downsample(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)
        p4_out = self.conv4_down(self.swish(p4_in + p4_up + self.
            p4_downsample(p3_out)))
        p5_out = self.conv5_down(self.swish(p5_in + p5_up + self.
            p5_downsample(p4_out)))
        p6_out = self.conv6_down(self.swish(p6_in + p6_up + self.
            p6_downsample(p5_out)))
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out))
            )
        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False
        ):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels,
            in_channels, norm=False, activation=False) for i in range(
            num_layers)])
        self.bn_list = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(
            in_channels, momentum=0.01, eps=0.001) for i in range(
            num_layers)]) for j in range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm
            =False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.
                conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)
            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers,
        onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels,
            in_channels, norm=False, activation=False) for i in range(
            num_layers)])
        self.bn_list = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(
            in_channels, momentum=0.01, eps=0.001) for i in range(
            num_layers)]) for j in range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors *
            num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.
                conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1],
                feat.shape[2], self.num_anchors, self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)
            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()
        return feats


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}',
            load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]


class BBoxTransform(nn.Module):

    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]
        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha
        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a
        ymin = y_centers - h / 2.0
        xmin = x_centers - w / 2.0
        ymax = y_centers + h / 2.0
        xmax = x_centers + w / 2.0
        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, (0)] = torch.clamp(boxes[:, :, (0)], min=0)
        boxes[:, :, (1)] = torch.clamp(boxes[:, :, (1)], min=0)
        boxes[:, :, (2)] = torch.clamp(boxes[:, :, (2)], max=width - 1)
        boxes[:, :, (3)] = torch.clamp(boxes[:, :, (3)], max=height - 1)
        return boxes


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4.0, pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = kwargs.get('strides', [(2 ** x) for x in self.
            pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 
            3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
            )
        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]
        if (image_shape == self.last_shape and image.device in self.
            last_anchors):
            return self.last_anchors[image.device]
        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape
        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError(
                        'input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                boxes = np.vstack((yv - anchor_size_y_2, xv -
                    anchor_size_x_2, yv + anchor_size_y_2, xv +
                    anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image
            .device)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype,
        device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None and 0 < self.
            _block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup,
                kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self.
                _bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup,
            groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom,
            eps=self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.
                input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=
                num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels,
                out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup,
            kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self.
            _bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = (self._block_args.input_filters,
            self._block_args.output_filters)
        if (self.id_skip and self._block_args.stride == 1 and input_filters ==
            output_filters):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training
                    )
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size',
    'num_repeat', 'input_filters', 'output_filters', 'expand_ratio',
    'id_skip', 'stride', 'se_ratio'])


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        assert 's' in options and len(options['s']) == 1 or len(options['s']
            ) == 2 and options['s'][0] == options['s'][1]
        return BlockArgs(kernel_size=int(options['k']), num_repeat=int(
            options['r']), input_filters=int(options['i']), output_filters=
            int(options['o']), expand_ratio=int(options['e']), id_skip=
            'noskip' not in block_string, se_ratio=float(options['se']) if 
            'se' in options else None, stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 
            's%d%d' % (block.strides[0], block.strides[1]), 'e%s' % block.
            expand_ratio, 'i%d' % block.input_filters, 'o%d' % block.
            output_filters]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])


def efficientnet(width_coefficient=None, depth_coefficient=None,
    dropout_rate=0.2, drop_connect_rate=0.2, image_size=None, num_classes=1000
    ):
    """ Creates a efficientnet model. """
    blocks_args = ['r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']
    blocks_args = BlockDecoder.decode(blocks_args)
    global_params = GlobalParams(batch_norm_momentum=0.99,
        batch_norm_epsilon=0.001, dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate, num_classes=num_classes,
        width_coefficient=width_coefficient, depth_coefficient=
        depth_coefficient, depth_divisor=8, min_depth=None, image_size=
        image_size)
    return blocks_args, global_params


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 
        1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 
        2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5), 'efficientnet-b8': (2.2, 
        3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5)}
    return params_dict[model_name]


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(width_coefficient=w,
            depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' %
            model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


url_map = {'efficientnet-b0':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b0-355c32eb.pth'
    , 'efficientnet-b1':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b1-f1951068.pth'
    , 'efficientnet-b2':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b2-8bb594d6.pth'
    , 'efficientnet-b3':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b3-5fb5a3c3.pth'
    , 'efficientnet-b4':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b4-6ed6700e.pth'
    , 'efficientnet-b5':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b5-b6417697.pth'
    , 'efficientnet-b6':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b6-c76e70fd.pth'
    , 'efficientnet-b7':
    'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b7-dcc49843.pth'
    }


url_map_advprop = {'efficientnet-b0':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b0-b64d5a18.pth'
    , 'efficientnet-b1':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b1-0f3ce85a.pth'
    , 'efficientnet-b2':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b2-6e9d97e5.pth'
    , 'efficientnet-b3':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b3-cdd7c0f4.pth'
    , 'efficientnet-b4':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b4-44fb3a87.pth'
    , 'efficientnet-b5':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b5-86493f6b.pth'
    , 'efficientnet-b6':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b6-ac80338e.pth'
    , 'efficientnet-b7':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b7-4652b6dd.pth'
    , 'efficientnet-b8':
    'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b8-22a8fe65.pth'
    }


def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    url_map_ = url_map_advprop if advprop else url_map
    state_dict = model_zoo.load_url(url_map_[model_name], map_location=
        torch.device('cpu'))
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        print(ret)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']
            ), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor *
        divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3,
            stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=
            bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(input_filters=round_filters(
                block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters,
                self._global_params), num_repeat=round_repeats(block_args.
                num_repeat, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.
                    output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self.
                    _global_params))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1,
            bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=
            bn_mom, eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name,
            override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, load_weights=True, advprop=True,
        num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes':
            num_classes})
        if load_weights:
            load_pretrained_weights(model, model_name, load_fc=num_classes ==
                1000, advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model.
                _global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels,
                kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = [('efficientnet-b' + str(i)) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(
                valid_models))


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]
            ] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] +
            1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] +
            1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1
            ] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0
            ] - h + self.kernel_size[0]
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1
            ] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0
            ] - h + self.kernel_size[0]
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.pool(x)
        return x


class ModelWithLoss(nn.Module):

    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression,
                anchors, annotations, imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression,
                anchors, annotations)
        return cls_loss, reg_loss


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum',
    'sum_size'])


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, "Previous result has't been fetched."
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()
            res = self._result
            self._result = None
            return res


_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier',
    'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(
                ), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True
        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())
        results = self._master_callback(intermediates)
        assert results[0][0
            ] == 0, 'The first result should belongs to the master.'
        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)
        for i in range(self.nr_slaves):
            assert self._queue.get() is True
        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps,
            momentum=momentum, affine=affine)
        self._sync_master = SyncMaster(self._data_parallel_master)
        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        if not (self._is_parallel and self.training):
            return F.batch_norm(input, self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps)
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(
                input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(
                input_sum, input_ssum, sum_size))
        if self.affine:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std *
                self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.
            get_device())
        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 +
                2])))
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size
        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum
                    ) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum
                    ) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum
                ) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum
                ) * self.running_var + self.momentum * unbias_var.data
        return mean, bias_var.clamp(self.eps) ** -0.5


class BatchNorm2dReimpl(nn.Module):
    """
    A re-implementation of batch normalization, used for testing the numerical
    stability.

    Author: acgtyrant
    See also:
    https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        self.running_mean = (1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
        unbias_var = sumvar / (numel - 1)
        self.running_var = (1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.detach()
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = (input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1
            ) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)
        return output.view(channels, batchsize, height, width).permute(1, 0,
            2, 3).contiguous()


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module,
            device_ids)
        execute_replication_callbacks(modules)
        return modules


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        devices = [('cuda:' + str(x)) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus
        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')
        return [(inputs[0][splits * device_idx:splits * (device_idx + 1)].
            to(f'cuda:{device_idx}', non_blocking=True), inputs[1][splits *
            device_idx:splits * (device_idx + 1)].to(f'cuda:{device_idx}',
            non_blocking=True)) for device_idx in range(len(devices))], [kwargs
            ] * len(devices)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zylo117_Yet_Another_EfficientDet_Pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SeparableConvBlock(*[], **{'in_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Regressor(*[], **{'in_channels': 4, 'num_anchors': 4, 'num_layers': 1}), [torch.rand([4, 4, 4, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Classifier(*[], **{'in_channels': 4, 'num_anchors': 4, 'num_classes': 4, 'num_layers': 1}), [torch.rand([4, 4, 4, 64, 64])], {})

    def test_003(self):
        self._check(BBoxTransform(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(ClipBoxes(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(MemoryEfficientSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Conv2dDynamicSamePadding(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(Conv2dStaticSamePadding(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(MaxPool2dStaticSamePadding(*[], **{'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(BatchNorm2dReimpl(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

