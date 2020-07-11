import sys
_module = sys.modules[__name__]
del sys
generate_anchors = _module
backbones = _module
fpn = _module
layers = _module
mobilenet = _module
resnet = _module
utils = _module
box = _module
dali = _module
data = _module
infer = _module
loss = _module
main = _module
model = _module
train = _module
utils = _module
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


import torch.nn as nn


import torch.nn.functional as F


from torchvision.models import resnet as vrn


from torchvision.models import mobilenet as vmn


import torch


from torch import nn


import torch.utils.model_zoo as model_zoo


import torchvision


import numpy as np


from math import ceil


import random


from torch.utils import data


import math


from torchvision.transforms.functional import adjust_brightness


from torchvision.transforms.functional import adjust_contrast


from torchvision.transforms.functional import adjust_hue


from torchvision.transforms.functional import adjust_saturation


import torch.cuda


import torch.distributed


import torch.multiprocessing


from math import isfinite


from torch.optim import SGD


from torch.optim import AdamW


from torch.optim.lr_scheduler import LambdaLR


import time


import warnings


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


class MobileNet(vmn.MobileNetV2):
    """MobileNetV2: Inverted Residuals and Linear Bottlenecks - https://arxiv.org/abs/1801.04381"""

    def __init__(self, outputs=[18], url=None):
        self.stride = 128
        self.url = url
        super().__init__()
        self.outputs = outputs

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        outputs = []
        for indx, feat in enumerate(self.features[:-1]):
            x = feat(x)
            if indx in self.outputs:
                outputs.append(x)
        return outputs


class ResNet(vrn.ResNet):
    """Deep Residual Network - https://arxiv.org/abs/1512.03385"""

    def __init__(self, layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, outputs=[5], groups=1, width_per_group=64, url=None):
        self.stride = 128
        self.bottleneck = bottleneck
        self.outputs = outputs
        self.url = url
        kwargs = {'block': bottleneck, 'layers': layers, 'groups': groups, 'width_per_group': width_per_group}
        super().__init__(**kwargs)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class FPN(nn.Module):
    """Feature Pyramid Network - https://arxiv.org/abs/1612.03144"""

    def __init__(self, features):
        super().__init__()
        self.stride = 128
        self.features = features
        if isinstance(features, ResNet):
            is_light = features.bottleneck == vrn.BasicBlock
            channels = [128, 256, 512] if is_light else [512, 1024, 2048]
        elif isinstance(features, MobileNet):
            channels = [32, 96, 320]
        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):

        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        self.apply(init_layer)
        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3
        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)
        return [p3, p4, p5, p6, p7]


class FixedBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed"""

    def __init__(self, n):
        super().__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x):
        return F.batch_norm(x, running_mean=self.running_mean, running_var=self.running_var, weight=self.weight, bias=self.bias)


class FocalLoss(nn.Module):
    """Focal Loss - https://arxiv.org/abs/1708.02002"""

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        return alpha * (1.0 - pt) ** self.gamma * ce


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss"""

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


def delta2box(deltas, anchors, size, stride):
    """Convert deltas from anchors to boxes"""
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh
    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([clamp(pred_ctr - 0.5 * pred_wh), clamp(pred_ctr + 0.5 * pred_wh - 1)], 1)


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
    """Box Decoding and Filtering"""
    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6
    if torch.cuda.is_available():
        return decode_cuda(all_cls_head.float(), all_box_head.float(), anchors.view(-1).tolist(), stride, threshold, top_n, rotated)
    device = all_cls_head.device
    anchors = anchors.type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]
    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)
    for batch in range(batch_size):
        cls_head = all_cls_head[(batch), :, :, :].contiguous().view(-1)
        box_head = all_box_head[(batch), :, :, :].contiguous().view(-1, num_boxes)
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = indices / width / height % num_classes
        classes = classes.type(all_cls_head.type())
        x = indices % width
        y = indices / width % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, num_boxes, height, width)
        boxes = box_head[(a), :, (y), (x)]
        if anchors is not None:
            grid = torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride + anchors[(a), :]
            boxes = delta2box(boxes, grid, [width, height], stride)
        out_scores[(batch), :scores.size()[0]] = scores
        out_boxes[(batch), :boxes.size()[0], :] = boxes
        out_classes[(batch), :classes.size()[0]] = classes
    return out_scores, out_boxes, out_classes


def order_points(pts):
    pts_reorder = []
    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, (0)])
        xSorted = pt[(idx), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[(torch.argsort(leftMost[:, (1)])), :]
        tl, bl = leftMost
        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        br, tr = rightMost[(torch.argsort(D, descending=True)), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))
    return torch.stack([p for p in pts_reorder])


def generate_anchors_rotated(stride, ratio_vals, scales_vals, angles_vals):
    """Generate anchors coordinates from scales/ratios/angles"""
    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))
    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.round(torch.sqrt(wh[:, (0)] * wh[:, (1)] / ratios))
    dwh = torch.stack([ws, torch.round(ws * ratios)], dim=1)
    xy0 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales) - 1
    xy1 = xy0 + (xy2 - xy0) * torch.FloatTensor([0, 1])
    xy3 = xy0 + (xy2 - xy0) * torch.FloatTensor([1, 0])
    angles = torch.FloatTensor(angles_vals)
    theta = angles.repeat(xy0.size(0), 1)
    theta = theta.transpose(0, 1).contiguous().view(-1, 1)
    xmin_ymin = xy0.repeat(int(theta.size(0) / xy0.size(0)), 1)
    xmax_ymax = xy2.repeat(int(theta.size(0) / xy2.size(0)), 1)
    widths_heights = dwh * scales
    widths_heights = widths_heights.repeat(int(theta.size(0) / widths_heights.size(0)), 1)
    u = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    l = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=1)
    R = torch.stack([u, l], dim=1)
    xy0R = torch.matmul(R, xy0.transpose(1, 0) - stride / 2 + 0.5) + stride / 2 - 0.5
    xy1R = torch.matmul(R, xy1.transpose(1, 0) - stride / 2 + 0.5) + stride / 2 - 0.5
    xy2R = torch.matmul(R, xy2.transpose(1, 0) - stride / 2 + 0.5) + stride / 2 - 0.5
    xy3R = torch.matmul(R, xy3.transpose(1, 0) - stride / 2 + 0.5) + stride / 2 - 0.5
    xy0R = xy0R.permute(0, 2, 1).contiguous().view(-1, 2)
    xy1R = xy1R.permute(0, 2, 1).contiguous().view(-1, 2)
    xy2R = xy2R.permute(0, 2, 1).contiguous().view(-1, 2)
    xy3R = xy3R.permute(0, 2, 1).contiguous().view(-1, 2)
    anchors_axis = torch.cat([xmin_ymin, xmax_ymax], dim=1)
    anchors_rotated = order_points(torch.stack([xy0R, xy1R, xy2R, xy3R], dim=1)).view(-1, 8)
    return anchors_axis, anchors_rotated


def rotate_boxes(boxes, points=False):
    """
    Rotate target bounding boxes 
    
    Input:  
        Target boxes (xmin_ymin, width_height, theta)
    Output:
        boxes_axis (xmin_ymin, xmax_ymax, theta)
        boxes_rotated (xy0, xy1, xy2, xy3)
    """
    u = torch.stack([torch.cos(boxes[:, (4)]), torch.sin(boxes[:, (4)])], dim=1)
    l = torch.stack([-torch.sin(boxes[:, (4)]), torch.cos(boxes[:, (4)])], dim=1)
    R = torch.stack([u, l], dim=1)
    if points:
        cents = torch.stack([(boxes[:, (0)] + boxes[:, (2)]) / 2, (boxes[:, (1)] + boxes[:, (3)]) / 2], 1).transpose(1, 0)
        boxes_rotated = torch.stack([boxes[:, (0)], boxes[:, (1)], boxes[:, (2)], boxes[:, (1)], boxes[:, (2)], boxes[:, (3)], boxes[:, (0)], boxes[:, (3)], boxes[:, (-2)], boxes[:, (-1)]], 1)
    else:
        cents = torch.stack([boxes[:, (0)] + (boxes[:, (2)] - 1) / 2, boxes[:, (1)] + (boxes[:, (3)] - 1) / 2], 1).transpose(1, 0)
        boxes_rotated = torch.stack([boxes[:, (0)], boxes[:, (1)], boxes[:, (0)] + boxes[:, (2)] - 1, boxes[:, (1)], boxes[:, (0)] + boxes[:, (2)] - 1, boxes[:, (1)] + boxes[:, (3)] - 1, boxes[:, (0)], boxes[:, (1)] + boxes[:, (3)] - 1, boxes[:, (-2)], boxes[:, (-1)]], 1)
    xy0R = torch.matmul(R, boxes_rotated[:, :2].transpose(1, 0) - cents) + cents
    xy1R = torch.matmul(R, boxes_rotated[:, 2:4].transpose(1, 0) - cents) + cents
    xy2R = torch.matmul(R, boxes_rotated[:, 4:6].transpose(1, 0) - cents) + cents
    xy3R = torch.matmul(R, boxes_rotated[:, 6:8].transpose(1, 0) - cents) + cents
    xy0R = torch.stack([xy0R[(i), :, (i)] for i in range(xy0R.size(0))])
    xy1R = torch.stack([xy1R[(i), :, (i)] for i in range(xy1R.size(0))])
    xy2R = torch.stack([xy2R[(i), :, (i)] for i in range(xy2R.size(0))])
    xy3R = torch.stack([xy3R[(i), :, (i)] for i in range(xy3R.size(0))])
    boxes_axis = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4] - 1, torch.sin(boxes[:, (-1), (None)]), torch.cos(boxes[:, (-1), (None)])], 1)
    boxes_rotated = order_points(torch.stack([xy0R, xy1R, xy2R, xy3R], dim=1)).view(-1, 8)
    return boxes_axis, boxes_rotated


def nms_rotated(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    """Non Maximum Suppression"""
    if torch.cuda.is_available():
        return nms_cuda(all_scores.float(), all_boxes.float(), all_classes.float(), nms, ndetections, True)
    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 6), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)
    for batch in range(batch_size):
        keep = (all_scores[(batch), :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[(batch), (keep), :].view(-1, 6)
        classes = all_classes[batch, keep].view(-1)
        theta = torch.atan2(boxes[:, (-2)], boxes[:, (-1)])
        boxes_theta = torch.cat([boxes[:, :-2], theta[:, (None)]], dim=1)
        if scores.nelement() == 0:
            continue
        scores, indices = torch.sort(scores, descending=True)
        boxes, boxes_theta, classes = boxes[indices], boxes_theta[indices], classes[indices]
        areas = (boxes_theta[:, (2)] - boxes_theta[:, (0)] + 1) * (boxes_theta[:, (3)] - boxes_theta[:, (1)] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)
        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break
            boxes_axis, boxes_rotated = rotate_boxes(boxes_theta, points=True)
            overlap, inter = iou(boxes_rotated.contiguous().view(-1), boxes_rotated[(i), :].contiguous().view(-1))
            inter = inter.squeeze()
            criterion = (scores > scores[i]) | (inter / (areas + areas[i] - inter) <= nms) | (classes != classes[i])
            criterion[i] = 1
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[(criterion.nonzero()), :].view(-1, 6)
            boxes_theta = boxes_theta[(criterion.nonzero()), :].view(-1, 5)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0
        out_scores[(batch), :i + 1] = scores[:i + 1]
        out_boxes[(batch), :i + 1, :] = boxes[:i + 1, :]
        out_classes[(batch), :i + 1] = classes[:i + 1]
    return out_scores, out_boxes, out_classes


def box2delta_rotated(boxes, anchors):
    """Convert boxes to deltas from anchors"""
    anchors_wh = anchors[:, 2:4] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:4] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
    boxes_sin = boxes[:, (4)]
    boxes_cos = boxes[:, (5)]
    return torch.cat([(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh), boxes_sin[:, (None)], boxes_cos[:, (None)]], 1)


def snap_to_anchors_rotated(boxes, size, stride, anchors, num_classes, device, anchor_ious):
    """Snap target boxes (x, y, w, h, a) to anchors"""
    anchors_axis, anchors_rotated = anchors
    num_anchors = anchors_rotated.size()[0] if anchors_rotated is not None else 1
    width, height = int(size[0] / stride), int(size[1] / stride)
    if boxes.nelement() == 0:
        return torch.zeros([num_anchors, num_classes, height, width], device=device), torch.zeros([num_anchors, 6, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device)
    boxes, classes = boxes.split(5, dim=1)
    boxes_axis, boxes_rotated = rotate_boxes(boxes)
    boxes_axis = boxes_axis
    boxes_rotated = boxes_rotated
    anchors_axis = anchors_axis
    anchors_rotated = anchors_rotated
    x, y = torch.meshgrid([torch.arange(0, size[i], stride, device=device, dtype=classes.dtype) for i in range(2)])
    xy_2corners = torch.stack((x, y, x, y), 2).unsqueeze(0)
    xy_4corners = torch.stack((x, y, x, y, x, y, x, y), 2).unsqueeze(0)
    anchors_axis = (xy_2corners + anchors_axis.view(-1, 1, 1, 4)).contiguous().view(-1, 4)
    anchors_rotated = (xy_4corners + anchors_rotated.view(-1, 1, 1, 8)).contiguous().view(-1, 8)
    if torch.cuda.is_available():
        iou = iou_cuda
    overlap = iou(boxes_rotated.contiguous().view(-1), anchors_rotated.contiguous().view(-1))[0]
    overlap, indices = overlap.max(1)
    box_target = box2delta_rotated(boxes_axis[indices], anchors_axis)
    box_target = box_target.view(num_anchors, 1, width, height, 6)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()
    depth = torch.ones_like(overlap, device=device) * -1
    depth[overlap < anchor_ious[0]] = 0
    depth[overlap >= anchor_ious[1]] = classes[indices][overlap >= anchor_ious[1]].squeeze() + 1
    depth = depth.view(num_anchors, width, height).transpose(1, 2).contiguous()
    cls_target = torch.zeros((anchors_axis.size()[0], num_classes + 1), device=device, dtype=boxes_axis.dtype)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[overlap < anchor_ious[0]] = num_classes
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()
    return cls_target.view(num_anchors, num_classes, height, width), box_target.view(num_anchors, 6, height, width), depth.view(num_anchors, 1, height, width)


class Model(nn.Module):
    """RetinaNet - https://arxiv.org/abs/1708.02002"""

    def __init__(self, backbones='ResNet50FPN', classes=80, ratios=[1.0, 2.0, 0.5], scales=[(4 * 2 ** (i / 3)) for i in range(3)], angles=None, rotated_bbox=False, anchor_ious=[0.4, 0.5], config={}):
        super().__init__()
        if not isinstance(backbones, list):
            backbones = [backbones]
        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.name = 'RetinaNet'
        self.exporting = False
        self.rotated_bbox = rotated_bbox
        self.anchor_ious = anchor_ious
        self.ratios = ratios
        self.scales = scales
        self.angles = angles if angles is not None else [-np.pi / 6, 0, np.pi / 6] if self.rotated_bbox else None
        self.anchors = {}
        self.classes = classes
        self.threshold = config.get('threshold', 0.05)
        self.top_n = config.get('top_n', 1000)
        self.nms = config.get('nms', 0.5)
        self.detections = config.get('detections', 100)
        self.stride = max([b.stride for _, b in self.backbones.items()])

        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)
        self.num_anchors = len(self.ratios) * len(self.scales)
        self.num_anchors = self.num_anchors if not self.rotated_bbox else self.num_anchors * len(self.angles)
        self.cls_head = make_head(classes * self.num_anchors)
        self.box_head = make_head(4 * self.num_anchors) if not self.rotated_bbox else make_head(6 * self.num_anchors)
        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss(beta=0.11)

    def __repr__(self):
        return '\n'.join(['     model: {}'.format(self.name), '  backbone: {}'.format(', '.join([k for k, _ in self.backbones.items()])), '   classes: {}, anchors: {}'.format(self.classes, self.num_anchors)])

    def initialize(self, pre_trained):
        if pre_trained:
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))
            None
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            if self.rotated_bbox:
                ignored += ['box_head.8.bias', 'box_head.8.weight']
            weights = {k: v for k, v in chk['state_dict'].items() if k not in ignored}
            state_dict.update(weights)
            self.load_state_dict(state_dict)
            del chk, weights
            torch.cuda.empty_cache()
        else:
            for _, backbone in self.backbones.items():
                backbone.initialize()

            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)

        def initialize_prior(layer):
            pi = 0.01
            b = -math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)
        self.cls_head[-1].apply(initialize_prior)
        if self.rotated_bbox:
            self.box_head[-1].apply(initialize_prior)

    def forward(self, x, rotated_bbox=None):
        if self.training:
            x, targets = x
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))
        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]
        if self.training:
            return self._compute_loss(x, cls_heads, box_heads, targets.float())
        cls_heads = [cls_head.sigmoid() for cls_head in cls_heads]
        if self.exporting:
            self.strides = [(x.shape[-1] // cls_head.shape[-1]) for cls_head in cls_heads]
            return cls_heads, box_heads
        global nms, generate_anchors
        if self.rotated_bbox:
            nms = nms_rotated
            generate_anchors = generate_anchors_rotated
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            stride = x.shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)
            decoded.append(decode(cls_head, box_head, stride, self.threshold, self.top_n, self.anchors[stride], self.rotated_bbox))
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms, self.detections)

    def _extract_targets(self, targets, stride, size):
        global generate_anchors, snap_to_anchors
        if self.rotated_bbox:
            generate_anchors = generate_anchors_rotated
            snap_to_anchors = snap_to_anchors_rotated
        cls_target, box_target, depth = [], [], []
        for target in targets:
            target = target[target[:, (-1)] > -1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales, self.angles)
            anchors = self.anchors[stride]
            if not self.rotated_bbox:
                anchors = anchors
            snapped = snap_to_anchors(target, [(s * stride) for s in size[::-1]], stride, anchors, self.classes, targets.device, self.anchor_ious)
            for l, s in zip((cls_target, box_target, depth), snapped):
                l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def _compute_loss(self, x, cls_heads, box_heads, targets):
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_head, box_head in zip(cls_heads, box_heads):
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]
            cls_target, box_target, depth = self._extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))
            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())
            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())
        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def save(self, state):
        checkpoint = {'backbone': [k for k, _ in self.backbones.items()], 'classes': self.classes, 'state_dict': self.state_dict(), 'ratios': self.ratios, 'scales': self.scales}
        if self.rotated_bbox and self.angles:
            checkpoint['angles'] = self.angles
        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in state:
                checkpoint[key] = state[key]
        torch.save(checkpoint, state['path'])

    @classmethod
    def load(cls, filename, rotated_bbox=False):
        if not os.path.isfile(filename):
            raise ValueError('No checkpoint {}'.format(filename))
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        kwargs = {}
        for i in ['ratios', 'scales', 'angles']:
            if i in checkpoint:
                kwargs[i] = checkpoint[i]
        if 'angles' in checkpoint or rotated_bbox:
            kwargs['rotated_bbox'] = True
        model = cls(backbones=checkpoint['backbone'], classes=checkpoint['classes'], **kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        state = {}
        for key in ('iteration', 'optimizer', 'scheduler'):
            if key in checkpoint:
                state[key] = checkpoint[key]
        del checkpoint
        torch.cuda.empty_cache()
        return model, state

    def export(self, size, batch, precision, calibration_files, calibration_table, verbose, onnx_only=False):
        import torch.onnx.symbolic_opset10 as onnx_symbolic

        def upsample_nearest2d(g, input, output_size, *args):
            scales = g.op('Constant', value_t=torch.tensor([1.0, 1.0, 2.0, 2.0]))
            return g.op('Resize', input, scales, mode_s='nearest')
        onnx_symbolic.upsample_nearest2d = upsample_nearest2d
        None
        self.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([1, 3, *size])
        extra_args = {'opset_version': 10, 'verbose': verbose}
        torch.onnx.export(self, zero_input, onnx_bytes, **extra_args)
        self.exporting = False
        if onnx_only:
            return onnx_bytes.getvalue()
        model_name = '_'.join([k for k, _ in self.backbones.items()])
        anchors = []
        if not self.rotated_bbox:
            anchors = [generate_anchors(stride, self.ratios, self.scales, self.angles).view(-1).tolist() for stride in self.strides]
        else:
            anchors = [generate_anchors_rotated(stride, self.ratios, self.scales, self.angles)[0].view(-1).tolist() for stride in self.strides]
        batch = 1
        return Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision, self.threshold, self.top_n, anchors, self.rotated_bbox, self.nms, self.detections, calibration_files, model_name, calibration_table, verbose)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FixedBatchNorm2d,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_NVIDIA_retinanet_examples(_paritybench_base):
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

