import sys
_module = sys.modules[__name__]
del sys
mAP_evaluation = _module
config = _module
dataset = _module
loss = _module
model = _module
utils = _module
test_dataset = _module
test_video = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import math


from torchvision.ops.boxes import nms as nms_torch


import numpy as np


from torch.utils.data import DataLoader


from torchvision import transforms


def calc_iou(a, b):
    area = (b[:, (2)] - b[:, (0)]) * (b[:, (3)] - b[:, (1)])
    iw = torch.min(torch.unsqueeze(a[:, (2)], dim=1), b[:, (2)]) - torch.max(
        torch.unsqueeze(a[:, (0)], 1), b[:, (0)])
    ih = torch.min(torch.unsqueeze(a[:, (3)], dim=1), b[:, (3)]) - torch.max(
        torch.unsqueeze(a[:, (1)], 1), b[:, (1)])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, (2)] - a[:, (0)]) * (a[:, (3)] - a[:, (1)]),
        dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-08)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchor = anchors[(0), :, :]
        anchor_widths = anchor[:, (2)] - anchor[:, (0)]
        anchor_heights = anchor[:, (3)] - anchor[:, (1)]
        anchor_ctr_x = anchor[:, (0)] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, (1)] + 0.5 * anchor_heights
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            regression = regressions[(j), :, :]
            bbox_annotation = annotations[(j), :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, (4)] != -1]
            if bbox_annotation.shape[0] == 0:
                if torch.is_available():
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())
                continue
            classification = torch.clamp(classification, 0.0001, 1.0 - 0.0001)
            IoU = calc_iou(anchors[(0), :, :], bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            targets = torch.ones(classification.shape) * -1
            if torch.is_available():
                targets = targets
            targets[(torch.lt(IoU_max, 0.4)), :] = 0
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[(IoU_argmax), :]
            targets[(positive_indices), :] = 0
            targets[positive_indices, assigned_annotations[positive_indices,
                4].long()] = 1
            alpha_factor = torch.ones(targets.shape) * alpha
            if torch.is_available():
                alpha_factor = alpha_factor
            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor,
                1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 -
                classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) *
                torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            zeros = torch.zeros(cls_loss.shape)
            if torch.is_available():
                zeros = zeros
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)
            classification_losses.append(cls_loss.sum() / torch.clamp(
                num_positive_anchors.float(), min=1.0))
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
                targets = torch.stack((targets_dx, targets_dy, targets_dw,
                    targets_dh))
                targets = targets.t()
                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                if torch.is_available():
                    norm = norm
                targets = targets / norm
                regression_diff = torch.abs(targets - regression[(
                    positive_indices), :])
                regression_loss = torch.where(torch.le(regression_diff, 1.0 /
                    9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), 
                    regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            elif torch.is_available():
                regression_losses.append(torch.tensor(0).float())
            else:
                regression_losses.append(torch.tensor(0).float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True
            ), torch.stack(regression_losses).mean(dim=0, keepdim=True)


class ConvBlock(nn.Module):

    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(num_channels, num_channels,
            kernel_size=3, stride=1, padding=1, groups=num_channels), nn.
            Conv2d(num_channels, num_channels, kernel_size=1, stride=1,
            padding=0), nn.BatchNorm2d(num_features=num_channels, momentum=
            0.9997, eps=4e-05), nn.ReLU())

    def forward(self, input):
        return self.conv(input)


class BiFPN(nn.Module):

    def __init__(self, num_channels, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
            P7_0 -------------------------- P7_2 -------->

            P6_0 ---------- P6_1 ---------- P6_2 -------->

            P5_0 ---------- P5_1 ---------- P5_2 -------->

            P4_0 ---------- P4_1 ---------- P4_2 -------->

            P3_0 -------------------------- P3_2 -------->
        """
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.
            p6_upsample(p7_in))
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.
            p5_upsample(p6_up))
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.
            p4_upsample(p5_up))
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.
            p3_upsample(p4_up))
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(weight[0] * p4_in + weight[1] * p4_up + 
            weight[2] * self.p4_downsample(p3_out))
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(weight[0] * p5_in + weight[1] * p5_up + 
            weight[2] * self.p5_downsample(p4_out))
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(weight[0] * p6_in + weight[1] * p6_up + 
            weight[2] * self.p6_downsample(p5_out))
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.
            p7_downsample(p6_out))
        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):

    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3,
                stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3,
            stride=1, padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        output = inputs.permute(0, 2, 3, 1)
        return output.contiguous().view(output.shape[0], -1, 4)


class Classifier(nn.Module):

    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3,
                stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes,
            kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        output = inputs.contiguous().view(inputs.shape[0], inputs.shape[1],
            inputs.shape[2], self.num_anchors, self.num_classes)
        return output.contiguous().view(output.shape[0], -1, self.num_classes)


class EfficientNet(nn.Module):

    def __init__(self):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained('efficientnet-b0')
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
        return feature_maps[1:]


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, (4)], thresh)


class EfficientDet(nn.Module):

    def __init__(self, num_anchors=9, num_classes=20, compound_coef=0):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef
        self.num_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.
            compound_coef]
        self.conv3 = nn.Conv2d(40, self.num_channels, kernel_size=1, stride
            =1, padding=0)
        self.conv4 = nn.Conv2d(80, self.num_channels, kernel_size=1, stride
            =1, padding=0)
        self.conv5 = nn.Conv2d(192, self.num_channels, kernel_size=1,
            stride=1, padding=0)
        self.conv6 = nn.Conv2d(192, self.num_channels, kernel_size=3,
            stride=2, padding=1)
        self.conv7 = nn.Sequential(nn.ReLU(), nn.Conv2d(self.num_channels,
            self.num_channels, kernel_size=3, stride=2, padding=1))
        self.bifpn = nn.Sequential(*[BiFPN(self.num_channels) for _ in
            range(min(2 + self.compound_coef, 8))])
        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.num_channels,
            num_anchors=num_anchors, num_layers=3 + self.compound_coef // 3)
        self.classifier = Classifier(in_channels=self.num_channels,
            num_anchors=num_anchors, num_classes=num_classes, num_layers=3 +
            self.compound_coef // 3)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior)
            )
        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)
        self.backbone_net = EfficientNet()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        if len(inputs) == 2:
            is_training = True
            img_batch, annotations = inputs
        else:
            is_training = False
            img_batch = inputs
        c3, c4, c5 = self.backbone_net(img_batch)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        features = [p3, p4, p5, p6, p7]
        features = self.bifpn(features)
        regression = torch.cat([self.regressor(feature) for feature in
            features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in
            features], dim=1)
        anchors = self.anchors(img_batch)
        if is_training:
            return self.focalLoss(classification, regression, anchors,
                annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch
                )
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[(0), :, (0)]
            if scores_over_thresh.sum() == 0:
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
            classification = classification[:, (scores_over_thresh), :]
            transformed_anchors = transformed_anchors[:, (
                scores_over_thresh), :]
            scores = scores[:, (scores_over_thresh), :]
            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores],
                dim=2)[(0), :, :], 0.5)
            nms_scores, nms_class = classification[(0), (anchors_nms_idx), :
                ].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[(0), (
                anchors_nms_idx), :]]


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.
                float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).
                astype(np.float32))
        else:
            self.std = std
        if torch.is_available():
            self.mean = self.mean
            self.std = self.std

    def forward(self, boxes, deltas):
        widths = boxes[:, :, (2)] - boxes[:, :, (0)]
        heights = boxes[:, :, (3)] - boxes[:, :, (1)]
        ctr_x = boxes[:, :, (0)] + 0.5 * widths
        ctr_y = boxes[:, :, (1)] + 0.5 * heights
        dx = deltas[:, :, (0)] * self.std[0] + self.mean[0]
        dy = deltas[:, :, (1)] * self.std[1] + self.mean[1]
        dw = deltas[:, :, (2)] * self.std[2] + self.mean[2]
        dh = deltas[:, :, (3)] * self.std[3] + self.mean[3]
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1,
            pred_boxes_x2, pred_boxes_y2], dim=2)
        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, (0)] = torch.clamp(boxes[:, :, (0)], min=0)
        boxes[:, :, (1)] = torch.clamp(boxes[:, :, (1)], min=0)
        boxes[:, :, (2)] = torch.clamp(boxes[:, :, (2)], max=width)
        boxes[:, :, (3)] = torch.clamp(boxes[:, :, (3)], max=height)
        return boxes


def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, (2)] * anchors[:, (3)]
    anchors[:, (2)] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, (3)] = anchors[:, (2)] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, (2)] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, (3)] * 0.5, (2, 1)).T
    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
        shift_y.ravel())).transpose()
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)
        ).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors


class Anchors(nn.Module):

    def __init__(self, pyramid_levels=None, strides=None, sizes=None,
        ratios=None, scales=None):
        super(Anchors, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [(2 ** x) for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [(2 ** (x + 2)) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
                )

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [((image_shape + 2 ** x - 1) // 2 ** x) for x in
            self.pyramid_levels]
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=
                self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx],
                anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.is_available():
            anchors = anchors
        return anchors


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_signatrix_efficientdet(_paritybench_base):
    pass
    def test_000(self):
        self._check(Classifier(*[], **{'in_channels': 4, 'num_anchors': 4, 'num_classes': 4, 'num_layers': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ClipBoxes(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ConvBlock(*[], **{'num_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Regressor(*[], **{'in_channels': 4, 'num_anchors': 4, 'num_layers': 1}), [torch.rand([4, 4, 4, 4])], {})

