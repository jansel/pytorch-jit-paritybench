import sys
_module = sys.modules[__name__]
del sys
coco_validation = _module
retinanet = _module
anchors = _module
coco_eval = _module
csv_eval = _module
dataloader = _module
losses = _module
model = _module
oid_dataset = _module
utils = _module
train = _module
visualize = _module

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


import numpy as np


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


import collections


import torch.optim as optim


from torch.utils.data import DataLoader


import copy


from torch.utils.data import Dataset


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
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
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32))
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))


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
                if torch.cuda.is_available():
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
            if torch.cuda.is_available():
                targets = targets
            targets[(torch.lt(IoU_max, 0.4)), :] = 0
            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[(IoU_argmax), :]
            targets[(positive_indices), :] = 0
            targets[positive_indices, assigned_annotations[positive_indices,
                4].long()] = 1
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape) * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor,
                1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 -
                classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) *
                torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss,
                    torch.zeros(cls_loss.shape))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss,
                    torch.zeros(cls_loss.shape))
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
                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                negative_indices = 1 + ~positive_indices
                regression_diff = torch.abs(targets - regression[(
                    positive_indices), :])
                regression_loss = torch.where(torch.le(regression_diff, 1.0 /
                    9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2), 
                    regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            elif torch.cuda.is_available():
                regression_losses.append(torch.tensor(0).float())
            else:
                regression_losses.append(torch.tensor(0).float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True
            ), torch.stack(regression_losses).mean(dim=0, keepdim=True)


class PyramidFeatures(nn.Module):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=
            1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=
            1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=
            1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2,
            padding=1)
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        P6_x = self.P6(C5)
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=
            3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=9, num_classes=80,
        prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes,
            kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels,
                self.layer3[layers[2] - 1].conv2.out_channels, self.layer4[
                layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                self.layer3[layers[2] - 1].conv3.out_channels, self.layer4[
                layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f'Block type {block} not understood')
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=
            num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 -
            prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(feature) for feature in
            features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for
            feature in features], dim=1)
        anchors = self.anchors(img_batch)
        if self.training:
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
            anchors_nms_idx = nms(transformed_anchors[(0), :, :], scores[(0
                ), :, (0)], 0.5)
            nms_scores, nms_class = classification[(0), (anchors_nms_idx), :
                ].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[(0), (
                anchors_nms_idx), :]]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(
                    np.float32))
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(
                    np.float32))
        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).
                    astype(np.float32))
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).
                    astype(np.float32))
        else:
            self.std = std

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

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, (0)] = torch.clamp(boxes[:, :, (0)], min=0)
        boxes[:, :, (1)] = torch.clamp(boxes[:, :, (1)], min=0)
        boxes[:, :, (2)] = torch.clamp(boxes[:, :, (2)], max=width)
        boxes[:, :, (3)] = torch.clamp(boxes[:, :, (3)], max=height)
        return boxes


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yhenon_pytorch_retinanet(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Anchors(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BBoxTransform(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ClassificationModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(ClipBoxes(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(RegressionModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

