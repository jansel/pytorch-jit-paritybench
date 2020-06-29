import sys
_module = sys.modules[__name__]
del sys
anchors = _module
dataloader = _module
detect = _module
eval_widerface = _module
losses = _module
model = _module
pose = _module
datasets = _module
detect_image = _module
hopenet = _module
test_alexnet = _module
test_hopenet = _module
test_on_video = _module
test_on_video_dlib = _module
test_on_video_dockerface = _module
test_resnet50_regression = _module
train_alexnet = _module
train_hopenet = _module
train_resnet50_regression = _module
utils = _module
pose_detect = _module
torchvision_model = _module
train = _module
utils = _module
video_detect = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.utils.data.sampler import Sampler


from torch.utils.data import Dataset


import torch.nn.functional as F


import random


import math


import numpy as numpy


import time


import torch.utils.model_zoo as model_zoo


from torch.autograd import Variable


from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn


from collections import OrderedDict


import torch.optim as optim


from torch.optim import lr_scheduler


import torch.distributed as dist


from torch.utils.data.distributed import DistributedSampler


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([1, 1, 1])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    num_anchors = len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, 1)).T
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
            self.pyramid_levels = [3, 4, 5]
        if strides is None:
            self.strides = [(2 ** x) for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** 4.0, 2 ** 6.0, 2 ** 8.0]
        if ratios is None:
            self.ratios = np.array([1, 1, 1])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1 / 2.0), 2 ** 1.0])

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


class LossLayer(nn.Module):

    def __init__(self):
        super(LossLayer, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self, classifications, bbox_regressions, ldm_regressions,
        anchors, annotations):
        batch_size = classifications.shape[0]
        classification_losses = []
        bbox_regression_losses = []
        ldm_regression_losses = []
        anchor = anchors[(0), :, :]
        anchor_widths = anchor[:, (2)] - anchor[:, (0)]
        anchor_heights = anchor[:, (3)] - anchor[:, (1)]
        anchor_ctr_x = anchor[:, (0)] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, (1)] + 0.5 * anchor_heights
        positive_indices_list = []
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            bbox_regression = bbox_regressions[(j), :, :]
            ldm_regression = ldm_regressions[(j), :, :]
            annotation = annotations[(j), :, :]
            annotation = annotation[annotation[:, (0)] > 0]
            bbox_annotation = annotation[:, :4]
            ldm_annotation = annotation[:, 4:]
            if bbox_annotation.shape[0] == 0:
                bbox_regression_losses.append(torch.tensor(0.0,
                    requires_grad=True))
                classification_losses.append(torch.tensor(0.0,
                    requires_grad=True))
                ldm_regression_losses.append(torch.tensor(0.0,
                    requires_grad=True))
                positive_indices_list.append([])
                continue
            IoU = calc_iou(anchors[(0), :, :], bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            targets = torch.ones(classification.shape) * -1
            targets = targets
            negative_indices = torch.lt(IoU_max, 0.3)
            targets[(negative_indices), :] = 0
            targets[negative_indices, 1] = 1
            positive_indices = torch.ge(IoU_max, 0.5)
            positive_indices_list.append(positive_indices)
            num_positive_anchors = positive_indices.sum()
            keep_negative_anchors = num_positive_anchors * 3
            bbox_assigned_annotations = bbox_annotation[(IoU_argmax), :]
            ldm_assigned_annotations = ldm_annotation[(IoU_argmax), :]
            targets[(positive_indices), :] = 0
            targets[positive_indices, 0] = 1
            ldm_sum = ldm_assigned_annotations.sum(dim=1)
            ge0_mask = ldm_sum > 0
            ldm_positive_indices = ge0_mask & positive_indices
            negative_losses = classification[negative_indices, 1] * -1
            sorted_losses, _ = torch.sort(negative_losses, descending=True)
            if sorted_losses.numel() > keep_negative_anchors:
                sorted_losses = sorted_losses[:keep_negative_anchors]
            positive_losses = classification[positive_indices, 0] * -1
            focal_loss = False
            if focal_loss:
                alpha = 0.25
                gamma = 2.0
                alpha_factor = torch.ones(targets.shape) * alpha
                alpha_factor = torch.where(torch.eq(targets, 1.0),
                    alpha_factor, 1.0 - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 -
                    classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                bce = -(targets * torch.log(classification) + (1.0 -
                    targets) * torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss,
                    torch.zeros(cls_loss.shape))
                classification_losses.append(cls_loss.sum() / torch.clamp(
                    num_positive_anchors.float(), min=1.0))
            elif positive_indices.sum() > 0:
                classification_losses.append(positive_losses.mean() +
                    sorted_losses.mean())
            else:
                classification_losses.append(torch.tensor(0.0,
                    requires_grad=True))
            if positive_indices.sum() > 0:
                bbox_assigned_annotations = bbox_assigned_annotations[(
                    positive_indices), :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                gt_widths = bbox_assigned_annotations[:, (2)
                    ] - bbox_assigned_annotations[:, (0)]
                gt_heights = bbox_assigned_annotations[:, (3)
                    ] - bbox_assigned_annotations[:, (1)]
                gt_ctr_x = bbox_assigned_annotations[:, (0)] + 0.5 * gt_widths
                gt_ctr_y = bbox_assigned_annotations[:, (1)] + 0.5 * gt_heights
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi +
                    1e-14)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi
                     + 1e-14)
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                bbox_targets = torch.stack((targets_dx, targets_dy,
                    targets_dw, targets_dh))
                bbox_targets = bbox_targets.t()
                bbox_targets = bbox_targets / torch.Tensor([[0.1, 0.1, 0.2,
                    0.2]])
                bbox_regression_loss = self.smoothl1(bbox_targets,
                    bbox_regression[(positive_indices), :])
                bbox_regression_losses.append(bbox_regression_loss)
            else:
                bbox_regression_losses.append(torch.tensor(0.0,
                    requires_grad=True))
            if ldm_positive_indices.sum() > 0:
                ldm_assigned_annotations = ldm_assigned_annotations[(
                    ldm_positive_indices), :]
                anchor_widths_l = anchor_widths[ldm_positive_indices]
                anchor_heights_l = anchor_heights[ldm_positive_indices]
                anchor_ctr_x_l = anchor_ctr_x[ldm_positive_indices]
                anchor_ctr_y_l = anchor_ctr_y[ldm_positive_indices]
                l0_x = (ldm_assigned_annotations[:, (0)] - anchor_ctr_x_l) / (
                    anchor_widths_l + 1e-14)
                l0_y = (ldm_assigned_annotations[:, (1)] - anchor_ctr_y_l) / (
                    anchor_heights_l + 1e-14)
                l1_x = (ldm_assigned_annotations[:, (2)] - anchor_ctr_x_l) / (
                    anchor_widths_l + 1e-14)
                l1_y = (ldm_assigned_annotations[:, (3)] - anchor_ctr_y_l) / (
                    anchor_heights_l + 1e-14)
                l2_x = (ldm_assigned_annotations[:, (4)] - anchor_ctr_x_l) / (
                    anchor_widths_l + 1e-14)
                l2_y = (ldm_assigned_annotations[:, (5)] - anchor_ctr_y_l) / (
                    anchor_heights_l + 1e-14)
                l3_x = (ldm_assigned_annotations[:, (6)] - anchor_ctr_x_l) / (
                    anchor_widths_l + 1e-14)
                l3_y = (ldm_assigned_annotations[:, (7)] - anchor_ctr_y_l) / (
                    anchor_heights_l + 1e-14)
                l4_x = (ldm_assigned_annotations[:, (8)] - anchor_ctr_x_l) / (
                    anchor_widths_l + 1e-14)
                l4_y = (ldm_assigned_annotations[:, (9)] - anchor_ctr_y_l) / (
                    anchor_heights_l + 1e-14)
                ldm_targets = torch.stack((l0_x, l0_y, l1_x, l1_y, l2_x,
                    l2_y, l3_x, l3_y, l4_x, l4_y))
                ldm_targets = ldm_targets.t()
                scale = torch.ones(1, 10) * 0.1
                ldm_targets = ldm_targets / scale
                ldm_regression_loss = self.smoothl1(ldm_targets,
                    ldm_regression[(ldm_positive_indices), :])
                ldm_regression_losses.append(ldm_regression_loss)
            else:
                ldm_regression_losses.append(torch.tensor(0.0,
                    requires_grad=True))
        return torch.stack(classification_losses), torch.stack(
            bbox_regression_losses), torch.stack(ldm_regression_losses)


class PyramidFeatures(nn.Module):

    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
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
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2,
            padding=1)
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=
            1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            stride=1, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)
        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)
        P6_x = self.P6(C5)
        return [P2_x, P3_x, P4_x, P5_x, P6_x]


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2,
            kernel_size=(1, 1), stride=1)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        return out.contiguous().view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(
            1, 1), stride=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=
            (1, 1), stride=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 10)


class ClassHead_(nn.Module):

    def __init__(self, inchannels=256, num_anchors=3):
        super(ClassHead_, self).__init__()
        self.num_anchors = num_anchors
        self.feature_head = self._make_head(self.num_anchors * 2)
        self.output_act = nn.LogSoftmax(dim=-1)

    def _make_head(self, out_size):
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
                ]
        layers += [nn.Conv2d(256, out_size, 3, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_head(x)
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        return out.contiguous().view(out.shape[0], -1, 2)


class BboxHead_(nn.Module):

    def __init__(self, inchannels=256, num_anchors=3):
        super(BboxHead_, self).__init__()
        self.num_anchors = num_anchors
        self.feature_head = self._make_head(self.num_anchors * 4)

    def _make_head(self, out_size):
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
                ]
        layers += [nn.Conv2d(256, out_size, 3, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_head(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class LandmarkHead_(nn.Module):

    def __init__(self, inchannels=256, num_anchors=3):
        super(LandmarkHead_, self).__init__()
        self.num_anchors = num_anchors
        self.feature_head = self._make_head(self.num_anchors * 10)

    def _make_head(self, out_size):
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
                ]
        layers += [nn.Conv2d(256, out_size, 3, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_head(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 10)


class CBR(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(CBR, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, outchannels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CB(nn.Module):

    def __init__(self, inchannels):
        super(CB, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, inchannels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(inchannels)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn(x)
        return x


class Concat(nn.Module):

    def forward(self, *feature):
        out = torch.cat(feature, dim=1)
        return out


class Context(nn.Module):

    def __init__(self, inchannels=256):
        super(Context, self).__init__()
        self.context_plain = inchannels // 2
        self.conv1 = CB(inchannels)
        self.conv2 = CBR(inchannels, self.context_plain)
        self.conv2_1 = CB(self.context_plain)
        self.conv2_2_1 = CBR(self.context_plain, self.context_plain)
        self.conv2_2_2 = CB(self.context_plain)
        self.concat = Concat()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.conv1(x)
        f2_ = self.conv2(x)
        f2 = self.conv2_1(f2_)
        f3 = self.conv2_2_1(f2_)
        f3 = self.conv2_2_2(f3)
        out = self.concat(f1, f2, f3)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, num_anchors=3):
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
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[
                layers[2] - 1].conv2.out_channels, self.layer4[layers[3] - 
                1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels,
                self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[
                layers[2] - 1].conv3.out_channels, self.layer4[layers[3] - 
                1].conv3.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],
            fpn_sizes[3])
        self.context = self._make_contextlayer()
        self.clsHead = ClassHead_()
        self.bboxHead = BboxHead_()
        self.ldmHead = LandmarkHead_()
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()
        self.losslayer = losses.LossLayer()
        self.freeze_bn()
        for layer in self.context:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_contextlayer(self, fpn_num=5, inchannels=256):
        context = nn.ModuleList()
        for i in range(fpn_num):
            context.append(Context())
        return context

    def _make_class_head(self, fpn_num=5, inchannels=512, anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=5, inchannels=512, anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=5, inchannels=512, anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_first_layer(self):
        """Freeze First layer"""
        for param in self.conv1.parameters():
            param.requires_grad = False

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
        features = self.fpn([x1, x2, x3, x4])
        bbox_regressions = torch.cat([self.bboxHead(feature) for feature in
            features], dim=1)
        ldm_regressions = torch.cat([self.ldmHead(feature) for feature in
            features], dim=1)
        classifications = torch.cat([self.clsHead(feature) for feature in
            features], dim=1)
        anchors = self.anchors(img_batch)
        if self.training:
            return self.losslayer(classifications, bbox_regressions,
                ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions,
                ldm_regressions, img_batch)
            return classifications, bboxes, landmarks


class Hopenet(nn.Module):

    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)
        return pre_yaw, pre_pitch, pre_roll


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
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
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=
            1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6,
            4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 
            4096), nn.ReLU(inplace=True))
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll


class ContextModule(nn.Module):

    def __init__(self, in_channels=256):
        super(ContextModule, self).__init__()
        self.det_conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels,
            kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels))
        self.det_context_conv1 = nn.Sequential(nn.Conv2d(in_channels, 
            in_channels // 2, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True))
        self.det_context_conv2 = nn.Sequential(nn.Conv2d(in_channels // 2, 
            in_channels // 2, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(in_channels // 2))
        self.det_context_conv3_1 = nn.Sequential(nn.Conv2d(in_channels // 2,
            in_channels // 2, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True))
        self.det_context_conv3_2 = nn.Sequential(nn.Conv2d(in_channels // 2,
            in_channels // 2, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(in_channels // 2))
        self.det_concat_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.det_conv1(x)
        x_ = self.det_context_conv1(x)
        x2 = self.det_context_conv2(x_)
        x3_ = self.det_context_conv3_1(x_)
        x3 = self.det_context_conv3_2(x3_)
        out = torch.cat((x1, x2, x3), 1)
        act_out = self.det_concat_relu(out)
        return act_out


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_blocks = nn.ModuleList()
        self.context_blocks = nn.ModuleList()
        self.aggr_blocks = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                continue
            lateral_block_module = nn.Sequential(nn.Conv2d(in_channels,
                out_channels, kernel_size=1, stride=1, padding=0), nn.
                BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            aggr_block_module = nn.Sequential(nn.Conv2d(out_channels,
                out_channels, kernel_size=3, stride=1, padding=1), nn.
                BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            context_block_module = ContextModule(out_channels)
            self.lateral_blocks.append(lateral_block_module)
            self.context_blocks.append(context_block_module)
            if i > 0:
                self.aggr_blocks.append(aggr_block_module)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        last_inner = self.lateral_blocks[-1](x[-1])
        results = []
        results.append(self.context_blocks[-1](last_inner))
        for feature, lateral_block, context_block, aggr_block in zip(x[:-1]
            [::-1], self.lateral_blocks[:-1][::-1], self.context_blocks[:-1
            ][::-1], self.aggr_blocks[::-1]):
            if not lateral_block:
                continue
            lateral_feature = lateral_block(feature)
            feat_shape = lateral_feature.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape,
                mode='nearest')
            last_inner = lateral_feature + inner_top_down
            last_inner = aggr_block(last_inner)
            results.insert(0, context_block(last_inner))
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2,
            kernel_size=(1, 1), stride=1, padding=0)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        return out.contiguous().view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(
            1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=
            (1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):

    def __init__(self, backbone, return_layers, anchor_nums=3):
        super(RetinaFace, self).__init__()
        assert backbone, 'Backbone can not be none!'
        assert len(return_layers
            ) > 0, 'There must be at least one return layers'
        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = 256
        in_channels_list = [in_channels_stage2 * 2, in_channels_stage2 * 4,
            in_channels_stage2 * 8]
        out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        self.ClassHead = self._make_class_head()
        self.BboxHead = self._make_bbox_head()
        self.LandmarkHead = self._make_landmark_head()
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()
        self.losslayer = losses.LossLayer()

    def _make_class_head(self, fpn_num=3, inchannels=512, anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=512, anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=512, anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

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
        out = self.body(img_batch)
        features = self.fpn(out)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i,
            feature in enumerate(features.values())], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i,
            feature in enumerate(features.values())], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i,
            feature in enumerate(features.values())], dim=1)
        anchors = self.anchors(img_batch)
        if self.training:
            return self.losslayer(classifications, bbox_regressions,
                ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions,
                ldm_regressions, img_batch)
            return classifications, bboxes, landmarks


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


class RegressionTransform(nn.Module):

    def __init__(self, mean=None, std_box=None, std_ldm=None):
        super(RegressionTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.
                float32))
        else:
            self.mean = mean
        if std_box is None:
            self.std_box = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).
                astype(np.float32))
        else:
            self.std_box = std_box
        if std_ldm is None:
            self.std_ldm = torch.ones(1, 10) * 0.1

    def forward(self, anchors, bbox_deltas, ldm_deltas, img):
        widths = anchors[:, :, (2)] - anchors[:, :, (0)]
        heights = anchors[:, :, (3)] - anchors[:, :, (1)]
        ctr_x = anchors[:, :, (0)] + 0.5 * widths
        ctr_y = anchors[:, :, (1)] + 0.5 * heights
        ldm_deltas = ldm_deltas * self.std_ldm
        bbox_deltas = bbox_deltas * self.std_box
        bbox_dx = bbox_deltas[:, :, (0)]
        bbox_dy = bbox_deltas[:, :, (1)]
        bbox_dw = bbox_deltas[:, :, (2)]
        bbox_dh = bbox_deltas[:, :, (3)]
        pred_ctr_x = ctr_x + bbox_dx * widths
        pred_ctr_y = ctr_y + bbox_dy * heights
        pred_w = torch.exp(bbox_dw) * widths
        pred_h = torch.exp(bbox_dh) * heights
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1,
            pred_boxes_x2, pred_boxes_y2], dim=2)
        pt0_x = ctr_x + ldm_deltas[:, :, (0)] * widths
        pt0_y = ctr_y + ldm_deltas[:, :, (1)] * heights
        pt1_x = ctr_x + ldm_deltas[:, :, (2)] * widths
        pt1_y = ctr_y + ldm_deltas[:, :, (3)] * heights
        pt2_x = ctr_x + ldm_deltas[:, :, (4)] * widths
        pt2_y = ctr_y + ldm_deltas[:, :, (5)] * heights
        pt3_x = ctr_x + ldm_deltas[:, :, (6)] * widths
        pt3_y = ctr_y + ldm_deltas[:, :, (7)] * heights
        pt4_x = ctr_x + ldm_deltas[:, :, (8)] * widths
        pt4_y = ctr_y + ldm_deltas[:, :, (9)] * heights
        pred_landmarks = torch.stack([pt0_x, pt0_y, pt1_x, pt1_y, pt2_x,
            pt2_y, pt3_x, pt3_y, pt4_x, pt4_y], dim=2)
        B, C, H, W = img.shape
        pred_boxes[:, :, ::2] = torch.clamp(pred_boxes[:, :, ::2], min=0, max=W
            )
        pred_boxes[:, :, 1::2] = torch.clamp(pred_boxes[:, :, 1::2], min=0,
            max=H)
        pred_landmarks[:, :, ::2] = torch.clamp(pred_landmarks[:, :, ::2],
            min=0, max=W)
        pred_landmarks[:, :, 1::2] = torch.clamp(pred_landmarks[:, :, 1::2],
            min=0, max=H)
        return pred_boxes, pred_landmarks


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_supernotman_RetinaFace_Pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Anchors(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(BboxHead(*[], **{}), [torch.rand([4, 512, 64, 64])], {})

    def test_003(self):
        self._check(BboxHead_(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    def test_004(self):
        self._check(CB(*[], **{'inchannels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(CBR(*[], **{'inchannels': 4, 'outchannels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ClassHead(*[], **{}), [torch.rand([4, 512, 64, 64])], {})

    def test_007(self):
        self._check(ClassHead_(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    @_fails_compile()
    def test_008(self):
        self._check(Context(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    def test_009(self):
        self._check(ContextModule(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    def test_010(self):
        self._check(LandmarkHead(*[], **{}), [torch.rand([4, 512, 64, 64])], {})

    def test_011(self):
        self._check(LandmarkHead_(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    @_fails_compile()
    def test_012(self):
        self._check(LossLayer(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_013(self):
        self._check(RegressionTransform(*[], **{}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 10]), torch.rand([4, 4, 4, 4])], {})

