import sys
_module = sys.modules[__name__]
del sys
anchors = _module
dataloader = _module
detect = _module
down = _module
eval_widerface = _module
img_tester = _module
losses = _module
magic_convert = _module
mnas = _module
mobile = _module
mobile_testing = _module
model = _module
test_argu = _module
torchvision_model = _module
train = _module
utils = _module
video_detect = _module

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


import torch.nn as nn


import torchvision.transforms as transforms


from torch.utils.data.sampler import Sampler


from torch.utils.data import Dataset


import torch.nn.functional as F


import random


import math


from scipy import misc


import torchvision


import torchvision.ops as ops


import time


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import numpy as numpy


from collections import OrderedDict


import torch.utils.model_zoo as model_zoo


import torchvision.models.detection.backbone_utils as backbone_utils


import torchvision.models.resnet as resnet


import torchvision.models._utils as _utils


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data import random_split


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
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors


class Anchors(nn.Module):

    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
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
        image_shapes = [((image_shape + 2 ** x - 1) // 2 ** x) for x in self.pyramid_levels]
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        return torch.from_numpy(all_anchors.astype(np.float32))


class WingLoss(nn.Module):

    def __init__(self, omega=1, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        None
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        None
        sdf
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class AdaptiveWingLoss(nn.Module):

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        """
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-08)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


class LossLayer(nn.Module):

    def __init__(self):
        super(LossLayer, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss()

    def forward(self, classifications, bbox_regressions, ldm_regressions, anchors, annotations):
        batch_size = classifications.shape[0]
        classification_losses = []
        bbox_regression_losses = []
        ldm_regression_losses = []
        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        positive_indices_list = []
        for j in range(batch_size):
            classification = classifications[j, :, :]
            bbox_regression = bbox_regressions[j, :, :]
            ldm_regression = ldm_regressions[j, :, :]
            annotation = annotations[j, :, :]
            annotation = annotation[annotation[:, 0] > 0]
            bbox_annotation = annotation[:, :4]
            ldm_annotation = annotation[:, 4:]
            if bbox_annotation.shape[0] == 0:
                bbox_regression_losses.append(torch.tensor(0.0, requires_grad=True))
                classification_losses.append(torch.tensor(0.0, requires_grad=True))
                ldm_regression_losses.append(torch.tensor(0.0, requires_grad=True))
                positive_indices_list.append([])
                continue
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            targets = torch.ones(classification.shape) * -1
            targets = targets
            negative_indices = torch.lt(IoU_max, 0.3)
            targets[negative_indices, :] = 0
            targets[negative_indices, 1] = 1
            positive_indices = torch.ge(IoU_max, 0.7)
            positive_indices_list.append(positive_indices)
            num_positive_anchors = positive_indices.sum()
            keep_negative_anchors = num_positive_anchors * 3
            bbox_assigned_annotations = bbox_annotation[IoU_argmax, :]
            ldm_assigned_annotations = ldm_annotation[IoU_argmax, :]
            targets[positive_indices, :] = 0
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
                alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
                classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            elif positive_indices.sum() > 0:
                classification_losses.append(positive_losses.mean() + sorted_losses.mean())
            else:
                classification_losses.append(torch.tensor(0.0, requires_grad=True))
            if positive_indices.sum() > 0:
                bbox_assigned_annotations = bbox_assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                gt_widths = bbox_assigned_annotations[:, 2] - bbox_assigned_annotations[:, 0]
                gt_heights = bbox_assigned_annotations[:, 3] - bbox_assigned_annotations[:, 1]
                gt_ctr_x = bbox_assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = bbox_assigned_annotations[:, 1] + 0.5 * gt_heights
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / (anchor_widths_pi + 1e-14)
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / (anchor_heights_pi + 1e-14)
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                bbox_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                bbox_targets = bbox_targets.t()
                bbox_targets = bbox_targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                bbox_regression_loss = self.smoothl1(bbox_targets, bbox_regression[positive_indices, :])
                bbox_regression_losses.append(bbox_regression_loss)
            else:
                bbox_regression_losses.append(torch.tensor(0.0, requires_grad=True))
            if ldm_positive_indices.sum() > 0:
                ldm_assigned_annotations = ldm_assigned_annotations[ldm_positive_indices, :]
                anchor_widths_l = anchor_widths[ldm_positive_indices]
                anchor_heights_l = anchor_heights[ldm_positive_indices]
                anchor_ctr_x_l = anchor_ctr_x[ldm_positive_indices]
                anchor_ctr_y_l = anchor_ctr_y[ldm_positive_indices]
                ldm_targets = []
                for i in range(0, 136):
                    if i % 2 == 0:
                        candidate = (ldm_assigned_annotations[:, i] - anchor_ctr_x_l) / (anchor_widths_l + 1e-14)
                    else:
                        candidate = (ldm_assigned_annotations[:, i] - anchor_ctr_y_l) / (anchor_heights_l + 1e-14)
                    ldm_targets.append(candidate)
                ldm_targets = torch.stack(ldm_targets)
                ldm_targets = ldm_targets.t()
                scale = torch.ones(1, 136) * 0.1
                ldm_targets = ldm_targets / scale
                s1 = torch.ones(1, 99)
                s2 = torch.ones(1, 37) * 3
                s = torch.cat([s1, s2], dim=-1)
                aaaaaaa = WingLoss()
                ldm_regression_loss = self.smoothl1(ldm_targets * s, ldm_regression[ldm_positive_indices, :] * s)
                ldm_regression_losses.append(ldm_regression_loss)
            else:
                ldm_regression_losses.append(torch.tensor(0.0, requires_grad=True))
        return torch.stack(classification_losses), torch.stack(bbox_regression_losses), torch.stack(ldm_regression_losses)


class MBConv3_3x3(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(MBConv3_3x3, self).__init__()
        mid_channels = int(3 * in_channels)
        self.block = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))
        self.use_skip_connect = 1 == stride and in_channels == out_channels

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x)


class MBConv3_5x5(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(MBConv3_5x5, self).__init__()
        mid_channels = int(3 * in_channels)
        self.block = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))
        self.use_skip_connect = 1 == stride and in_channels == out_channels

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x)


class MBConv6_3x3(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(MBConv6_3x3, self).__init__()
        mid_channels = int(6 * in_channels)
        self.block = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6(), nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))
        self.use_skip_connect = 1 == stride and in_channels == out_channels

    def forward(self, x):
        if self.use_skip_connect:
            return self.block(x) + x
        else:
            return self.block(x)


last_fm_list = []


result_list = []


class MBConv6_5x5(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(MBConv6_5x5, self).__init__()
        mid_channels = int(6 * in_channels / 1.125)
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6())
        self.block2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 5, stride, 2, groups=mid_channels, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU6())
        self.block3 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))
        self.use_skip_connect = 1 == stride and in_channels == out_channels

    def forward(self, x):
        if self.use_skip_connect:
            x1 = self.block1(x)
            x1 = self.block2(x1)
            last_fm_list.append(x1)
            x1 = self.block3(x1)
            return x1 + x
        else:
            x1 = self.block1(x)
            result_list.append(x1)
            x1 = self.block2(x1)
            x1 = self.block3(x1)
            return x1


def Conv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU6())


def SepConv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU6(), nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))


class MnasNet(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MnasNet, self).__init__()
        self.out_channels = int(1280 * width_mult)
        self.conv1 = Conv_3x3(3, int(32 * width_mult), 2)
        self.conv2 = SepConv_3x3(int(32 * width_mult), int(16 * width_mult), 1)
        self.feature = nn.Sequential(self._make_layer(MBConv3_3x3, 3, int(16 * width_mult), int(24 * width_mult), 2), self._make_layer(MBConv3_5x5, 3, int(24 * width_mult), int(48 * width_mult), 2), self._make_layer(MBConv6_5x5, 3, int(48 * width_mult), int(80 * width_mult), 2), self._make_layer(MBConv6_3x3, 2, int(80 * width_mult), int(96 * width_mult), 1), self._make_layer(MBConv6_5x5, 4, int(96 * width_mult), int(192 * width_mult), 2))
        self._initialize_weights()

    def _make_layer(self, block, blocks, in_channels, out_channels, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(block(in_channels, out_channels, _stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.feature(x)
        result = OrderedDict()
        result_list.append(last_fm_list[-1])
        result[0] = result_list[0]
        result[1] = result_list[1]
        result[2] = result_list[2]
        return result


class mobileV1(nn.Module):

    def __init__(self):
        super(mobileV1, self).__init__()
        self.mmm = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32 * 4, kernel_size=7, stride=4, padding=2, bias=False), nn.BatchNorm2d(num_features=32 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mmm1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=4, padding=2, bias=False), nn.BatchNorm2d(num_features=3, momentum=0.9), nn.ReLU(inplace=True))
        self.mmm2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=32 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8 * 4, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(num_features=8 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv1 = nn.Sequential(nn.Conv2d(in_channels=8 * 4, out_channels=8 * 4, kernel_size=3, stride=1, padding=1, groups=8 * 4, bias=False), nn.BatchNorm2d(num_features=8 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv2 = nn.Sequential(nn.Conv2d(in_channels=8 * 4, out_channels=16 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=16 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv3 = nn.Sequential(nn.Conv2d(in_channels=16 * 4, out_channels=16 * 4, kernel_size=3, stride=2, padding=1, groups=16 * 4, bias=False), nn.BatchNorm2d(num_features=16 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv4 = nn.Sequential(nn.Conv2d(in_channels=16 * 4, out_channels=32 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=32 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv5 = nn.Sequential(nn.Conv2d(in_channels=32 * 4, out_channels=32 * 4, kernel_size=3, stride=1, padding=1, groups=32 * 4, bias=False), nn.BatchNorm2d(num_features=32 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv6 = nn.Sequential(nn.Conv2d(in_channels=32 * 4, out_channels=32 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=32 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv7 = nn.Sequential(nn.Conv2d(in_channels=32 * 4, out_channels=32 * 4, kernel_size=3, stride=2, padding=1, groups=32 * 4, bias=False), nn.BatchNorm2d(num_features=32 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv8 = nn.Sequential(nn.Conv2d(in_channels=32 * 4, out_channels=64 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=64 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv9 = nn.Sequential(nn.Conv2d(in_channels=64 * 4, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, groups=64 * 4, bias=False), nn.BatchNorm2d(num_features=64 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv10 = nn.Sequential(nn.Conv2d(in_channels=64 * 4, out_channels=64 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=64 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv11 = nn.Sequential(nn.Conv2d(in_channels=64 * 4, out_channels=64 * 4, kernel_size=3, stride=2, padding=1, groups=64 * 4, bias=False), nn.BatchNorm2d(num_features=64 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv12 = nn.Sequential(nn.Conv2d(in_channels=64 * 4, out_channels=128 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv13 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=3, stride=1, padding=1, groups=128 * 4, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv14 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv15 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=3, stride=1, padding=1, groups=128 * 4, bias=False), nn.BatchNorm2d(num_features=128 * 4), nn.ReLU(inplace=True))
        self.mobilenet0_conv16 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv17 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=3, stride=1, padding=1, groups=128 * 4, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv18 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv19 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=3, stride=1, padding=1, groups=128 * 4, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv20 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv21 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=3, stride=1, padding=1, groups=128 * 4, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv22 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv23 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=128 * 4, kernel_size=3, stride=2, padding=1, groups=128 * 4, bias=False), nn.BatchNorm2d(num_features=128 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv24 = nn.Sequential(nn.Conv2d(in_channels=128 * 4, out_channels=256 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=256 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv25 = nn.Sequential(nn.Conv2d(in_channels=256 * 4, out_channels=256 * 4, kernel_size=3, stride=1, padding=1, groups=256 * 4, bias=False), nn.BatchNorm2d(num_features=256 * 4, momentum=0.9), nn.ReLU(inplace=True))
        self.mobilenet0_conv26 = nn.Sequential(nn.Conv2d(in_channels=256 * 4, out_channels=256 * 4, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(num_features=256 * 4, momentum=0.9), nn.ReLU(inplace=True))

    def forward(self, x):
        result = OrderedDict()
        batchsize = x.shape[0]
        x = self.mmm(x)
        x = self.mobilenet0_conv5(x)
        x = self.mobilenet0_conv6(x)
        x = self.mobilenet0_conv7(x)
        x = self.mobilenet0_conv8(x)
        x = self.mobilenet0_conv9(x)
        x10 = self.mobilenet0_conv10(x)
        x = self.mobilenet0_conv11(x10)
        x = self.mobilenet0_conv12(x)
        x = self.mobilenet0_conv13(x)
        x = self.mobilenet0_conv14(x)
        x = self.mobilenet0_conv15(x)
        x = self.mobilenet0_conv16(x)
        x = self.mobilenet0_conv17(x)
        x = self.mobilenet0_conv18(x)
        x = self.mobilenet0_conv19(x)
        x = self.mobilenet0_conv20(x)
        x = self.mobilenet0_conv21(x)
        x22 = self.mobilenet0_conv22(x)
        x = self.mobilenet0_conv23(x22)
        x = self.mobilenet0_conv24(x)
        x = self.mobilenet0_conv25(x)
        x26 = self.mobilenet0_conv26(x)
        result[1] = x10
        result[2] = x22
        result[3] = x26
        return result


class PyramidFeatures(nn.Module):

    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

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

    def __init__(self, inchannels=64, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)
        self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, 2)
        out = self.output_act(out)
        return out.contiguous().view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, inchannels=64, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=64, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 136, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 136)


class ClassHead_(nn.Module):

    def __init__(self, inchannels=256, num_anchors=3):
        super(ClassHead_, self).__init__()
        self.num_anchors = num_anchors
        self.feature_head = self._make_head(self.num_anchors * 2)
        self.output_act = nn.LogSoftmax(dim=-1)

    def _make_head(self, out_size):
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)]
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
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)]
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
            layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(256, out_size, 3, padding=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_head(x)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 10)


class CBR(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(CBR, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CB(nn.Module):

    def __init__(self, inchannels):
        super(CB, self).__init__()
        self.conv3x3 = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(inchannels)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std_box is None:
            self.std_box = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std_box = std_box
        if std_ldm is None:
            self.std_ldm = torch.ones(1, 136) * 0.1

    def forward(self, anchors, bbox_deltas, ldm_deltas, img):
        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights
        ldm_deltas = ldm_deltas * self.std_ldm
        bbox_deltas = bbox_deltas * self.std_box
        bbox_dx = bbox_deltas[:, :, 0]
        bbox_dy = bbox_deltas[:, :, 1]
        bbox_dw = bbox_deltas[:, :, 2]
        bbox_dh = bbox_deltas[:, :, 3]
        pred_ctr_x = ctr_x + bbox_dx * widths
        pred_ctr_y = ctr_y + bbox_dy * heights
        pred_w = torch.exp(bbox_dw) * widths
        pred_h = torch.exp(bbox_dh) * heights
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_landmarks = []
        for i in range(0, 136):
            if i % 2 == 0:
                candidate = ctr_x + ldm_deltas[:, :, i] * widths
            else:
                candidate = ctr_y + ldm_deltas[:, :, i] * heights
            pred_landmarks.append(candidate)
        pred_landmarks = torch.stack(pred_landmarks, dim=2)
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
        B, C, H, W = img.shape
        pred_boxes[:, :, ::2] = torch.clamp(pred_boxes[:, :, ::2], min=0, max=W)
        pred_boxes[:, :, 1::2] = torch.clamp(pred_boxes[:, :, 1::2], min=0, max=H)
        pred_landmarks[:, :, ::2] = torch.clamp(pred_landmarks[:, :, ::2], min=0, max=W)
        pred_landmarks[:, :, 1::2] = torch.clamp(pred_landmarks[:, :, 1::2], min=0, max=H)
        return pred_boxes, pred_landmarks


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, num_anchors=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels, self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels, self.layer4[layers[3] - 1].conv3.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
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
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
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
        bbox_regressions = torch.cat([self.bboxHead(feature) for feature in features], dim=1)
        ldm_regressions = torch.cat([self.ldmHead(feature) for feature in features], dim=1)
        classifications = torch.cat([self.clsHead(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)
        if self.training:
            return self.losslayer(classifications, bbox_regressions, ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)
            return classifications, bboxes, landmarks


class ContextModule(nn.Module):

    def __init__(self, in_channels=256):
        super(ContextModule, self).__init__()
        self.det_conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels))
        self.det_context_conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True))
        self.det_context_conv2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels // 2))
        self.det_context_conv3_1 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True))
        self.det_context_conv3_2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_channels // 2))
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
            lateral_block_module = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            aggr_block_module = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
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
        for feature, lateral_block, context_block, aggr_block in zip(x[:-1][::-1], self.lateral_blocks[:-1][::-1], self.context_blocks[:-1][::-1], self.aggr_blocks[::-1]):
            if not lateral_block:
                continue
            lateral_feature = lateral_block(feature)
            feat_shape = lateral_feature.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
            last_inner = lateral_feature + inner_top_down
            last_inner = aggr_block(last_inner)
            results.insert(0, context_block(last_inner))
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class RetinaFace(nn.Module):

    def __init__(self, backbone, return_layers, anchor_nums=3):
        super(RetinaFace, self).__init__()
        assert backbone, 'Backbone can not be none!'
        assert len(return_layers) > 0, 'There must be at least one return layers'
        self.body = mobileV1()
        in_channels_stage2 = 32
        in_channels_list = [in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
        out_channels = 32
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
        self.ClassHead = self._make_class_head()
        self.BboxHead = self._make_bbox_head()
        self.LandmarkHead = self._make_landmark_head()
        self.anchors = Anchors()
        self.regressBoxes = RegressionTransform()
        self.losslayer = losses.LossLayer()

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=3):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=3):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=3):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def freeze_bn(self):
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
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features.values())], dim=1)
        anchors = self.anchors(img_batch)
        if self.training:
            return self.losslayer(classifications, bbox_regressions, ldm_regressions, anchors, annotations)
        else:
            bboxes, landmarks = self.regressBoxes(anchors, bbox_regressions, ldm_regressions, img_batch)
            return classifications, bboxes, landmarks


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveWingLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Anchors,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BboxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (BboxHead_,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (CB,
     lambda: ([], {'inchannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CBR,
     lambda: ([], {'inchannels': 4, 'outchannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ClassHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ClassHead_,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (Context,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     False),
    (ContextModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (LandmarkHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (LandmarkHead_,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (LossLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MBConv3_3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConv3_5x5,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConv6_3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MBConv6_5x5,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MnasNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (RegressionTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 136]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (mobileV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_ElvishElvis_68_Retinaface_Pytorch_version(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

