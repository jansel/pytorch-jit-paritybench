import sys
_module = sys.modules[__name__]
del sys
lib = _module
datasets = _module
factory = _module
imdb = _module
pascal_voc = _module
voc_eval = _module
model = _module
bbox_transform = _module
bbox_transform_cpu = _module
config = _module
nms_wrapper = _module
test = _module
train_val = _module
train_val_ori = _module
nets = _module
fpn = _module
layers_util = _module
network = _module
network_fpn = _module
resnet = _module
vgg16 = _module
nms = _module
build = _module
nms_test = _module
py_cpu_nms = _module
pth_nms = _module
roi_data_layer = _module
layer = _module
minibatch = _module
roidb = _module
roi_pooling = _module
_ext = _module
roi_pooling = _module
build = _module
roi_pool = _module
roi_pool_py = _module
rpn = _module
anchor_target_layer_cpu = _module
anchor_target_layer_gpu = _module
anchor_target_layer_cpu = _module
generate_anchors_global = _module
proposal_layer = _module
proposal_target_layer = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer = _module
setup = _module
utils = _module
bbox = _module
blob = _module
timer = _module
test = _module
train = _module

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


import torch


import numpy as np


import math


import time


import torchvision.utils as vutils


import torchvision.transforms as torchtrans


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


from torch.autograd import Function


import numpy.random as npr


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


class Resnet_Ori(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Resnet_Ori, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

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
        x = self.fc(x)
        return x


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RoIPool(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width))
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(roi[1:].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))
                    is_empty = hend <= hstart or wend <= wstart
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = features[batch_ind]
                        outputs[roi_ind, :, ph, pw] = torch.max(torch.max(data[:, hstart:hend, wstart:wend], 1)[0], 2)[0].view(-1)
        return outputs


class FPN_ROI_Pooling(nn.Module):

    def __init__(self, pooled_height, pooled_width, feat_strides):
        super(FPN_ROI_Pooling, self).__init__()
        self.roi_pool_p2 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[0])
        self.roi_pool_p3 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[1])
        self.roi_pool_p4 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[2])
        self.roi_pool_p5 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[3])
        self.roi_pool_p6 = RoIPool(pooled_height, pooled_width, 1.0 / feat_strides[4])

    def forward(self, features, rois):
        feat_list = list()
        if rois[0] is not None:
            feat_p2 = self.roi_pool_p2(features[0], rois[0])
            feat_list.append(feat_p2)
        if rois[1] is not None:
            feat_p3 = self.roi_pool_p3(features[1], rois[1])
            feat_list.append(feat_p3)
        if rois[2] is not None:
            feat_p4 = self.roi_pool_p4(features[2], rois[2])
            feat_list.append(feat_p4)
        if rois[3] is not None:
            feat_p5 = self.roi_pool_p5(features[3], rois[3])
            feat_list.append(feat_p5)
        if rois[4] is not None:
            feat_p6 = self.roi_pool_p6(features[4], rois[4])
            feat_list.append(feat_p6)
        return torch.cat(feat_list, dim=0)


class FC(nn.Module):

    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self._layers = {}
        self._channels = {}
        self._feat_stride = None
        self._anchor_scales = None
        self._anchor_ratios = None
        self._num_anchors = None

    def _image_to_head(self, input):
        raise NotImplementedError

    def _head_to_tail(self, pool5):
        raise NotImplementedError

    def _load_pre_trained_model(self, pre_trained_model):
        raise NotImplementedError

    def _init_network(self):
        raise NotImplementedError


Debug = False


DEBUG = False


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)
    return targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = (targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    return torch.cat([labels.unsqueeze(1), targets], 1)


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()
    else:
        out_fn = lambda x: x
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def anchor_target_layer(rpn_cls_score_list, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A_s = num_anchors
    total_anchors = all_anchors.shape[0]
    _allowed_border = 0
    heights = [rpn_cls_score.shape[2] for rpn_cls_score in rpn_cls_score_list]
    widths = [rpn_cls_score.shape[3] for rpn_cls_score in rpn_cls_score_list]
    inds_inside = np.where((all_anchors[:, 0] >= -_allowed_border) & (all_anchors[:, 1] >= -_allowed_border) & (all_anchors[:, 2] < im_info[1] + _allowed_border) & (all_anchors[:, 3] < im_info[0] + _allowed_border))[0]
    if DEBUG:
        None
        None
    anchors = all_anchors[inds_inside, :]
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=len(fg_inds) - num_fg, replace=False)
        labels[disable_inds] = -1
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=len(bg_inds) - num_bg, replace=False)
        labels[disable_inds] = -1
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)
        positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels == 1)
        negative_weights = (1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    begin_cnt = 0
    end_cnt = 0
    begin_cnt_bbox = 0
    end_cnt_bbox = 0
    labels_list = list()
    bbox_targets_list = list()
    bbox_inside_weights_list = list()
    bbox_outside_weights_list = list()
    for height, width, A in zip(heights, widths, A_s):
        begin_cnt = end_cnt
        end_cnt += 1 * height * width * A
        labels_part = labels[begin_cnt:end_cnt]
        labels_part = labels_part.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels_part = labels_part.reshape((1, 1, A * height, width)).reshape((-1,))
        labels_list.append(labels_part)
    assert total_anchors == end_cnt
    labels = np.concatenate(labels_list, axis=0)
    rpn_labels = labels
    rpn_bbox_targets = bbox_targets
    rpn_bbox_inside_weights = bbox_inside_weights
    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _whctrs(anchor):
    """
  Return width, height, x center, and y center for an anchor (window).
  """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _ratio_enum(anchor, ratios):
    """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
  Enumerate a set of anchors for each scale wrt an anchor.
  """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    return anchors


def generate_anchors_global(feat_strides, heights, widths, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    assert len(feat_strides) == len(heights)
    assert len(heights) == len(widths)
    anchors_list = list()
    for index in range(len(feat_strides)):
        anchors = generate_anchors(base_size=feat_strides[index], ratios=np.array(anchor_ratios), scales=np.array([anchor_scales[index]]))
        anchors_list.append(anchors)
    num_anchors = np.asarray([anchors.shape[0] for anchors in anchors_list])

    def global_anchors(height, width, feat_stride, anchors):
        shift_x = np.arange(0, width) * feat_stride
        shift_y = np.arange(0, height) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        A = anchors.shape[0]
        K = shifts.shape[0]
        anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
        return anchors
    global_anchors_list = list()
    for index in range(len(feat_strides)):
        anchors = global_anchors(height=heights[index], width=widths[index], feat_stride=feat_strides[index], anchors=anchors_list[index])
        global_anchors_list.append(anchors)
    anchors = np.concatenate(global_anchors_list, axis=0)
    return anchors, num_anchors


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, requires_grad=False):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if is_cuda:
        v = v
    return v


def bbox_transform_inv(boxes, deltas):
    if len(boxes) == 0:
        return deltas.detach() * 0
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)
    pred_boxes = torch.cat([_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
  Clip boxes to image boundaries.
  """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()
    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([boxes[:, :, 0].clamp(0, im_shape[1] - 1), boxes[:, :, 1].clamp(0, im_shape[0] - 1), boxes[:, :, 2].clamp(0, im_shape[1] - 1), boxes[:, :, 3].clamp(0, im_shape[0] - 1)], 2).view(boxes.size(0), -1)
    return boxes


def pth_nms(dets, thresh):
    """
  dets has to be a tensor
  """
    if not dets.is_cuda:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.cpu_nms(keep, num_out, dets, order, areas, thresh)
        return keep[:num_out[0]]
    else:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        dets = dets[order].contiguous()
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.gpu_nms(keep, num_out, dets, thresh)
        return order[keep[:num_out[0]]].contiguous()


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
    return pth_nms(dets, thresh)


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """
    rpn_cls_prob = [n, h, w, c=a*2]
    rpn_bbox_pred = [n, h, w, c=4*a]
  """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    bbox_deltas = rpn_bbox_pred.view((-1, 4))
    scores = scores.contiguous().view((-1, 1))
    proposals = bbox_transform_inv(anchors, bbox_deltas)
    proposals = clip_boxes(proposals, im_info[:2])
    _, order = scores.view(-1).sort(descending=True)
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order.data, :]
    scores = scores[order.data, :]
    keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep, :]
    batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
    blob = torch.cat((batch_inds, proposals), 1)
    del batch_inds
    return blob, scores


class RPN(nn.Module):

    def __init__(self, net):
        super(RPN, self).__init__()
        self._network = net
        self.cross_entropy = None
        self.loss_box = None
        self.cache_dict = {}

    def _init_network(self):
        self._network._init_network()
        self.rpn_conv = nn.Conv2d(self._network._channels['head'], 512, (3, 3), padding=1)
        self.rpn_score = nn.Conv2d(512, self._network._num_anchors * 2, (1, 1))
        self.rpn_bbox = nn.Conv2d(512, self._network._num_anchors * 4, (1, 1))

    def forward(self, im_data, im_info, gt_boxes=None):
        c2 = self._network._layers['c2'](im_data)
        c3 = self._network._layers['c3'](c2)
        c4 = self._network._layers['c4'](c3)
        c5 = self._network._layers['c5'](c4)
        p5 = self._network._layers['p5'](c5)
        p6 = self._network._layers['p6'](p5)
        p4_fusion = F.upsample(p5, size=c4.size()[-2:], mode='bilinear') + self._network._layers['p5_p4_lateral'](c4)
        p4 = self._network._layers['p4'](p4_fusion)
        p3_fusion = F.upsample(p4, size=c3.size()[-2:], mode='bilinear') + self._network._layers['p4_p3_lateral'](c3)
        p3 = self._network._layers['p3'](p3_fusion)
        p2_fusion = F.upsample(p3, size=c2.size()[-2:], mode='bilinear') + self._network._layers['p3_p2_lateral'](c2)
        p2 = self._network._layers['p2'](p2_fusion)
        p_list = [p2, p3, p4, p5, p6]
        if Debug:
            c_list = [c2, c3, c4, c5]
            None
            for p in p_list:
                None
            None
            for c in c_list:
                None
            None
        rpn_cls_prob_final_list = list()
        rpn_bbox_score_list = list()
        rpn_cls_score_list = list()
        rpn_cls_score_reshape_list = list()
        for feature in p_list:
            rpn_feature = self.rpn_conv(feature)
            rpn_cls_score = self.rpn_score(rpn_feature)
            rpn_cls_score_list.append(rpn_cls_score)
            rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2)
            rpn_cls_score_reshape_list.append(rpn_cls_score_reshape)
            rpn_cls_prob = F.softmax(rpn_cls_score_reshape, 1)
            rpn_cls_prob_final = self._reshape_layer(rpn_cls_prob, self._network._num_anchors * 2).permute(0, 2, 3, 1).contiguous()
            rpn_cls_prob_final_list.append(rpn_cls_prob_final)
            rpn_bbox_score = self.rpn_bbox(rpn_feature)
            rpn_bbox_score = rpn_bbox_score.permute(0, 2, 3, 1).contiguous()
            rpn_bbox_score_list.append(rpn_bbox_score)
        if Debug:
            None
            for i in rpn_cls_prob_final_list:
                None
        self._generate_anchors(rpn_cls_score_list)
        rois, scores = self._region_proposal(rpn_cls_prob_final_list, rpn_bbox_score_list, im_info)
        if Debug:
            None
        if self.training:
            assert gt_boxes is not None
            rpn_data = self._anchor_target_layer(rpn_cls_score_list, gt_boxes, im_info)
            self.cross_entropy, self.loss_box = self._build_loss(rpn_cls_score_reshape_list, rpn_bbox_score_list, rpn_data)
        self.cache_dict['rpn_cls_prob_final_list'] = rpn_cls_prob_final_list
        self.cache_dict['rpn_bbox_score_list'] = rpn_bbox_score_list
        self.cache_dict['rpn_cls_score_list'] = rpn_cls_score_list
        self.cache_dict['rpn_cls_score_reshape_list'] = rpn_cls_score_reshape_list
        return rois, scores, p_list

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * cfg.TRAIN.LOSS_RATIO

    def _build_loss(self, rpn_cls_score_reshape_list, rpn_bbox_score_list, rpn_data, sigma_rpn=3):
        rpn_cls_score = [rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2) for rpn_cls_score_reshape in rpn_cls_score_reshape_list]
        rpn_cls_score = torch.cat(rpn_cls_score, dim=0)
        rpn_label = rpn_data[0].view(-1)
        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())
        if Debug:
            None
            None
            None
        assert rpn_keep.numel() == cfg.TRAIN.RPN_BATCHSIZE
        if cfg.CUDA_IF:
            rpn_keep = rpn_keep
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
        rpn_bbox_score = [rpn_bbox_score.view((-1, 4)) for rpn_bbox_score in rpn_bbox_score_list]
        rpn_bbox_score = torch.cat(rpn_bbox_score, dim=0)
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_score, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[0, 1])
        return rpn_cross_entropy, rpn_loss_box

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1.0 / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (abs_in_box_diff - 0.5 / sigma_2) * (1.0 - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        loss_box = loss_box.mean()
        return loss_box

    def _reshape_layer(self, x, d):
        """
    :param x: n [a1 b1,a2 b2] h w
    :param d: d
    :return: n 2 a*h w
    """
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] * input_shape[2]) / float(d)), input_shape[3])
        return x

    def _generate_anchors(self, rpn_cls_score_list):
        heights = [rpn_cls_score.size()[-2] for rpn_cls_score in rpn_cls_score_list]
        widths = [rpn_cls_score.size()[-1] for rpn_cls_score in rpn_cls_score_list]
        anchors, num_anchors = generate_anchors_global(feat_strides=self._network._feat_stride, heights=heights, widths=widths, anchor_scales=self._network._anchor_scales, anchor_ratios=self._network._anchor_ratios)
        self._anchors = Variable(torch.from_numpy(anchors)).float()
        self._num_anchors = torch.from_numpy(num_anchors)
        if cfg.CUDA_IF:
            self._anchors = self._anchors
            self._num_anchors = self._num_anchors
        self.cache_dict['anchors_cache'] = self._anchors
        self.cache_dict['num_anchors_cache'] = self._num_anchors

    def _region_proposal(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info):
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois, rpn_scores = proposal_layer(rpn_cls_prob_list=rpn_cls_prob_reshape, rpn_bbox_pred_list=rpn_bbox_pred, im_info=im_info, cfg_key=cfg_key, _feat_stride=self._network._feat_stride, anchors=self._anchors, num_anchors_list=self._num_anchors)
        if cfg_key == 'TEST':
            leveled_rois = self.rois_split_level(rois)
            self.cache_dict['rpn_leveled_rois'] = leveled_rois
            return leveled_rois, rpn_scores
        else:
            self.cache_dict['rois'] = rois
            self.cache_dict['rpn_scores'] = rpn_scores
            return rois, rpn_scores

    @staticmethod
    def rois_split_level(rois):

        def calc_level(width, height):
            value = width * height
            if value == 0:
                inner = 0
            else:
                inner = 4 + np.log2(np.sqrt(value) / 224)
            return min(6, max(2, int(inner)))
        level = lambda roi: calc_level(roi[3] - roi[1], roi[4] - roi[2])
        rois_data = rois.data.cpu().numpy()
        leveled_rois = [None] * 5
        leveled_idxs = [[], [], [], [], []]
        for idx, roi in enumerate(rois_data):
            level_idx = level(roi) - 2
            leveled_idxs[level_idx].append(idx)
        if Debug:
            None
            for i in range(5):
                None
        for level_index in range(0, 5):
            if len(leveled_idxs[level_index]) != 0:
                k = torch.from_numpy(np.asarray(leveled_idxs[level_index]))
                if cfg.CUDA_IF:
                    k = k
                leveled_rois[level_index] = rois[k]
        return leveled_rois

    def _anchor_target_layer(self, rpn_cls_score_list, gt_boxes, im_info):
        rpn_cls_score_l = [rpn_cls_score.data for rpn_cls_score in rpn_cls_score_list]
        gt_boxes = gt_boxes.data.cpu().numpy()
        all_anchors = self._anchors.data.cpu().numpy()
        all_num_anchors = self._num_anchors.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(rpn_cls_score_list=rpn_cls_score_l, gt_boxes=gt_boxes, im_info=im_info, _feat_stride=self._network._feat_stride, all_anchors=all_anchors, num_anchors=all_num_anchors)
        rpn_labels = np_to_variable(rpn_labels, is_cuda=cfg.CUDA_IF, dtype=torch.LongTensor)
        rpn_bbox_targets = np_to_variable(rpn_bbox_targets, is_cuda=cfg.CUDA_IF)
        rpn_bbox_inside_weights = np_to_variable(rpn_bbox_inside_weights, is_cuda=cfg.CUDA_IF)
        rpn_bbox_outside_weights = np_to_variable(rpn_bbox_outside_weights, is_cuda=cfg.CUDA_IF)
        self.cache_dict['rpn_labels'] = rpn_labels
        self.cache_dict['rpn_bbox_targets'] = rpn_bbox_targets
        self.cache_dict['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self.cache_dict['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
  gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
  """
    jittered_boxes = gt_boxes.clone()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_rand = Variable((jittered_boxes.data.new(jittered_boxes.size()[0]).random_(0, 1) - 0.5) * jitter)
    width_offset = width_rand * ws
    height_rand = Variable((jittered_boxes.data.new(jittered_boxes.size()[0]).random_(0, 1) - 0.5) * jitter)
    height_offset = height_rand * hs
    del width_rand
    del height_rand
    jittered_boxes[:, 0] = jittered_boxes[:, 0] + width_offset
    jittered_boxes[:, 2] = jittered_boxes[:, 2] + width_offset
    jittered_boxes[:, 1] = jittered_boxes[:, 1] + height_offset
    jittered_boxes[:, 3] = jittered_boxes[:, 3] + height_offset
    return jittered_boxes


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
    clss = bbox_target_data[:, 0]
    bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()
    bbox_inside_weights = clss.new(bbox_targets.shape).zero_()
    inds = (clss > 0).nonzero().view(-1)
    if inds.numel() > 0:
        clss = clss[inds].contiguous().view(-1, 1)
        dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
        dim2_inds = torch.cat([4 * clss, 4 * clss + 1, 4 * clss + 2, 4 * clss + 3], 1).long()
        bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
        bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)
    return bbox_targets, bbox_inside_weights


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
    overlaps = bbox_overlaps(all_rois[:, 1:5].data, gt_boxes[:, :4].data)
    max_overlaps, gt_assignment = overlaps.max(1)
    labels = gt_boxes[gt_assignment, [4]]
    fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
    bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)
    if fg_inds.numel() > 0 and bg_inds.numel() > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
        fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long()]
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.numel() < bg_rois_per_image
        bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long()]
    elif fg_inds.numel() > 0:
        to_replace = fg_inds.numel() < rois_per_image
        fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long()]
        fg_rois_per_image = rois_per_image
    elif bg_inds.numel() > 0:
        to_replace = bg_inds.numel() < rois_per_image
        bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long()]
        fg_rois_per_image = 0
    else:
        pdb.set_trace()
    keep_inds = torch.cat([fg_inds, bg_inds], 0)
    labels = labels[keep_inds].contiguous()
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds].contiguous()
    roi_scores = all_scores[keep_inds].contiguous()
    bbox_target_data = _compute_targets(rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data)
    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
    """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """
    all_rois = rpn_rois
    all_scores = rpn_scores
    if cfg.TRAIN.USE_GT:
        if cfg.TRAIN.GT_OFFSET:
            gt_easyboxes = gt_boxes
            jittered_gt_boxes = _jitter_gt_boxes(gt_easyboxes)
            zeros = rpn_rois.data.new(gt_boxes.size()[0] * 2, 1).zero_()
            zeros = Variable(zeros)
            jit_cat = torch.cat((gt_easyboxes[:, :-1], jittered_gt_boxes[:, :-1]), 0)
            jit_cat = torch.cat((zeros, jit_cat), 1)
            all_rois = torch.cat((all_rois, jit_cat), 0)
        else:
            zeros = rpn_rois.data.new(gt_boxes.shape[0], 1).zero_()
            zeros = Variable(zeros)
            zeros_cat = torch.cat((zeros, gt_boxes[:, :-1]), dim=1)
            all_rois = torch.cat((all_rois, zeros_cat), dim=0)
        all_scores = torch.cat((all_scores, zeros), 0)
        del zeros
    num_images = 1
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes)
    rois = rois.view(-1, 5)
    roi_scores = roi_scores.view(-1)
    labels = labels.view(-1, 1)
    bbox_targets = bbox_targets.view(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
    bbox_outside_weights = (bbox_inside_weights > 0).float()
    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


class FasterRCNN(nn.Module):

    def __init__(self, net, classes=None):
        super(FasterRCNN, self).__init__()
        assert classes is not None, 'class can not be none!'
        self._classes = np.array(classes)
        self._num_classes = len(classes)
        self._rpn = RPN(net=net)
        self.cross_entropy = None
        self.loss_box = None
        self.cache_dict = {}
        self.metrics_dict = {}

    def init_fasterRCNN(self):
        self._rpn._init_network()
        self.roi_pool = FPN_ROI_Pooling(7, 7, self._rpn._network._feat_stride)
        self.score_fc = nn.Linear(self._rpn._network._channels['tail'], self._num_classes)
        self.bbox_fc = nn.Linear(self._rpn._network._channels['tail'], self._num_classes * 4)

    def _predict(self, im_data, im_info, gt_boxes):
        cudnn.benchmark = False
        rois, rpn_scores, features = self._rpn(im_data, im_info, gt_boxes)
        if self.training:
            roi_data = self._proposal_target_layer(rpn_rois=rois, gt_boxes=gt_boxes, rpn_scores=rpn_scores)
            rois = roi_data[0]
        else:
            roi_data = None
        pooled_features = self.roi_pool(features, rois)
        self.cache_dict['pooled_features'] = pooled_features
        if self.training:
            assert pooled_features.size()[0] == cfg.TRAIN.BATCH_SIZE
            cudnn.benchmark = True
        x = self._rpn._network._head_to_tail(pooled_features)
        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score, 1)
        bbox_pred = self.bbox_fc(x)
        if self.training:
            self.cross_entropy, self.loss_box = self._build_loss(cls_score, bbox_pred, roi_data)
        if self.training:
            return cls_prob, bbox_pred, roi_data[1]
        else:
            leveled_rois_t = [i for i in rois if i is not None]
            new_rois = torch.cat(leveled_rois_t, 0)
            return cls_prob, bbox_pred, new_rois

    def forward(self, im_data, im_info, gt_boxes=None):
        im_data = np_to_variable(im_data, is_cuda=cfg.CUDA_IF).permute(0, 3, 1, 2)
        self.cache_dict['im_data'] = im_data
        gt_boxes = np_to_variable(gt_boxes, is_cuda=cfg.CUDA_IF) if gt_boxes is not None else None
        self.cache_dict['gt_boxes'] = gt_boxes
        cls_prob, bbox_pred, rois = self._predict(im_data, im_info, gt_boxes)
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            bbox_pred = bbox_pred.mul(Variable(stds)).add(Variable(means))
        if not self.training:
            self._delete_cache()
        return cls_prob, bbox_pred, rois

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * cfg.TRAIN.LOSS_RATIO

    def _build_loss(self, cls_score, bbox_pred, roi_data, sigma_rpn=1):
        label = roi_data[2].squeeze()
        assert label.dim() == 1
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt
        self.metrics_dict['fg'] = fg_cnt
        self.metrics_dict['bg'] = bg_cnt
        _, predict = cls_score.data.max(1)
        label_data = label.data
        tp = torch.sum(predict.eq(label_data) & label_data.ne(0)) if fg_cnt > 0 else 0
        tf = torch.sum(predict.eq(label_data) & label_data.eq(0)) if bg_cnt > 0 else 0
        self.metrics_dict['tp'] = tp
        self.metrics_dict['tf'] = tf
        assert cfg.TRAIN.BATCH_SIZE == label.numel()
        cross_entropy = F.cross_entropy(cls_score, label)
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[3:]
        loss_box = self._rpn._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=sigma_rpn)
        return cross_entropy, loss_box

    def init_special_bbox_fc(self, dev=0.001):

        def _gaussian_init(m, dev):
            m.weight.data.normal_(0.0, dev)
            if hasattr(m.bias, 'data'):
                m.bias.data.zero_()
        model = self.bbox_fc
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                _gaussian_init(m, dev)
            elif isinstance(m, nn.Linear):
                _gaussian_init(m, dev)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _delete_cache(self):
        for dic in [self._rpn.cache_dict, self.cache_dict]:
            for key in dic.keys():
                del dic[key]

    def _proposal_target_layer(self, rpn_rois, gt_boxes, rpn_scores):
        rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, self._num_classes)
        leveled_rois, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = self.rois_split_level(rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        labels = labels.long()
        bbox_targets = Variable(bbox_targets)
        bbox_inside_weights = Variable(bbox_inside_weights)
        bbox_outside_weights = Variable(bbox_outside_weights)
        self.cache_dict['labels'] = labels
        self.cache_dict['bbox_targets'] = bbox_targets
        self.cache_dict['bbox_inside_weights'] = bbox_inside_weights
        self.cache_dict['bbox_outside_weights'] = bbox_outside_weights
        return leveled_rois, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @staticmethod
    def rois_split_level(rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights):

        def calc_level(width, height):
            value = width * height
            if value == 0:
                inner = 0
            else:
                inner = 4 + np.log2(np.sqrt(value) / 224)
            return min(6, max(2, int(inner)))
        level = lambda roi: calc_level(roi[3] - roi[1], roi[4] - roi[2])
        rois_data = rois.data.cpu().numpy()
        leveled_rois = [None] * 5
        leveled_labels = [None] * 5
        leveled_bbox_targets = [None] * 5
        leveled_bbox_inside_weights = [None] * 5
        leveled_bbox_outside_weights = [None] * 5
        leveled_idxs = [[], [], [], [], []]
        for idx, roi in enumerate(rois_data):
            level_idx = level(roi) - 2
            leveled_idxs[level_idx].append(idx)
        for level_index in range(0, 5):
            if len(leveled_idxs[level_index]) != 0:
                k = torch.from_numpy(np.asarray(leveled_idxs[level_index]))
                if cfg.CUDA_IF:
                    k = k
                leveled_rois[level_index] = rois[k]
                leveled_labels[level_index] = labels[k]
                leveled_bbox_targets[level_index] = bbox_targets[k]
                leveled_bbox_inside_weights[level_index] = bbox_inside_weights[k]
                leveled_bbox_outside_weights[level_index] = bbox_outside_weights[k]
        leveled_rois_t = [i for i in leveled_rois if i is not None]
        leveled_labels = [i for i in leveled_labels if i is not None]
        leveled_bbox_targets = [i for i in leveled_bbox_targets if i is not None]
        leveled_bbox_inside_weights = [i for i in leveled_bbox_inside_weights if i is not None]
        leveled_bbox_outside_weights = [i for i in leveled_bbox_outside_weights if i is not None]
        new_rois = torch.cat(leveled_rois_t, 0)
        new_labels = torch.cat(leveled_labels, 0)
        new_bbox_targets = torch.cat(leveled_bbox_targets, 0)
        new_bbox_inside_weights = torch.cat(leveled_bbox_inside_weights, 0)
        new_bbox_outside_weights = torch.cat(leveled_bbox_outside_weights, 0)
        return leveled_rois, new_rois, new_labels, new_bbox_targets, new_bbox_inside_weights, new_bbox_outside_weights

    def train_operation(self, blobs, optimizer, image_if=False, clip_parameters=None):
        im_data = blobs['data']
        im_info = blobs['im_info']
        gt_boxes = blobs['gt_boxes']
        result_cls_prob, result_bbox_pred, result_rois = self(im_data, im_info, gt_boxes)
        loss = self.loss + self._rpn.loss
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if clip_parameters is not None:
                nn.utils.clip_grad_norm(self._parameters, max_norm=10)
            optimizer.step()
        loss = loss.data.cpu()[0]
        rpn_cls_loss = self._rpn.cross_entropy.data.cpu()[0]
        rpn_bbox_loss = self._rpn.loss_box.data.cpu()[0]
        fast_rcnn_cls_loss = self.cross_entropy.data.cpu()[0]
        fast_rcnn_bbox_loss = self.loss_box.data.cpu()[0]
        image = None
        if image_if:
            image = self.visual_image(blobs, result_cls_prob, result_bbox_pred, result_rois)
        self._delete_cache()
        return (loss, rpn_cls_loss, rpn_bbox_loss, fast_rcnn_cls_loss, fast_rcnn_bbox_loss), image

    def visual_image(self, blobs, result_cls_prob, result_bbox_pred, result_rois):
        new_gt_boxes = blobs['gt_boxes'].copy()
        new_gt_boxes[:, :4] = new_gt_boxes[:, :4]
        image = self.back_to_image(blobs['data']).astype(np.uint8)
        im_shape = image.shape
        pred_boxes, scores, classes = self.interpret_faster_rcnn_scale(result_cls_prob, result_bbox_pred, result_rois, im_shape, min_score=0.1)
        image = self.draw_photo(image, pred_boxes, scores, classes, new_gt_boxes)
        image = torchtrans.ToTensor()(image)
        image = vutils.make_grid([image])
        return image

    @staticmethod
    def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
        dets = np.hstack((pred_boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thresh)
        if inds is None:
            return pred_boxes[keep], scores[keep]
        return pred_boxes[keep], scores[keep], inds[keep]

    def interpret_faster_rcnn_scale(self, cls_prob, bbox_pred, rois, im_shape, nms=True, clip=True, min_score=0.0):
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([box_deltas[i, inds[i] * 4:inds[i] * 4 + 4] for i in range(len(inds))], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5]
        if len(keep) != 0:
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
        else:
            pred_boxes = boxes
        if clip and pred_boxes.shape[0] > 0:
            pred_boxes = clip_boxes(pred_boxes, im_shape)
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = self.nms_detections(pred_boxes, scores, 0.3, inds=inds)
        return pred_boxes, scores, self._classes[inds]

    def draw_photo(self, image, dets, scores, classes, gt_boxes):
        im2show = image
        for i, det in enumerate(dets):
            det = tuple(int(x) for x in det)
            r = min(0 + i * 10, 255)
            r_i = i / 5
            g = min(150 + r_i * 10, 255)
            g_i = r_i / 5
            b = min(200 + g_i, 255)
            color_b_c = r, g, b
            cv2.rectangle(im2show, det[0:2], det[2:4], color_b_c, 2)
            cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
        for i, det in enumerate(gt_boxes):
            det = tuple(int(x) for x in det)
            gt_class = self._classes[det[-1]]
            cv2.rectangle(im2show, det[0:2], det[2:4], (255, 0, 0), 2)
            cv2.putText(im2show, '%s' % gt_class, (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
        return im2show

    def back_to_image(self, img):
        image = img[0] + cfg.PIXEL_MEANS
        image = image[:, :, ::-1].copy(order='C')
        return image


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def set_BN_eval(model, eval_if=True):
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            if eval_if:
                set_trainable(mod, False)
                mod.eval()


class Resnet(Network):

    def __init__(self, resnet_type, feat_strdie=(16,), anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        Network.__init__(self)
        self._resnet_type = resnet_type
        self._channels['head'] = None
        self._channels['tail'] = None
        self._feat_stride = feat_strdie
        self._anchor_scales = anchor_scales
        self._anchor_ratios = anchor_ratios
        self._num_anchors = len(anchor_scales) * len(anchor_ratios)

    def _init_network(self):
        if self._resnet_type == 18:
            layers = [2, 2, 2, 2]
            self._resnet = Resnet_Ori(BasicBlock, layers)
        elif self._resnet_type == 34:
            layers = [3, 4, 6, 3]
            self._resnet = Resnet_Ori(BasicBlock, layers)
        elif self._resnet_type == 50:
            layers = [3, 4, 6, 3]
            self._resnet = Resnet_Ori(Bottleneck, layers)
        elif self._resnet_type == 101:
            layers = [3, 4, 23, 3]
            self._resnet = Resnet_Ori(Bottleneck, layers)
        else:
            raise NotImplementedError
        set_trainable(self._resnet.conv1, False)
        set_trainable(self._resnet.bn1, False)
        assert 0 <= cfg.RESNET.FIXED_BLOCKS < 4
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            set_trainable(self._resnet.layer3, False)
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            set_trainable(self._resnet.layer2, False)
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            set_trainable(self._resnet.layer1, False)
        self._layers['head'] = nn.Sequential(self._resnet.conv1, self._resnet.bn1, self._resnet.relu, self._resnet.maxpool, self._resnet.layer1, self._resnet.layer2, self._resnet.layer3)
        self._layers['tail'] = nn.Sequential(self._resnet.layer4)
        self._channels['head'] = self._resnet.block.expansion * 256
        self._channels['tail'] = self._resnet.block.expansion * 512

    def _bn_eval(self):
        set_BN_eval(self._resnet.bn1, True)
        assert 0 <= cfg.RESNET.FIXED_BLOCKS < 4
        set_BN_eval(self._resnet.layer4, True)
        set_BN_eval(self._resnet.layer3, True)
        set_BN_eval(self._resnet.layer2, True)
        set_BN_eval(self._resnet.layer1, True)

    def _image_to_head(self, input):
        return self._layers['head'](input)

    def _head_to_tail(self, pool5):
        x = self._layers['tail'](pool5)
        x = x.mean(3).mean(2)
        x = x.view(x.size()[0], -1)
        return x

    def _load_pre_trained_model(self, pre_trained_model):
        pre_model = torch.load(pre_trained_model)
        state_dict = self._resnet.state_dict()
        pre_model_dict = {k: v for k, v in pre_model.items() if k in state_dict}
        None
        state_dict.update(pre_model_dict)
        self._resnet.load_state_dict(pre_model_dict)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout())


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


vgg_cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]}


class VGG16(Network):

    def __init__(self, feat_strdie=(16,), anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        Network.__init__(self)
        self._channels['head'] = 512
        self._channels['tail'] = 4096
        self._feat_stride = feat_strdie
        self._anchor_scales = anchor_scales
        self._anchor_ratios = anchor_ratios
        self._num_anchors = len(anchor_scales) * len(anchor_ratios)

    def _init_network(self, bn=False):
        self._vgg = VGG(make_layers(vgg_cfg['D']))
        for layer in range(10):
            for p in self._vgg.features[layer].parameters():
                p.requires_grad = False
        self._layers['head'] = self._vgg.features
        self._layers['tail'] = self._vgg.classifier

    def _image_to_head(self, input):
        return self._layers['head'](input)

    def _head_to_tail(self, pool5):
        x = pool5.view(pool5.size()[0], -1)
        return self._layers['tail'](x)

    def _load_pre_trained_model(self, pre_trained_model):
        pre_model = torch.load(pre_trained_model)
        state_dict = self._vgg.state_dict()
        pre_model_dict = {k: v for k, v in pre_model.items() if k in state_dict}
        for key in state_dict.keys():
            if 'classifier' in key:
                key_split = key.split('.')
                new_key = [key_split[0]] + [str(int(key_split[1]) + 1)] + [key_split[2]]
                pre_model_dict[key] = pre_model['.'.join(new_key)]
            else:
                pass
        state_dict.update(pre_model_dict)
        self._vgg.load_state_dict(pre_model_dict)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yingxingde_FasterRCNN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

