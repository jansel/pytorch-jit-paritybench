import sys
_module = sys.modules[__name__]
del sys
LiTS_main = _module
backbone = _module
config = _module
mask_branch = _module
model = _module
preprocessing = _module
utils = _module
backbone = _module
heart_main = _module
mask_branch = _module
model = _module
utils = _module

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


import math


import torch


import time


import re


import numpy as np


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


def conv_S(in_planes, out_planes, stride=1, padding=1):
    """conv_S is the spatial conv layer"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=padding)


def conv_T(in_planes, out_planes, stride=1, padding=1):
    """conv_T is the temporal conv layer"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, padding=padding)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block, expand=False, stride=1, ST_structure=('A', 'B', 'C')):
        """A wrapper for different Bottlenecks.
        block: identify Block_A/B/C.
        expand: whether to expand the final output channel by multiplying expansion.
        """
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.expand = expand
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.ST = list(ST_structure)[(block - 1) % len(ST_structure)]
        self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
        self.bn3 = nn.BatchNorm3d(planes)
        if expand:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1)
            self.bn4 = nn.BatchNorm3d(planes * 4)
            self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes * 4, kernel_size=1, stride=2), nn.BatchNorm3d(planes * 4))
        else:
            self.conv4 = nn.Conv3d(planes, inplanes, kernel_size=1)
            self.bn4 = nn.BatchNorm3d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

    def ST_B(self, x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x + tmp_x

    def ST_C(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)
        return x + tmp_x

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.ST == 'A':
            out = self.ST_A(out)
        elif self.ST == 'B':
            out = self.ST_B(out)
        elif self.ST == 'C':
            out = self.ST_C(out)
        out = self.conv4(out)
        out = self.bn4(out)
        if self.expand:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class P3D(nn.Module):

    def __init__(self, block, layers, input_channel=1, config=None):
        super(P3D, self).__init__()
        self.inplanes = config.BACKBONE_CHANNELS[0]
        self.C1 = nn.Sequential(nn.Conv3d(input_channel, config.BACKBONE_CHANNELS[0], kernel_size=(3, 7, 7), stride=2, padding=(1, 3, 3)), nn.BatchNorm3d(config.BACKBONE_CHANNELS[0]), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=2, stride=2))
        self.C2 = self._make_layer(block, config.BACKBONE_CHANNELS[0], layers[0], stride=2)
        self.C3 = self._make_layer(block, config.BACKBONE_CHANNELS[1], layers[1], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, 1, True, stride))
        self.inplanes = planes * block.expansion
        for i in range(2, blocks + 1):
            layers.append(block(self.inplanes, planes, i, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3]


class Modified3DUNet(nn.Module):

    def __init__(self, in_channels, n_classes, stage, base_n_filter=32):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter
        self.stage = stage
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 8)
        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 4)
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 2)
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter)
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_upscale_conv = self.upscale_conv(self.n_classes, self.n_classes)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out), nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.LeakyReLU(), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(nn.InstanceNorm3d(feat_in), nn.LeakyReLU(), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm3d(feat_out), nn.LeakyReLU())

    def upscale_conv(self, feat_in, feat_out):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv3d(feat_in, feat_out, kernel_size=5, stride=1, padding=2, bias=False))

    def forward(self, x):
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        if self.stage == 'finetune':
            out_upscale = self.out_upscale_conv(out)
            out = self.upsacle(out) + out_upscale
        seg_layer = out
        return seg_layer


class FPN(nn.Module):

    def __init__(self, C1, C2, C3, out_channels, config):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.P3_conv1 = nn.Conv3d(config.BACKBONE_CHANNELS[1] * 4, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P2_conv1 = nn.Conv3d(config.BACKBONE_CHANNELS[0] * 4, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        p3_out = self.P3_conv1(c3_out)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)
        return [p2_out, p3_out]


class RPN(nn.Module):
    """Builds the model of Region Proposal Network.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_logits: [batch, D, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, D, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, D, H, W, (dz, dy, dx, log(dd), log(dh), log(dw))] Deltas to be applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, channel, conv_channel):
        super(RPN, self).__init__()
        self.conv_shared = nn.Conv3d(channel, conv_channel, kernel_size=3, stride=anchor_stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv3d(conv_channel, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv3d(conv_channel, 6 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.conv_shared(x))
        rpn_class_logits = self.conv_class(x)
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 4, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)
        rpn_probs = self.softmax(rpn_class_logits)
        rpn_bbox = self.conv_bbox(x)
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 4, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 6)
        return [rpn_class_logits, rpn_probs, rpn_bbox]


def RoI_Align(feature_map, pool_size, boxes):
    """Implementation of 3D RoI Align (actually it's just pooling rather than align).
    feature_map: [channels, depth, height, width]. Generated from FPN.
    pool_size: [D, H, W]. The shape of the output.
    boxes: [num_boxes, (z1, y1, x1, z2, y2, x2)].
    """
    boxes = utils.denorm_boxes_graph(boxes, (feature_map.size()[1], feature_map.size()[2], feature_map.size()[3]))
    boxes[:, 0] = boxes[:, 0].floor()
    boxes[:, 1] = boxes[:, 1].floor()
    boxes[:, 2] = boxes[:, 2].floor()
    boxes[:, 3] = boxes[:, 3].ceil()
    boxes[:, 4] = boxes[:, 4].ceil()
    boxes[:, 5] = boxes[:, 5].ceil()
    boxes = boxes.long()
    output = torch.zeros((boxes.size()[0], feature_map.size()[0], pool_size[0], pool_size[1], pool_size[2]))
    for i in range(boxes.size()[0]):
        try:
            output[i] = F.interpolate(feature_map[:, boxes[i][0]:boxes[i][3], boxes[i][1]:boxes[i][4], boxes[i][2]:boxes[i][5]].unsqueeze(0), size=pool_size, mode='trilinear', align_corners=True)
        except:
            None
            None
            pass
    return output


def log2(x):
    """Implementation of log2. Pytorch doesn't have a native implementation."""
    ln2 = torch.log(torch.FloatTensor([2.0]))
    if x.is_cuda:
        ln2 = ln2
    return torch.log(x) / ln2


def pyramid_roi_align(inputs, pool_size, test_flag=False):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_size: [depth, height, width] of the output pooled regions. Usually [7, 7, 7]
    - image_shape: [height, width, depth, channels]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] in normalized coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, depth, height, width]
    Output:
    Pooled regions in the shape: [num_boxes, channels, depth, height, width].
    The width, height and depth are those specific in the pool_shape in the layer
    constructor.
    """
    if test_flag:
        for i in range(0, len(inputs)):
            inputs[i] = inputs[i].squeeze(0)
    else:
        for i in range(1, len(inputs)):
            inputs[i] = inputs[i].squeeze(0)
    boxes = inputs[0]
    feature_maps = inputs[1:]
    z1, y1, x1, z2, y2, x2 = boxes.chunk(6, dim=1)
    d = z2 - z1
    h = y2 - y1
    w = x2 - x1
    roi_level = 4 + 1.0 / 3.0 * log2(h * w * d)
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 3)
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 4)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.detach(), :]
        box_to_level.append(ix.detach())
        level_boxes = level_boxes.detach()
        pooled_features = RoI_Align(feature_maps[i], pool_size, level_boxes)
        pooled.append(pooled_features)
    pooled = torch.cat(pooled, dim=0)
    box_to_level = torch.cat(box_to_level, dim=0)
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :, :]
    return pooled


class Classifier(nn.Module):

    def __init__(self, channel, pool_size, image_shape, num_classes, fc_size, test_flag=False):
        super(Classifier, self).__init__()
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.fc_size = fc_size
        self.test_flag = test_flag
        self.conv1 = nn.Conv3d(channel, fc_size, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm3d(fc_size, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv3d(fc_size, fc_size, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm3d(fc_size, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.linear_class = nn.Linear(fc_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear_bbox = nn.Linear(fc_size, num_classes * 6)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.test_flag)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(-1, self.fc_size)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)
        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 6)
        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):

    def __init__(self, channel, pool_size, num_classes, conv_channel, stage, test_flag=False):
        super(Mask, self).__init__()
        self.pool_size = pool_size
        self.test_flag = test_flag
        self.modified_u_net = mask_branch.Modified3DUNet(channel, num_classes, stage, conv_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.test_flag)
        x = self.modified_u_net(x)
        output = self.softmax(x)
        return x, output


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{'source': '', 'id': 0, 'name': 'BG'}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert '.' not in source, 'Source name cannot contain a dot'
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        self.class_info.append({'source': source, 'id': class_id, 'name': class_name})

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {'id': image_id, 'source': source, 'path': path}
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ''

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ','.join(name.split(',')[:1])
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c['name']) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        self.class_from_source_map = {'{}.{}'.format(info['source'], info['id']): id for info, id in zip(self.class_info, self.class_ids)}
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c['map']:
                self.external_to_class_id[ds + str(id)] = i
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info['ds'] + str(info['id'])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]['path']

    def load_image(self, image_id):
        """Load the specified image and return a [H, W, D, 1] Numpy array."""
        image = nib.load(self.image_info[image_id]['path']).get_data().copy()
        return np.expand_dims(image, -1)

    def load_mask(self, image_id):
        """Load the specified mask and return a [H, W, D] Numpy array."""
        mask = np.empty([0, 0, 0])
        return mask


def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.
    image_id: An int ID of the image. Useful for debugging.
    image_shape: [channels, depth, height, width]
    window: (z1, y1, x1, z2, y2, x2) in pixels. The volume of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array([image_id] + list(image_shape) + list(window) + list(active_class_ids))
    return meta


def compute_backbone_shapes(config, image_shape):
    """Computes the depth, width and height of each stage of the backbone network.
    Returns:
        [N, (depth, height, width)]. Where N is the number of stages
    """
    H, W, D = image_shape[:3]
    return np.array([[int(math.ceil(D / stride)), int(math.ceil(H / stride)), int(math.ceil(W / stride))] for stride in config.BACKBONE_STRIDES])


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dz, dy, dx, log(dd), log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    if target_class_ids.size()[0] != 0:
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.detach()].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)
        target_bbox = target_bbox[indices[:, 0].detach(), :]
        pred_bbox = pred_bbox[indices[:, 0].detach(), indices[:, 1].detach(), :]
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    if target_class_ids.size()[0] != 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_mrcnn_mask_edge_loss(target_masks, target_class_ids, pred_masks):
    """Mask edge mean square error loss for the Edge Agreement Head.
    Here I use the Sobel kernel without smoothing the ground_truth masks.
        target_masks: [batch, num_rois, depth, height, width].
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, num_classes, depth, height, width] float32 tensor with values from 0 to 1.
    """
    if target_class_ids.size()[0] != 0:
        kernel_x = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], [[2, 4, 2], [0, 0, 0], [-2, -4, -2]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
        kernel_y = kernel_x.transpose((1, 0, 2))
        kernel_z = kernel_x.transpose((0, 2, 1))
        kernel = torch.from_numpy(np.array([kernel_x, kernel_y, kernel_z]).reshape((3, 1, 3, 3, 3))).float()
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.detach()].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        y_true = target_masks[:indices.size()[0], 1:, :, :]
        y_pred = pred_masks[indices[:, 0].detach(), 1:, :, :, :]
        loss_fn = nn.MSELoss()
        loss = torch.FloatTensor([0])
        for i in range(indices.size()[0]):
            y_true_ = y_true[i]
            y_pred_ = y_pred[i].unsqueeze(0)
            for j in range(7):
                y_true_final = F.conv3d(y_true_[j, :, :, :].unsqueeze(0).unsqueeze(0).float(), kernel)
                y_pred_final = F.conv3d(y_pred_[:, j, :, :, :].unsqueeze(1), kernel)
                y_true_final = torch.sqrt(torch.pow(y_true_final[:, 0], 2) + torch.pow(y_true_final[:, 1], 2) + torch.pow(y_true_final[:, 0], 2))
                y_pred_final = torch.sqrt(torch.pow(y_pred_final[:, 0], 2) + torch.pow(y_pred_final[:, 1], 2) + torch.pow(y_pred_final[:, 0], 2))
                loss += loss_fn(y_pred_final, y_true_final)
        loss /= indices.size()[0]
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, depth, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, num_classes, depth, height, width] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size()[0] != 0:
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.detach()].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        y_true_ = target_masks[indices[:, 0], :, :, :]
        y_true = y_true_.long()
        y_true = torch.argmax(y_true, dim=1)
        y_pred = pred_masks[indices[:, 0].detach(), :, :, :, :]
        los = nn.CrossEntropyLoss()
        loss = los(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss
    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    target_bbox: [batch, max positive anchors, (dz, dy, dx, log(dd), log(dh), log(dw))].
        Uses 0 padding to fill in unused bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    rpn_match = rpn_match.squeeze(2)
    indices = torch.nonzero(rpn_match == 1)
    rpn_bbox = rpn_bbox[indices.detach()[:, 0], indices.detach()[:, 1]]
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)
    return loss


def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    rpn_match = rpn_match.squeeze(2)
    anchor_class = (rpn_match == 1).long()
    indices = torch.nonzero(rpn_match != 0)
    rpn_class_logits = rpn_class_logits[indices.detach()[:, 0], indices.detach()[:, 1], :]
    anchor_class = anchor_class[indices.detach()[:, 0], indices.detach()[:, 1]]
    loss = F.cross_entropy(rpn_class_logits, anchor_class)
    return loss


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits, stage):
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(torch.from_numpy(np.where(target_class_ids > 0, 1, 0)), mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, torch.from_numpy(np.where(target_class_ids > 0, 1, 0)), mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask_logits)
    if stage == 'finetune':
        mrcnn_mask_edge_loss = compute_mrcnn_mask_edge_loss(target_mask, target_class_ids, mrcnn_mask)
    else:
        mrcnn_mask_edge_loss = Variable(torch.FloatTensor([0]), requires_grad=False)
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_mask_edge_loss]


def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:5]
    window = meta[:, 5:11]
    active_class_ids = meta[:, 11:]
    return image_id, image_shape, window, active_class_ids


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is z1, y1, x1, z2, y2, x2
    deltas: [N, 6] where each row is [dz, dy, dx, log(dd), log(dh), log(dw)]
    """
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= torch.exp(deltas[:, 3])
    height *= torch.exp(deltas[:, 4])
    width *= torch.exp(deltas[:, 5])
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([z1, y1, x1, z2, y2, x2], dim=1)
    return result


def clip_to_window(window, boxes):
    """window: (z1, y1, x1, z2, y2, x2). The window in the image we want to clip to.
        boxes: [N, (z1, y1, x1, z2, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[3]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[4]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[2]), float(window[5]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[0]), float(window[3]))
    boxes[:, 4] = boxes[:, 4].clamp(float(window[1]), float(window[4]))
    boxes[:, 5] = boxes[:, 5].clamp(float(window[2]), float(window[5]))
    return boxes


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).detach()]


def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.detach()]


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.
    Inputs:
        rois: [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dz, dy, dx, log(dd), log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (z1, y1, x1, z2, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.
    Returns detections shaped: [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
    """
    _, class_ids = torch.max(probs, dim=1)
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx
    class_scores = probs[idx, class_ids.detach()]
    deltas_specific = deltas[idx, class_ids.detach()]
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    height, width, depth = config.IMAGE_SHAPE[:3]
    scale = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        scale = scale
    refined_rois *= scale
    refined_rois = clip_to_window(window, refined_rois)
    refined_rois = torch.round(refined_rois)
    keep_bool = class_ids > 0
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, 0]
    pre_nms_class_ids = class_ids[keep.detach()]
    pre_nms_scores = class_scores[keep.detach()]
    pre_nms_rois = refined_rois[keep.detach()]
    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]
        ix_rois = pre_nms_rois[ixs.detach()]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.detach(), :]
        class_keep = utils.non_max_suppression(ix_rois.cpu().detach().numpy(), ix_scores.cpu().detach().numpy(), config.DETECTION_NMS_THRESHOLD, config.DETECTION_MAX_INSTANCES)
        class_keep = torch.from_numpy(class_keep).long()
        class_keep = keep[ixs[order[class_keep].detach()].detach()]
        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)
    roi_count = config.DETECTION_MAX_INSTANCES
    roi_count = min(roi_count, keep.size()[0])
    top_ids = class_scores[keep.detach()].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.detach()]
    result = torch.cat((refined_rois[keep.detach()], class_ids[keep.detach()].unsqueeze(1).float(), class_scores[keep.detach()].unsqueeze(1)), dim=1)
    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    Returns:
    [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_score)] in pixels
    """
    rois = rois.squeeze(0)
    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)
    return detections


def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    """
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 6)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = boxes1.chunk(6, dim=1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = boxes2.chunk(6, dim=1)
    z1 = torch.max(b1_z1, b2_z1)[:, 0]
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    z2 = torch.min(b1_z2, b2_z2)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(z1.size()[0]), requires_grad=False)
    if z1.is_cuda:
        zeros = zeros
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros) * torch.max(z2 - z1, zeros)
    b1_volume = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_volume = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_volume[:, 0] + b2_volume[:, 0] - intersection
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.
    Inputs:
    proposals: [batch, N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, MAX_GT_INSTANCES, depth, height, width] of np.int32 type
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dz, dy, dx, log(dd), log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, depth, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    overlaps = bbox_overlaps(proposals, gt_boxes)
    roi_iou_max = torch.max(overlaps, dim=1)[0]
    None
    positive_roi_bool = roi_iou_max >= config.DETECTION_TARGET_IOU_THRESHOLD
    if torch.nonzero(positive_roi_bool).size()[0] != 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.detach(), :]
        positive_overlaps = overlaps[positive_indices.detach(), :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.detach(), :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.detach()]
        deltas = Variable(utils.box_refinement(positive_rois.detach(), roi_gt_boxes.detach()), requires_grad=False)
        std_dev = torch.from_numpy(config.BBOX_STD_DEV).float()
        if config.GPU_COUNT:
            std_dev = std_dev
        deltas /= std_dev
        roi_gt_masks = np.zeros((positive_rois.shape[0], 8) + config.MASK_SHAPE)
        for i in range(0, positive_rois.shape[0]):
            z1 = int(gt_masks.shape[1] * positive_rois[i, 0])
            z2 = int(gt_masks.shape[1] * positive_rois[i, 3])
            y1 = int(gt_masks.shape[2] * positive_rois[i, 1])
            y2 = int(gt_masks.shape[2] * positive_rois[i, 4])
            x1 = int(gt_masks.shape[3] * positive_rois[i, 2])
            x2 = int(gt_masks.shape[3] * positive_rois[i, 5])
            crop_mask = gt_masks[:, z1:z2, y1:y2, x1:x2].cpu().numpy()
            crop_mask = utils.resize(crop_mask, (8,) + config.MASK_SHAPE, order=0, preserve_range=True)
            roi_gt_masks[i, :, :, :, :] = crop_mask
        roi_gt_masks = torch.from_numpy(roi_gt_masks)
        roi_gt_masks = roi_gt_masks.type(torch.DoubleTensor)
    else:
        positive_count = 0
    negative_roi_bool = roi_iou_max < config.DETECTION_TARGET_IOU_THRESHOLD
    negative_roi_bool = negative_roi_bool
    if torch.nonzero(negative_roi_bool).size()[0] != 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.detach(), :]
    else:
        negative_count = 0
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).long()
        if config.GPU_COUNT:
            zeros = zeros
        roi_gt_class_ids = torch.cat([roi_gt_class_ids.long(), zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 6), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.MASK_SHAPE[2]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        masks = roi_gt_masks
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        positive_rois = Variable(torch.FloatTensor(), requires_grad=False)
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
            positive_rois = positive_rois
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 6), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.MASK_SHAPE[2]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros
        masks = zeros
    else:
        positive_rois = Variable(torch.FloatTensor(), requires_grad=False)
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            positive_rois = positive_rois
            rois = rois
            roi_gt_class_ids = roi_gt_class_ids
            deltas = deltas
            masks = masks
    return positive_rois, rois, roi_gt_class_ids, deltas, masks


def build_rpn_targets(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (z1, y1, x1, z2, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (z1, y1, x1, z2, y2, x2)]
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dz, dy, dx, log(dd), log(dh), log(dw))] Anchor bbox deltas.
    """
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6))
    overlaps = utils.compute_overlaps(anchors, gt_boxes)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    rpn_match[anchor_iou_max >= 0.7] = 1
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        gt_d = gt[3] - gt[0]
        gt_h = gt[4] - gt[1]
        gt_w = gt[5] - gt[2]
        gt_center_z = gt[0] + 0.5 * gt_d
        gt_center_y = gt[1] + 0.5 * gt_h
        gt_center_x = gt[2] + 0.5 * gt_w
        a_d = a[3] - a[0]
        a_h = a[4] - a[1]
        a_w = a[5] - a[2]
        a_center_z = a[0] + 0.5 * a_d
        a_center_y = a[1] + 0.5 * a_h
        a_center_x = a[2] + 0.5 * a_w
        rpn_bbox[ix] = [(gt_center_z - a_center_z) / a_d, (gt_center_y - a_center_y) / a_h, (gt_center_x - a_center_x) / a_w, np.log(gt_d / a_d), np.log(gt_h / a_h), np.log(gt_w / a_w)]
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox


def mold_image(images):
    """Normalize the input image to set its mean = 0 and std = 1."""
    return (images - images.mean()) / images.std()


def load_image_gt(image, mask, angle, dataset, config, anchors):
    """Load and return ground truth data for an image.
    angle: rotate the image and mask for augmentation
    anchors: used for generate rpn_match and rpn_bbox
    Returns:
    image: [1, depth, height, width]
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (z1, y1, x1, z2, y2, x2)]
    mask: [depth, height, width, instance_count]
    rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    rpn_bbox: [batch, N, (dz, dy, dx, log(dd), log(dh), log(dw))] Anchor bbox deltas
    """
    augment = iaa.Affine(rotate=angle, order=0)
    if augment is not None:
        MASK_AUGMENTERS = ['Sequential', 'SomeOf', 'OneOf', 'Sometimes', 'Fliplr', 'Flipud', 'CropAndPad', 'Affine', 'PiecewiseAffine']

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS
        image_shape = image.shape
        mask_shape = mask.shape
        image = np.squeeze(image, 3)
        det = augment.to_deterministic()
        image = det.augment_image(image)
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        image = np.expand_dims(image, -1)
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        mask = mask.astype(np.int32)
    image = image.transpose((3, 2, 0, 1))
    mask = mask.transpose((2, 0, 1))
    bbox = utils.extract_bboxes(np.expand_dims(mask, -1))
    z1, y1, x1, z2, y2, x2 = bbox[0, :]
    depth = z2 - z1
    height = y2 - y1
    width = x2 - x1
    z1 -= depth * 0.05
    z2 += depth * 0.05
    y1 -= height * 0.05
    y2 += height * 0.05
    x1 -= width * 0.05
    x2 += width * 0.05
    z1 = np.floor(max(0, z1))
    z2 = np.ceil(min(mask.shape[0], z2))
    y1 = np.floor(max(0, y1))
    y2 = np.ceil(min(mask.shape[1], y2))
    x1 = np.floor(max(0, x1))
    x2 = np.ceil(min(mask.shape[2], x2))
    bbox[0, :] = z1, y1, x1, z2, y2, x2
    bbox = np.tile(bbox.astype(np.int32), (config.NUM_CLASSES - 1, 1))
    masks, class_ids = dataset.process_mask(mask)
    rpn_match, rpn_bbox = build_rpn_targets(anchors, np.array([bbox[0]]), config)
    rpn_match = rpn_match[:, np.newaxis]
    image = mold_image(image.astype(np.float32))
    return image, rpn_match, rpn_bbox, class_ids, bbox, masks


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += 'shape: {:20}  min: {:10.5f}  max: {:10.5f}'.format(str(array.shape), array.min() if array.size else '', array.max() if array.size else '')
    None


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=''):
    """Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    None
    if iteration == total:
        None


def clip_boxes(boxes, window):
    """boxes: [N, 6] each col is z1, y1, x1, z2, y2, x2
    window: [6] in the form z1, y1, x1, z2, y2, x2
    """
    boxes = torch.stack([boxes[:, 0].clamp(float(window[0]), float(window[3])), boxes[:, 1].clamp(float(window[1]), float(window[4])), boxes[:, 2].clamp(float(window[2]), float(window[5])), boxes[:, 3].clamp(float(window[0]), float(window[3])), boxes[:, 4].clamp(float(window[1]), float(window[4])), boxes[:, 5].clamp(float(window[2]), float(window[5]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    Returns:
        Proposals in normalized coordinates [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)
    scores = inputs[0][:, 1]
    deltas = inputs[1]
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev
    deltas = deltas * std_dev
    pre_nms_limit = min(config.PRE_NMS_LIMIT, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.detach(), :]
    anchors = anchors[order.detach(), :]
    boxes = apply_box_deltas(anchors, deltas)
    height, width, depth = config.IMAGE_SHAPE[:3]
    window = np.array([0, 0, 0, depth, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)
    keep = utils.non_max_suppression(boxes.cpu().detach().numpy(), scores.cpu().detach().numpy(), nms_threshold, proposal_count)
    keep = torch.from_numpy(keep).long()
    boxes = boxes[keep, :]
    norm = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        norm = norm
    normalized_boxes = boxes / norm
    normalized_boxes = normalized_boxes.unsqueeze(0)
    return normalized_boxes


class MaskRCNN(nn.Module):
    """Encapsulates the 3D-Mask-RCNN model functionality."""

    def __init__(self, config, model_dir, test_flag=False):
        """config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.epoch = 0
        self.config = config
        self.model_dir = model_dir
        self.build(config=config, test_flag=test_flag)
        self.initialize_weights()

    def build(self, config, test_flag=False):
        """Build 3D-Mask-RCNN architecture."""
        h, w, d = config.IMAGE_SHAPE[:3]
        if h / 16 != int(h / 16) or w / 16 != int(w / 16) or d / 16 != int(d / 16):
            raise Exception('Image size must be dividable by 16. Use 256, 320, 512, ... etc.')
        P3D_Resnet = backbone.P3D19(config=config)
        C1, C2, C3 = P3D_Resnet.stages()
        self.fpn = FPN(C1, C2, C3, out_channels=config.TOP_DOWN_PYRAMID_SIZE, config=config)
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, compute_backbone_shapes(config, config.IMAGE_SHAPE), config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, config.TOP_DOWN_PYRAMID_SIZE, config.RPN_CONV_CHANNELS)
        self.classifier = Classifier(config.TOP_DOWN_PYRAMID_SIZE, config.POOL_SIZE, config.IMAGE_SHAPE, 2, config.FPN_CLASSIFY_FC_LAYERS_SIZE, test_flag)
        self.mask = Mask(1, config.MASK_POOL_SIZE, config.NUM_CLASSES, config.UNET_MASK_BRANCH_CHANNEL, self.config.STAGE, test_flag)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        if not config.TRAIN_BN:
            self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex):
        """Sets model layers as trainable if their names match the given regular expression."""
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def load_weights(self, file_path):
        """Modified version of the corresponding Keras function with the addition of multi-GPU support
        and the ability to exclude some layers from loading.
        exclude: list of layer names to exclude
        """
        if os.path.exists(file_path):
            pretrained_dict = torch.load(file_path)
            self.load_state_dict(pretrained_dict, strict=True)
            None
        else:
            None

    def detect(self, images):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes. [1, height, width, depth, channels]
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, z1, y2, x2, z2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, D, N] instance binary masks
        Transform all outputs from pytorch shape to normal shape here.
        """
        start_time = time.time()
        molded_images, image_metas, windows = self.mold_inputs(images)
        molded_images = torch.from_numpy(molded_images).float()
        if self.config.GPU_COUNT:
            molded_images = molded_images
        with torch.no_grad():
            molded_images = Variable(molded_images)
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')
        detections = detections.detach().cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 5, 2).detach().cpu().numpy()
        None
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_mask = self.unmold_detections(detections[i], mrcnn_mask[i], [image.shape[3], image.shape[2], image.shape[0], image.shape[1]], windows[i])
            results.append({'rois': final_rois, 'class_ids': final_class_ids, 'scores': final_scores, 'mask': final_mask})
        return results

    def predict(self, inputs, mode):
        molded_images = inputs[0]
        image_metas = inputs[1]
        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.apply(set_bn_eval)
        p2_out, p3_out = self.fpn(molded_images)
        rpn_feature_maps = [p2_out, p3_out]
        mrcnn_classifier_feature_maps = [p2_out, p3_out]
        mrcnn_mask_feature_maps = [molded_images, molded_images]
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == 'training' else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox], proposal_count=proposal_count, nms_threshold=self.config.RPN_NMS_THRESHOLD, anchors=self.anchors, config=self.config)
        if mode == 'inference':
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rpn_rois)
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)
            h, w, d = self.config.IMAGE_SHAPE[:3]
            scale = torch.from_numpy(np.array([d, h, w, d, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale
            detection_boxes = detections[:, :6] / scale
            detection_boxes = detection_boxes.unsqueeze(0)
            _, mrcnn_mask = self.mask(mrcnn_mask_feature_maps, detection_boxes)
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask]
        elif mode == 'training':
            gt_class_ids = inputs[2]
            gt_boxes = inputs[3]
            gt_masks = inputs[4]
            h, w, d = self.config.IMAGE_SHAPE[:3]
            scale = torch.from_numpy(np.array([d, h, w, d, h, w])).float()
            if self.config.GPU_COUNT:
                scale = scale
            gt_boxes = gt_boxes / scale
            p_rois, rois, target_class_ids, target_deltas, target_mask = detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)
            if rois.size()[0] == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_mask_logits = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits
                    mrcnn_class = mrcnn_class
                    mrcnn_bbox = mrcnn_bbox
                    mrcnn_mask = mrcnn_mask
                    mrcnn_mask_logits = mrcnn_mask_logits
            elif p_rois.size()[0] == 0:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rois)
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_mask_logits = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_mask = mrcnn_mask
                    mrcnn_mask_logits = mrcnn_mask_logits
            else:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_classifier_feature_maps, rois)
                mrcnn_mask_logits, mrcnn_mask = self.mask(mrcnn_mask_feature_maps, p_rois)
            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done already, so this actually determines
                the epochs to train in total rather than in this particular call.
        """
        layers = '.*'
        train_set = Dataset(train_dataset, self.config)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=23)
        val_set = Dataset(val_dataset, self.config)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=10)
        self.set_trainable(layers)
        trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' not in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([{'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY}, {'params': trainables_only_bn}], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
        total_start_time = time.time()
        start_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        if not os.path.exists('./logs/heart/' + str(start_datetime)):
            os.makedirs('./logs/heart/' + str(start_datetime))
        for epoch in range(self.epoch + 1, epochs + 1):
            log('Epoch {}/{}.'.format(epoch, epochs))
            start_time = time.time()
            angle = np.random.randint(-20, 21)
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_mrcnn_mask_edge = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH, angle, train_dataset)
            None
            if epoch % 5 == 0:
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_mrcnn_mask_edge = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS, angle, val_dataset)
                torch.save(self.state_dict(), './logs/heart/' + str(start_datetime) + '/model' + str(epoch) + '_loss: ' + str(round(loss, 4)) + '_val: ' + str(round(val_loss, 4)))
        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps, angle, dataset):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_mrcnn_mask_edge_sum = 0
        step = 0
        optimizer.zero_grad()
        for inputs in datagenerator:
            batch_count += 1
            image = inputs[0]
            image_metas = inputs[1]
            mask = inputs[2]
            image = image.squeeze(0).cpu().numpy()
            mask = mask.squeeze(0).cpu().numpy()
            images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = load_image_gt(image, mask, angle, dataset, self.config, self.anchors.cpu().numpy())
            image_metas = image_metas.numpy()
            images = Variable(torch.from_numpy(images).float().unsqueeze(0))
            rpn_match = Variable(torch.from_numpy(rpn_match).unsqueeze(0))
            rpn_bbox = Variable(torch.from_numpy(rpn_bbox).float().unsqueeze(0))
            gt_class_ids = Variable(torch.from_numpy(gt_class_ids).unsqueeze(0))
            gt_boxes = Variable(torch.from_numpy(gt_boxes).float().unsqueeze(0))
            gt_masks = Variable(torch.from_numpy(gt_masks).float().unsqueeze(0))
            if self.config.GPU_COUNT:
                images = images
                rpn_match = rpn_match
                rpn_bbox = rpn_bbox
                gt_class_ids = gt_class_ids
                gt_boxes = gt_boxes
                gt_masks = gt_masks
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits = self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_mask_edge_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits, self.config.STAGE)
            loss = self.config.LOSS_WEIGHTS['rpn_class_loss'] * rpn_class_loss + self.config.LOSS_WEIGHTS['rpn_bbox_loss'] * rpn_bbox_loss + self.config.LOSS_WEIGHTS['mrcnn_class_loss'] * mrcnn_class_loss + self.config.LOSS_WEIGHTS['mrcnn_bbox_loss'] * mrcnn_bbox_loss + self.config.LOSS_WEIGHTS['mrcnn_mask_loss'] * mrcnn_mask_loss + self.config.LOSS_WEIGHTS['mrcnn_mask_edge_loss'] * mrcnn_mask_edge_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if batch_count % self.config.BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0
            print_progress_bar(step + 1, steps, prefix='\t{}/{}'.format(step + 1, steps), suffix='Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - mrcnn_mask_edge_loss: {:.5f}'.format(loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['rpn_class_loss'] * rpn_class_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['rpn_bbox_loss'] * rpn_bbox_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_class_loss'] * mrcnn_class_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_bbox_loss'] * mrcnn_bbox_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_mask_loss'] * mrcnn_mask_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_mask_edge_loss'] * mrcnn_mask_edge_loss.detach().cpu().item()), length=45)
            loss_sum += loss.detach().cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.detach().cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.detach().cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.detach().cpu().item() / steps
            loss_mrcnn_mask_edge_sum += mrcnn_mask_edge_loss.detach().cpu().item() / steps
            if step == steps - 1:
                break
            step += 1
        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_mrcnn_mask_edge_sum

    def valid_epoch(self, datagenerator, steps, angle, dataset):
        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_mrcnn_mask_edge_sum = 0
        for inputs in datagenerator:
            image = inputs[0]
            image_metas = inputs[1]
            mask = inputs[2]
            image = image.squeeze(0).cpu().numpy()
            mask = mask.squeeze(0).cpu().numpy()
            images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = load_image_gt(image, mask, angle, dataset, self.config, self.anchors.cpu().numpy())
            image_metas = image_metas.numpy()
            images = Variable(torch.from_numpy(images).float().unsqueeze(0))
            rpn_match = Variable(torch.from_numpy(rpn_match).unsqueeze(0))
            rpn_bbox = Variable(torch.from_numpy(rpn_bbox).float().unsqueeze(0))
            gt_class_ids = Variable(torch.from_numpy(gt_class_ids).unsqueeze(0))
            gt_boxes = Variable(torch.from_numpy(gt_boxes).float().unsqueeze(0))
            gt_masks = Variable(torch.from_numpy(gt_masks).float().unsqueeze(0))
            if self.config.GPU_COUNT:
                images = images
                rpn_match = rpn_match
                rpn_bbox = rpn_bbox
                gt_class_ids = gt_class_ids
                gt_boxes = gt_boxes
                gt_masks = gt_masks
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits = self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')
            if target_class_ids.size()[0] == 0:
                continue
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_mask_edge_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, mrcnn_mask_logits, self.config.STAGE)
            loss = self.config.LOSS_WEIGHTS['rpn_class_loss'] * rpn_class_loss + self.config.LOSS_WEIGHTS['rpn_bbox_loss'] * rpn_bbox_loss + self.config.LOSS_WEIGHTS['mrcnn_class_loss'] * mrcnn_class_loss + self.config.LOSS_WEIGHTS['mrcnn_bbox_loss'] * mrcnn_bbox_loss + self.config.LOSS_WEIGHTS['mrcnn_mask_loss'] * mrcnn_mask_loss + self.config.LOSS_WEIGHTS['mrcnn_mask_edge_loss'] * mrcnn_mask_edge_loss
            print_progress_bar(step + 1, steps, prefix='\t{}/{}'.format(step + 1, steps), suffix='Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - mrcnn_mask_edge_loss: {:.5f}'.format(loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['rpn_class_loss'] * rpn_class_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['rpn_bbox_loss'] * rpn_bbox_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_class_loss'] * mrcnn_class_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_bbox_loss'] * mrcnn_bbox_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_mask_loss'] * mrcnn_mask_loss.detach().cpu().item(), self.config.LOSS_WEIGHTS['mrcnn_mask_edge_loss'] * mrcnn_mask_edge_loss.detach().cpu().item()), length=10)
            loss_sum += loss.detach().cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.detach().cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.detach().cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.detach().cpu().item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.detach().cpu().item() / steps
            loss_mrcnn_mask_edge_sum += mrcnn_mask_edge_loss.detach().cpu().item() / steps
            if step == steps - 1:
                break
            step += 1
        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_mrcnn_mask_edge_sum

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height, width, depth, channels]. Images can have
            different sizes.
        Returns 3 Numpy matrices:
        molded_images: [N, 1, d, h, w]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (z1, y1, x1, z2, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(image, min_dim=self.config.IMAGE_MIN_DIM, max_dim=self.config.IMAGE_MAX_DIM, min_scale=self.config.IMAGE_MIN_SCALE, mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image)
            molded_image = molded_image.transpose((3, 2, 0, 1))
            image_meta = compose_image_meta(0, image.shape, window, np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformat the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the application.
        detections: [N, (z1, y1, x1, z2, y2, x2, class_id, score)]
        mrcnn_mask: [N, depth, height, width, num_classes]
        image_shape: [channels, depth, height, width] Original size of the image before resizing
        window: [z1, y1, x1, z2, y2, x2] Box in the image where the real image is excluding the padding.
        Returns:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, depth] normal shape full mask
        """
        start_time = time.time()
        zero_ix = np.where(detections[:, 6] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        boxes = detections[:N, :6].astype(np.int32)
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]
        masks = mrcnn_mask[np.arange(N), :, :, :, :]
        d_scale = image_shape[1] / (window[3] - window[0])
        h_scale = image_shape[2] / (window[4] - window[1])
        w_scale = image_shape[3] / (window[5] - window[2])
        shift = window[:3]
        scales = np.array([d_scale, h_scale, w_scale, d_scale, h_scale, w_scale])
        shifts = np.array([shift[0], shift[1], shift[2], shift[0], shift[1], shift[2]])
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
        exclude_ix = np.where((boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
        full_masks = utils.unmold_mask(masks[0], boxes[0], image_shape)
        full_mask = np.argmax(full_masks, axis=3)
        boxes[:, [0, 1, 2, 3, 4, 5]] = boxes[:, [1, 2, 0, 4, 5, 3]]
        None
        return boxes, np.arange(1, 8), scores, full_mask.transpose((1, 2, 0))


class TopDownLayer(nn.Module):
    """Generate the Pyramid Feature Maps.
    Returns [p2_out, p3_out, c0_out], where p2_out and p3_out is used for RPN
    and c0_out is used for mrcnn mask branch.
    """

    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        y = F.interpolate(y, scale_factor=2)
        x = self.conv1(x)
        return self.conv2(x + y)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'block': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (TopDownLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Wuziyi616_CFUN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

