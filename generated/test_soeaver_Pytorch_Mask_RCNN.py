import sys
_module = sys.modules[__name__]
del sys
config = _module
convert_weights = _module
demo = _module
eval = _module
lib = _module
build = _module
nms = _module
_ext = _module
pth_nms = _module
nms_wrapper = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
roi_align = _module
crop_and_resize = _module
crop_and_resize = _module
roi_align = _module
network = _module
mask_rcnn = _module
postprocess = _module
ap = _module
visualize = _module
visualize_cap = _module
InputProcess = _module
preprocess = _module
coco_data_pipeline = _module
data_center = _module
test_data_loader = _module
realtime_demo = _module
tasks = _module
AnchorProcess = _module
BboxProcess = _module
bbox = _module
generate_anchors = _module
MaskProcess = _module
merge_task = _module
test_load = _module

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


import itertools


import logging


import math


import random


import re


import time


from collections import OrderedDict


import numpy as np


import scipy.misc


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import torch.backends.cudnn as cudnn


import torch.optim as optim


from torch.autograd import Function


from torch import nn


import torch.utils.model_zoo as model_zoo


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)
        if image.is_cuda:
            _backend.crop_and_resize_gpu_forward(image, boxes, box_ind,
                self.extrapolation_value, self.crop_height, self.crop_width,
                crops)
        else:
            _backend.crop_and_resize_forward(image, boxes, box_ind, self.
                extrapolation_value, self.crop_height, self.crop_width, crops)
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)
        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)
        if grad_outputs.is_cuda:
            _backend.crop_and_resize_gpu_backward(grad_outputs, boxes,
                box_ind, grad_image)
        else:
            _backend.crop_and_resize_backward(grad_outputs, boxes, box_ind,
                grad_image)
        return grad_image, None, None


class CropAndResize(nn.Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width,
            self.extrapolation_value)(image, boxes, box_ind)


class RoIAlign(nn.Module):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, featuremap, boxes, box_ind):
        """
        RoIAlign based on crop_and_resize.
        See more details on https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        spacing_w = (x2 - x1) / float(self.crop_width)
        spacing_h = (y2 - y1) / float(self.crop_height)
        image_height, image_width = featuremap.size()[2:4]
        nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
        ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
        nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1)
        nh = spacing_w * float(self.crop_height - 1) / float(image_height - 1)
        boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        return CropAndResizeFunction(self.crop_height, self.crop_width,
            self.extrapolation_value)(featuremap, boxes, box_ind)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001)
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


class resnet_graph(nn.Module):

    def __init__(self, block, layers, stage5=False):
        self.inplanes = 64
        super(resnet_graph, self).__init__()
        self.stage5 = stage5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if self.stage5:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001))
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
        C1 = self.maxpool(x)
        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        if self.stage5:
            C5 = self.layer4(C4)
        else:
            C5 = None
        return C1, C2, C3, C4, C5


class rpn_graph(nn.Module):

    def __init__(self, input_dims, anchors_per_location, anchor_stride):
        super(rpn_graph, self).__init__()
        self.rpn_conv_shared = nn.Conv2d(input_dims, 512, kernel_size=3,
            stride=anchor_stride, padding=1)
        self.rpn_class_raw = nn.Conv2d(512, 2 * anchors_per_location,
            kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(512, 4 * anchors_per_location,
            kernel_size=1)

    def forward(self, x):
        shared = F.relu(self.rpn_conv_shared(x), True)
        x = self.rpn_class_raw(shared)
        rpn_class_logits = x.permute(0, 2, 3, 1).contiguous().view(x.size(0
            ), -1, 2)
        rpn_probs = F.softmax(rpn_class_logits, dim=-1)
        x = self.rpn_bbox_pred(shared)
        rpn_bbox = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        return rpn_class_logits, rpn_probs, rpn_bbox


def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), math.log(2.0))


def ROIAlign(feature_maps, rois, config, pool_size, mode='bilinear'):
    """Implements ROI Align on the features.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    crop_resize = CropAndResize(pool_size, pool_size, 0)
    roi_number = rois.size()[1]
    pooled = rois.data.new(config.IMAGES_PER_GPU * rois.size(1), 256,
        pool_size, pool_size).zero_()
    rois = rois.view(config.IMAGES_PER_GPU * rois.size(1), 4)
    x_1 = rois[:, (0)]
    y_1 = rois[:, (1)]
    x_2 = rois[:, (2)]
    y_2 = rois[:, (3)]
    roi_level = log2_graph(torch.div(torch.sqrt((y_2 - y_1) * (x_2 - x_1)),
        224.0))
    roi_level = torch.clamp(torch.clamp(torch.add(torch.round(roi_level), 4
        ), min=2), max=5)
    for i, level in enumerate(range(2, 6)):
        scaling_ratio = 2 ** level
        height = float(config.IMAGE_MAX_DIM) / scaling_ratio
        width = float(config.IMAGE_MAX_DIM) / scaling_ratio
        ixx = torch.eq(roi_level, level)
        box_indices = ixx.view(-1).int() * 0
        ix = torch.unsqueeze(ixx, 1)
        level_boxes = torch.masked_select(rois, ix)
        if level_boxes.size()[0] == 0:
            continue
        level_boxes = level_boxes.view(-1, 4)
        crops = crop_resize(feature_maps[i], torch.div(level_boxes, float(
            config.IMAGE_MAX_DIM))[:, ([1, 0, 3, 2])], box_indices)
        indices_pooled = ixx.nonzero()[:, (0)]
        pooled[(indices_pooled.data), :, :, :] = crops.data
    pooled = pooled.view(config.IMAGES_PER_GPU, roi_number, 256, pool_size,
        pool_size)
    pooled = Variable(pooled).cuda()
    return pooled


class fpn_classifier_graph(nn.Module):

    def __init__(self, num_classes, config):
        super(fpn_classifier_graph, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.mrcnn_class_conv1 = nn.Conv2d(256, 1024, kernel_size=self.
            config.POOL_SIZE, stride=1, padding=0)
        self.mrcnn_class_bn1 = nn.BatchNorm2d(1024, eps=0.001)
        self.mrcnn_class_conv2 = nn.Conv2d(1024, 1024, kernel_size=1,
            stride=1, padding=0)
        self.mrcnn_class_bn2 = nn.BatchNorm2d(1024, eps=0.001)
        self.mrcnn_class_logits = nn.Linear(1024, self.num_classes)
        self.mrcnn_bbox_fc = nn.Linear(1024, self.num_classes * 4)

    def forward(self, x, rpn_rois):
        start = time.time()
        x = ROIAlign(x, rpn_rois, self.config, self.config.POOL_SIZE)
        spend = time.time() - start
        None
        roi_number = x.size()[1]
        x = x.view(self.config.IMAGES_PER_GPU * roi_number, 256, self.
            config.POOL_SIZE, self.config.POOL_SIZE)
        x = self.mrcnn_class_conv1(x)
        x = self.mrcnn_class_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.mrcnn_class_conv2(x)
        x = self.mrcnn_class_bn2(x)
        x = F.relu(x, inplace=True)
        shared = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        mrcnn_class_logits = self.mrcnn_class_logits(shared)
        mrcnn_probs = F.softmax(mrcnn_class_logits, dim=-1)
        x = self.mrcnn_bbox_fc(shared)
        mrcnn_bbox = x.view(x.size()[0], self.num_classes, 4)
        mrcnn_class_logits = mrcnn_class_logits.view(self.config.
            IMAGES_PER_GPU, roi_number, mrcnn_class_logits.size()[-1])
        mrcnn_probs = mrcnn_probs.view(self.config.IMAGES_PER_GPU,
            roi_number, mrcnn_probs.size()[-1])
        mrcnn_bbox = mrcnn_bbox.view(self.config.IMAGES_PER_GPU, roi_number,
            self.config.NUM_CLASSES, 4)
        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


class build_fpn_mask_graph(nn.Module):

    def __init__(self, num_classes, config):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from diffent layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_shape: [height, width, depth]
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results

        Returns: Masks [batch, roi_count, height, width, num_classes]
        """
        super(build_fpn_mask_graph, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.mrcnn_mask_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1)
        self.mrcnn_mask_bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.mrcnn_mask_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1)
        self.mrcnn_mask_bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.mrcnn_mask_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1)
        self.mrcnn_mask_bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.mrcnn_mask_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
            padding=1)
        self.mrcnn_mask_bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.mrcnn_mask_deconv = nn.ConvTranspose2d(256, 256, kernel_size=2,
            stride=2)
        self.mrcnn_mask = nn.Conv2d(256, self.num_classes, kernel_size=1,
            stride=1)

    def forward(self, x, rpn_rois):
        x = ROIAlign(x, rpn_rois, self.config, self.config.MASK_POOL_SIZE)
        roi_number = x.size()[1]
        x = x.view(self.config.IMAGES_PER_GPU * roi_number, 256, self.
            config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE)
        x = self.mrcnn_mask_conv1(x)
        x = self.mrcnn_mask_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.mrcnn_mask_conv2(x)
        x = self.mrcnn_mask_bn2(x)
        x = F.relu(x, inplace=True)
        x = self.mrcnn_mask_conv3(x)
        x = self.mrcnn_mask_bn3(x)
        x = F.relu(x, inplace=True)
        x = self.mrcnn_mask_conv4(x)
        x = self.mrcnn_mask_bn4(x)
        x = F.relu(x, inplace=True)
        x = self.mrcnn_mask_deconv(x)
        x = F.relu(x, inplace=True)
        x = self.mrcnn_mask(x)
        x = x.view(self.config.IMAGES_PER_GPU, roi_number, self.config.
            NUM_CLASSES, self.config.MASK_POOL_SIZE * 2, self.config.
            MASK_POOL_SIZE * 2)
        return x


def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    height = boxes[:, :, (2)] - boxes[:, :, (0)]
    width = boxes[:, :, (3)] - boxes[:, :, (1)]
    center_y = boxes[:, :, (0)] + 0.5 * height
    center_x = boxes[:, :, (1)] + 0.5 * width
    center_y += deltas[:, :, (0)] * height
    center_x += deltas[:, :, (1)] * width
    height *= torch.exp(deltas[:, :, (2)])
    width *= torch.exp(deltas[:, :, (3)])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = [y1, x1, y2, x2]
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    wy1, wx1, wy2, wx2 = window
    y1, x1, y2, x2 = boxes
    y1 = torch.max(torch.min(y1, wy2), wy1)
    x1 = torch.max(torch.min(x1, wx2), wx1)
    y2 = torch.max(torch.min(y2, wy2), wy1)
    x2 = torch.max(torch.min(x2, wx2), wx1)
    clipped = torch.stack([x1, y1, x2, y2], dim=2)
    return clipped


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([
        -1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 
        0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes,
    feature_strides, anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i
            ], feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    target_class_ids = target_class_ids.contiguous().view(-1)
    target_bbox = target_bbox.contiguous().view(-1, 4)
    pred_bbox = pred_bbox.contiguous().view(-1, pred_bbox.size()[2], 4)
    positive_roi_ix = torch.gt(target_class_ids, 0)
    positive_roi_class_ids = torch.masked_select(target_class_ids,
        positive_roi_ix)
    indices = target_class_ids
    loss = F.smooth_l1_loss(pred_bbox, target_bbox, size_average=True)
    return loss


def mrcnn_class_loss(target_class_ids, pred_class_logits, active_class_ids,
    config):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    pred_class_logits = pred_class_logits.contiguous().view(-1, config.
        NUM_CLASSES)
    target_class_ids = target_class_ids.contiguous().view(-1).type(torch.
        cuda.LongTensor)
    loss = F.cross_entropy(pred_class_logits, target_class_ids, weight=None,
        size_average=True)
    return loss


def mrcnn_mask_loss(target_masks, target_class_ids, pred_masks_logits):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    target_class_ids = target_class_ids.view(-1)
    loss = F.binary_cross_entropy_with_logits(pred_masks_logits, target_masks)
    return loss


def pth_nms(dets, thresh):
    """
  dets has to be a tensor
  """
    if not dets.is_cuda:
        x1 = dets[:, (0)]
        y1 = dets[:, (1)]
        x2 = dets[:, (2)]
        y2 = dets[:, (3)]
        scores = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.cpu_nms(keep, num_out, dets, order, areas, thresh)
        return keep[:num_out[0]]
    else:
        x1 = dets[:, (0)]
        y1 = dets[:, (1)]
        x2 = dets[:, (2)]
        y2 = dets[:, (3)]
        scores = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        dets = dets[order].contiguous()
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.gpu_nms(keep, num_out, dets, thresh)
        return order[keep[:num_out[0]].cuda()].contiguous()


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
    return pth_nms(dets, thresh)


def rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox, config):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    indices = torch.eq(rpn_match, 1)
    rpn_bbox = torch.masked_select(rpn_bbox, indices)
    batch_counts = torch.sum(indices.float(), dim=1)
    outputs = []
    for i in range(config.IMAGES_PER_GPU):
        outputs.append(target_bbox[torch.cuda.LongTensor([i]), torch.arange
            (int(batch_counts[i].cpu().data.numpy()[0])).type(torch.cuda.
            LongTensor)])
    target_bbox = torch.cat(outputs, dim=0)
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox, size_average=True)
    return loss


def rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    anchor_class = torch.eq(rpn_match, 1)
    indices = torch.ne(rpn_match, 0.0)
    rpn_class_logits = torch.masked_select(rpn_class_logits, indices)
    anchor_class = torch.masked_select(anchor_class, indices)
    rpn_class_logits = rpn_class_logits.contiguous().view(-1, 2)
    anchor_class = anchor_class.contiguous().view(-1).type(torch.cuda.
        LongTensor)
    loss = F.cross_entropy(rpn_class_logits, anchor_class, weight=None)
    return loss


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)
    height = box[:, (2)] - box[:, (0)]
    width = box[:, (3)] - box[:, (1)]
    center_y = box[:, (0)] + 0.5 * height
    center_x = box[:, (1)] + 0.5 * width
    gt_height = gt_box[:, (2)] - gt_box[:, (0)]
    gt_width = gt_box[:, (3)] - gt_box[:, (1)]
    gt_center_y = gt_box[:, (0)] + 0.5 * gt_height
    gt_center_x = gt_box[:, (1)] + 0.5 * gt_width
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)
    return np.stack([dy, dx, dh, dw], axis=1)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    y1 = np.maximum(box[0], boxes[:, (0)])
    y2 = np.minimum(box[2], boxes[:, (2)])
    x1 = np.maximum(box[1], boxes[:, (1)])
    x2 = np.minimum(box[3], boxes[:, (3)])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config
    ):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.
    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Grund truth masks. Can be full
              size or mini-masks.
    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinments.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, 'Expected int but got {}'.format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, 'Expected int but got {}'.format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, 'Expected bool but got {}'.format(
        gt_masks.dtype)
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, 'Image must contain instances.'
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, (instance_ids)]
    rpn_roi_area = (rpn_rois[:, (2)] - rpn_rois[:, (0)]) * (rpn_rois[:, (3)
        ] - rpn_rois[:, (1)])
    gt_box_area = (gt_boxes[:, (2)] - gt_boxes[:, (0)]) * (gt_boxes[:, (3)] -
        gt_boxes[:, (1)])
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, (i)] = compute_iou(gt, rpn_rois, gt_box_area[i],
            rpn_roi_area)
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax
        ]
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        if keep.shape[0] == 0:
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            keep_extra_ids = np.random.choice(keep_bg_ids, remaining,
                replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0
        ] == config.TRAIN_ROIS_PER_IMAGE, "keep doesn't match ROI batch size {}, {}".format(
        keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)
    rpn_roi_gt_boxes[(keep_bg_ids), :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES, 4),
        dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = box_refinement(rois[
        pos_ids], roi_gt_boxes[(pos_ids), :4])
    bboxes /= config.BBOX_STD_DEV
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0],
        config.MASK_SHAPE[1], config.NUM_CLASSES), dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, 'class id must be greater than 0'
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, (gt_id)]
        if config.USE_MINI_MASK:
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = np.round(scipy.misc.
                imresize(class_mask.astype(float), (gt_h, gt_w), interp=
                'nearest') / 255.0).astype(bool)
            class_mask = placeholder
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = scipy.misc.imresize(m.astype(float), config.MASK_SHAPE,
            interp='nearest') / 255.0
        masks[(i), :, :, (class_id)] = mask
    return rois, roi_gt_class_ids, bboxes, masks


def stage2_target(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    batch_rois = []
    batch_mrcnn_class_ids = []
    batch_mrcnn_bbox = []
    batch_mrcnn_mask = []
    for i in range(config.IMAGES_PER_GPU):
        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = (
            build_detection_targets(rpn_rois[i], gt_class_ids[i], gt_boxes[
            i], gt_masks[i], config))
        batch_rois.append(rois)
        batch_mrcnn_class_ids.append(mrcnn_class_ids)
        batch_mrcnn_bbox.append(mrcnn_bbox)
        batch_mrcnn_mask.append(mrcnn_mask)
    batch_rois = np.array(batch_rois)
    batch_mrcnn_class_ids = np.array(batch_mrcnn_class_ids)
    batch_mrcnn_bbox = np.array(batch_mrcnn_bbox)
    batch_mrcnn_mask = np.array(batch_mrcnn_mask)
    return (batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox,
        batch_mrcnn_mask)


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable


class MaskRCNN(nn.Module):
    """
    Encapsulates the Mask RCNN model functionality.
    
    """

    def __init__(self, config, mode='inference'):
        super(MaskRCNN, self).__init__()
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.mode = mode
        self.resnet_graph = resnet_graph(Bottleneck, [3, 4, 23, 3], stage5=True
            )
        self.fpn_c5p5 = nn.Conv2d(512 * 4, 256, kernel_size=1, stride=1,
            padding=0)
        self.fpn_c4p4 = nn.Conv2d(256 * 4, 256, kernel_size=1, stride=1,
            padding=0)
        self.fpn_c3p3 = nn.Conv2d(128 * 4, 256, kernel_size=1, stride=1,
            padding=0)
        self.fpn_c2p2 = nn.Conv2d(64 * 4, 256, kernel_size=1, stride=1,
            padding=0)
        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.scale_ratios = [4, 8, 16, 32]
        self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0,
            ceil_mode=False)
        self.anchors = generate_pyramid_anchors(self.config.
            RPN_ANCHOR_SCALES, self.config.RPN_ANCHOR_RATIOS, self.config.
            BACKBONE_SHAPES, self.config.BACKBONE_STRIDES, self.config.
            RPN_ANCHOR_STRIDE)
        self.anchors = self.anchors.astype(np.float32)
        self.rpn = rpn_graph(256, len(self.config.RPN_ANCHOR_RATIOS), self.
            config.RPN_ANCHOR_STRIDE)
        self.rpn_mask = build_fpn_mask_graph(config.NUM_CLASSES, config)
        self.rpn_class = fpn_classifier_graph(config.NUM_CLASSES, config)
        self.proposal_count = (self.config.POST_NMS_ROIS_TRAINING if self.
            mode == 'training' else self.config.POST_NMS_ROIS_INFERENCE)
        self._initialize_weights()

    def forward(self, x):
        start = time.time()
        saved_for_loss = []
        C1, C2, C3, C4, C5 = self.resnet_graph(x)
        resnet_time = time.time()
        None
        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + F.upsample(P5, scale_factor=2, mode='bilinear'
            )
        P3 = self.fpn_c3p3(C3) + F.upsample(P4, scale_factor=2, mode='bilinear'
            )
        P2 = self.fpn_c2p2(C2) + F.upsample(P3, scale_factor=2, mode='bilinear'
            )
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        P6 = self.fpn_p6(P5)
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        self.mrcnn_feature_maps = [P2, P3, P4, P5]
        rpn_class_logits_outputs = []
        rpn_class_outputs = []
        rpn_bbox_outputs = []
        for p in rpn_feature_maps:
            rpn_class_logits, rpn_probs, rpn_bbox = self.rpn(p)
            rpn_class_logits_outputs.append(rpn_class_logits)
            rpn_class_outputs.append(rpn_probs)
            rpn_bbox_outputs.append(rpn_bbox)
        rpn_class_logits = torch.cat(rpn_class_logits_outputs, dim=1)
        rpn_class = torch.cat(rpn_class_outputs, dim=1)
        rpn_bbox = torch.cat(rpn_bbox_outputs, dim=1)
        rpn_rois = self.proposal_layer(rpn_class, rpn_bbox)
        spend = time.time() - resnet_time
        None
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.rpn_class(self.
            mrcnn_feature_maps, rpn_rois)
        mrcnn_masks_logits = self.rpn_mask(self.mrcnn_feature_maps, rpn_rois)
        if self.mode == 'training':
            return [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois,
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_masks_logits
                ], [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois,
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_masks_logits
                ]
        else:
            return [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois,
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_masks_logits
                ]

    def proposal_layer(self, rpn_class, rpn_bbox):
        scores = rpn_class[:, :, (1)]
        deltas_mul = Variable(torch.from_numpy(np.reshape(self.config.
            RPN_BBOX_STD_DEV, [1, 1, 4]).astype(np.float32)))
        deltas = rpn_bbox * deltas_mul
        pre_nms_limit = min(6000, self.anchors.shape[0])
        scores, ix = torch.topk(scores, pre_nms_limit, dim=-1, largest=True,
            sorted=True)
        ix = torch.unsqueeze(ix, 2)
        ix = torch.cat([ix, ix, ix, ix], dim=2)
        deltas = torch.gather(deltas, 1, ix)
        _anchors = []
        for i in range(self.config.IMAGES_PER_GPU):
            anchors = Variable(torch.from_numpy(self.anchors.astype(np.
                float32)))
            _anchors.append(anchors)
        anchors = torch.stack(_anchors, 0)
        pre_nms_anchors = torch.gather(anchors, 1, ix)
        refined_anchors = apply_box_deltas_graph(pre_nms_anchors, deltas)
        height, width = self.config.IMAGE_SHAPE[:2]
        window = np.array([0, 0, height, width]).astype(np.float32)
        window = Variable(torch.from_numpy(window))
        refined_anchors_clipped = clip_boxes_graph(refined_anchors, window)
        refined_proposals = []
        for i in range(self.config.IMAGES_PER_GPU):
            indices = nms(torch.cat([refined_anchors_clipped.data[i],
                scores.data[i]], 1), 0.7)
            indices = indices[:self.proposal_count]
            indices = torch.stack([indices, indices, indices, indices], dim=1)
            indices = Variable(indices)
            proposals = torch.gather(refined_anchors_clipped[i], 0, indices)
            padding = self.proposal_count - proposals.size()[0]
            proposals = torch.cat([proposals, Variable(torch.zeros([padding,
                4]))], 0)
            refined_proposals.append(proposals)
        rpn_rois = torch.stack(refined_proposals, 0)
        return rpn_rois

    @staticmethod
    def build_loss(saved_for_loss, ground_truths, config):
        saved_for_log = OrderedDict()
        (predict_rpn_class_logits, predict_rpn_class, predict_rpn_bbox,
            predict_rpn_rois, predict_mrcnn_class_logits,
            predict_mrcnn_class, predict_mrcnn_bbox, predict_mrcnn_masks_logits
            ) = saved_for_loss
        (batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids,
            batch_gt_boxes, batch_gt_masks, active_class_ids) = ground_truths
        rpn_rois = predict_rpn_rois.cpu().data.numpy()
        rpn_rois = rpn_rois[:, :, ([1, 0, 3, 2])]
        (batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask
            ) = (stage2_target(rpn_rois, batch_gt_class_ids, batch_gt_boxes,
            batch_gt_masks, config))
        batch_mrcnn_mask = batch_mrcnn_mask.transpose(0, 1, 4, 2, 3)
        batch_mrcnn_class_ids = to_variable(batch_mrcnn_class_ids)
        batch_mrcnn_bbox = to_variable(batch_mrcnn_bbox)
        batch_mrcnn_mask = to_variable(batch_mrcnn_mask)
        rpn_cls_loss = rpn_class_loss(batch_rpn_match, predict_rpn_class_logits
            )
        rpn_reg_loss = rpn_bbox_loss(batch_rpn_bbox, batch_rpn_match,
            predict_rpn_bbox, config)
        stage2_reg_loss = mrcnn_bbox_loss(batch_mrcnn_bbox,
            batch_mrcnn_class_ids, predict_mrcnn_bbox)
        stage2_cls_loss = mrcnn_class_loss(batch_mrcnn_class_ids,
            predict_mrcnn_class_logits, active_class_ids, config)
        stage2_mask_loss = mrcnn_mask_loss(batch_mrcnn_mask,
            batch_mrcnn_class_ids, predict_mrcnn_masks_logits)
        total_loss = (rpn_cls_loss + rpn_reg_loss + stage2_cls_loss +
            stage2_reg_loss + stage2_mask_loss)
        saved_for_log['rpn_cls_loss'] = rpn_cls_loss.data[0]
        saved_for_log['rpn_reg_loss'] = rpn_reg_loss.data[0]
        saved_for_log['stage2_cls_loss'] = stage2_cls_loss.data[0]
        saved_for_log['stage2_reg_loss'] = stage2_reg_loss.data[0]
        saved_for_log['stage2_mask_loss'] = stage2_mask_loss.data[0]
        saved_for_log['total_loss'] = total_loss.data[0]
        return total_loss, saved_for_log

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_soeaver_Pytorch_Mask_RCNN(_paritybench_base):
    pass
    def test_000(self):
        self._check(rpn_graph(*[], **{'input_dims': 4, 'anchors_per_location': 4, 'anchor_stride': 1}), [torch.rand([4, 4, 4, 4])], {})

