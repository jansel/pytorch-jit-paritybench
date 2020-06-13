import sys
_module = sys.modules[__name__]
del sys
coco = _module
config = _module
demo_coco = _module
demo_synthia = _module
model = _module
build = _module
nms_wrapper = _module
pth_nms = _module
crop_and_resize = _module
roi_align = _module
synthia = _module
utils = _module
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


import math


import random


import re


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


from torch.autograd import Function


from torch import nn


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = (out_width - 1) * self.stride[0] + self.kernel_size[0
            ] - in_width
        pad_along_height = (out_height - 1) * self.stride[1
            ] + self.kernel_size[1] - in_height
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom),
            'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=0.001,
            momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.padding2(out)
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


class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ['resnet50', 'resnet101']
        self.inplanes = 64
        self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5
        self.C1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2,
            padding=3), nn.BatchNorm2d(64, eps=0.001, momentum=0.01), nn.
            ReLU(inplace=True), SamePad2d(kernel_size=3, stride=2), nn.
            MaxPool2d(kernel_size=3, stride=2))
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2
                )
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride), nn.
                BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class FPN(nn.Module):

    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1,
            stride=1)
        self.P2_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
            stride=1))
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1,
            stride=1)
        self.P3_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
            stride=1))
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1,
            stride=1)
        self.P4_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
            stride=1))
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1,
            stride=1)
        self.P5_conv2 = nn.Sequential(SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,
            stride=1))
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)
        p6_out = self.P6(p5_out)
        return [p2_out, p3_out, p4_out, p5_out, p6_out]


class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth
        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride
            =self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location,
            kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location,
            kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.conv_shared(self.padding(x)))
        rpn_class_logits = self.conv_class(x)
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)
        rpn_probs = self.softmax(rpn_class_logits)
        rpn_bbox = self.conv_bbox(x)
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)
        return [rpn_class_logits, rpn_probs, rpn_bbox]


def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


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


def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)
    boxes = inputs[0]
    feature_maps = inputs[1:]
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1
    image_area = Variable(torch.FloatTensor([float(image_shape[0] *
        image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, (0)]
        level_boxes = boxes[(ix.data), :]
        box_to_level.append(ix.data)
        level_boxes = level_boxes.detach()
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False
            ).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(
            feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)
    pooled = torch.cat(pooled, dim=0)
    box_to_level = torch.cat(box_to_level, dim=0)
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[(box_to_level), :, :]
    return pooled


class Classifier(nn.Module):

    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size,
            stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)
        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)
        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):

    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array([image_id] + list(image_shape) + list(window) + list(
        active_class_ids))
    return meta


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def load_image_gt(dataset, config, image_id, augment=False, use_mini_mask=False
    ):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = utils.resize_image(image, min_dim=
        config.IMAGE_MIN_DIM, max_dim=config.IMAGE_MAX_DIM, padding=config.
        IMAGE_PADDING)
    mask = utils.resize_mask(mask, scale, padding)
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
    bbox = utils.extract_bboxes(mask)
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id
        ]['source']]
    active_class_ids[source_class_ids] = 1
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    return image, image_meta, class_ids, bbox, mask


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)
    overlaps = utils.compute_overlaps(anchors, gt_boxes)
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & no_crowd_bool] = -1
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    rpn_match[anchor_iou_max >= 0.7] = 1
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(
        rpn_match == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchor_iou_argmax[i]]
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w
        rpn_bbox[ix] = [(gt_center_y - a_center_y) / a_h, (gt_center_x -
            a_center_x) / a_w, np.log(gt_h / a_h), np.log(gt_w / a_w)]
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, config, augment=True):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.

            dataset: The Dataset object to pick data from
            config: The model config object
            shuffle: If True, shuffles the samples before every epoch
            augment: If True, applies image augmentation to images (currently only
                     horizontal flips are supported)

            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containtes
            of the lists differs depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_metas: [batch, size of image meta]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas,
                and masks.
            """
        self.b = 0
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0
        self.dataset = dataset
        self.config = config
        self.augment = augment
        self.anchors = utils.generate_pyramid_anchors(config.
            RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, config.
            BACKBONE_SHAPES, config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        image, image_metas, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
            self.dataset, self.config, image_id, augment=self.augment,
            use_mini_mask=self.config.USE_MINI_MASK)
        if not np.any(gt_class_ids > 0):
            return None
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
            gt_class_ids, gt_boxes, self.config)
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(np.arange(gt_boxes.shape[0]), self.
                config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, (ids)]
        rpn_match = rpn_match[:, (np.newaxis)]
        images = mold_image(image.astype(np.float32), self.config)
        images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)
            ).float()
        return (images, image_metas, rpn_match, rpn_bbox, gt_class_ids,
            gt_boxes, gt_masks)

    def __len__(self):
        return self.image_ids.shape[0]


def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]


def pth_nms(dets, thresh):
    """
  dets has to be a tensor
  """
    if not dets.is_cuda:
        x1 = dets[:, (1)]
        y1 = dets[:, (0)]
        x2 = dets[:, (3)]
        y2 = dets[:, (2)]
        scores = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.cpu_nms(keep, num_out, dets, order, areas, thresh)
        return keep[:num_out[0]]
    else:
        x1 = dets[:, (1)]
        y1 = dets[:, (0)]
        x2 = dets[:, (3)]
        y2 = dets[:, (2)]
        scores = dets[:, (4)]
        dets_temp = torch.FloatTensor(dets.size()).cuda()
        dets_temp[:, (0)] = dets[:, (1)]
        dets_temp[:, (1)] = dets[:, (0)]
        dets_temp[:, (2)] = dets[:, (3)]
        dets_temp[:, (3)] = dets[:, (2)]
        dets_temp[:, (4)] = dets[:, (4)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        dets = dets[order].contiguous()
        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        nms.gpu_nms(keep, num_out, dets_temp, thresh)
        return order[keep[:num_out[0]].cuda()].contiguous()


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
    return pth_nms(dets, thresh)


def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, (0)] = boxes[:, (0)].clamp(float(window[0]), float(window[2]))
    boxes[:, (1)] = boxes[:, (1)].clamp(float(window[1]), float(window[3]))
    boxes[:, (2)] = boxes[:, (2)].clamp(float(window[0]), float(window[2]))
    boxes[:, (3)] = boxes[:, (3)].clamp(float(window[1]), float(window[3]))
    return boxes


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    height = boxes[:, (2)] - boxes[:, (0)]
    width = boxes[:, (3)] - boxes[:, (1)]
    center_y = boxes[:, (0)] + 0.5 * height
    center_x = boxes[:, (1)] + 0.5 * width
    center_y += deltas[:, (0)] * height
    center_x += deltas[:, (1)] * width
    height *= torch.exp(deltas[:, (2)])
    width *= torch.exp(deltas[:, (3)])
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    _, class_ids = torch.max(probs, dim=1)
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV,
        [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height,
        width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale
    refined_rois = clip_to_window(window, refined_rois)
    refined_rois = torch.round(refined_rois)
    keep_bool = class_ids > 0
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.
            DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, (0)]
    pre_nms_class_ids = class_ids[keep.data]
    pre_nms_scores = class_scores[keep.data]
    pre_nms_rois = refined_rois[keep.data]
    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, (0)]
        ix_rois = pre_nms_rois[ixs.data]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[(order.data), :]
        class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1
            ).data, config.DETECTION_NMS_THRESHOLD)
        class_keep = keep[ixs[order[class_keep].data].data]
        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]
    result = torch.cat((refined_rois[keep.data], class_ids[keep.data].
        unsqueeze(1).float(), class_scores[keep.data].unsqueeze(1)), dim=1)
    return result


def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, (0)]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """
    rois = rois.squeeze(0)
    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window,
        config)
    return detections


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack([boxes[:, (0)].clamp(float(window[0]), float(window
        [2])), boxes[:, (1)].clamp(float(window[1]), float(window[3])),
        boxes[:, (2)].clamp(float(window[0]), float(window[2])), boxes[:, (
        3)].clamp(float(window[1]), float(window[3]))], 1)
    return boxes


def proposal_generator(inputs, proposal_count, nms_threshold, anchors,
    config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)
    scores = inputs[0][:, (1)]
    deltas = inputs[1]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV,
        [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[(order.data), :]
    anchors = anchors[(order.data), :]
    boxes = apply_box_deltas(anchors, deltas)
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[(keep), :]
    norm = Variable(torch.from_numpy(np.array([height, width, height, width
        ])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm
    normalized_boxes = normalized_boxes.unsqueeze(0)
    return normalized_boxes


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += 'shape: {:20}  min: {:10.5f}  max: {:10.5f}'.format(str(
            array.shape), array.min() if array.size else '', array.max() if
            array.size else '')
    print(text)


def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, (0)]
    x1 = torch.max(b1_x1, b2_x1)[:, (0)]
    y2 = torch.min(b1_y2, b2_y2)[:, (0)]
    x2 = torch.min(b1_x2, b2_x2)[:, (0)]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, (0)] + b2_area[:, (0)] - intersection
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config
    ):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    if torch.nonzero(gt_class_ids < 0).size()[0]:
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, (0)]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, (0)]
        crowd_boxes = gt_boxes[(crowd_ix.data), :]
        crowd_masks = gt_masks[(crowd_ix.data), :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[(non_crowd_ix.data), :]
        gt_masks = gt_masks[(non_crowd_ix.data), :]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [
            True]), requires_grad=False)
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool.cuda()
    overlaps = bbox_overlaps(proposals, gt_boxes)
    roi_iou_max = torch.max(overlaps, dim=1)[0]
    positive_roi_bool = roi_iou_max >= 0.5
    if torch.nonzero(positive_roi_bool).size()[0]:
        positive_indices = torch.nonzero(positive_roi_bool)[:, (0)]
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.
            ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[(positive_indices.data), :]
        positive_overlaps = overlaps[(positive_indices.data), :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[(roi_gt_box_assignment.data), :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]
        deltas = Variable(utils.box_refinement(positive_rois.data,
            roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(),
            requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev
        roi_masks = gt_masks[(roi_gt_box_assignment.data), :, :]
        boxes = positive_rois
        if config.USE_MINI_MASK:
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad
            =False).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()
        masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config
            .MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data,
            requires_grad=False)
        masks = masks.squeeze(1)
        masks = torch.round(masks)
    else:
        positive_count = 0
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    if torch.nonzero(negative_roi_bool).size()[0] and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, (0)]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[(negative_indices.data), :]
    else:
        negative_count = 0
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int(
            )
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0],
            config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False
            ).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0],
            config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
    return rois, roi_gt_class_ids, deltas, masks


def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    rpn_match = rpn_match.squeeze(2)
    anchor_class = (rpn_match == 1).long()
    indices = torch.nonzero(rpn_match != 0)
    rpn_class_logits = rpn_class_logits[(indices.data[:, (0)]), (indices.
        data[:, (1)]), :]
    anchor_class = anchor_class[indices.data[:, (0)], indices.data[:, (1)]]
    loss = F.cross_entropy(rpn_class_logits, anchor_class)
    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size()[0]:
        positive_ix = torch.nonzero(target_class_ids > 0)[:, (0)]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)
        y_true = target_masks[(indices[:, (0)].data), :, :]
        y_pred = pred_masks[(indices[:, (0)].data), (indices[:, (1)].data),
            :, :]
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    if target_class_ids.size()[0]:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    rpn_match = rpn_match.squeeze(2)
    indices = torch.nonzero(rpn_match == 1)
    rpn_bbox = rpn_bbox[indices.data[:, (0)], indices.data[:, (1)]]
    target_bbox = target_bbox[(0), :rpn_bbox.size()[0], :]
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)
    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    if target_class_ids.size()[0]:
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, (0)]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)
        target_bbox = target_bbox[(indices[:, (0)].data), :]
        pred_bbox = pred_bbox[(indices[:, (0)].data), (indices[:, (1)].data), :
            ]
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
    return loss


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox,
    target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
    target_mask, mrcnn_mask):
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids,
        mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas,
        target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids,
        mrcnn_mask)
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss,
        mrcnn_bbox_loss, mrcnn_mask_loss]


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception(
                'Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc. '
                )
        resnet = ResNet('resnet101', stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)
        self.anchors = Variable(torch.from_numpy(utils.
            generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.
            RPN_ANCHOR_RATIOS, config.BACKBONE_SHAPES, config.
            BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)).float(),
            requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.
            RPN_ANCHOR_STRIDE, 256)
        self.classifier = Classifier(256, config.POOL_SIZE, config.
            IMAGE_SHAPE, config.NUM_CLASSES)
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE,
            config.NUM_CLASSES)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        self.epoch = 0
        now = datetime.datetime.now()
        if model_path:
            regex = (
                '.*/\\w+(\\d{4})(\\d{2})(\\d{2})T(\\d{2})(\\d{2})/mask\\_rcnn\\_\\w+(\\d{4})\\.pth'
                )
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))
        self.log_dir = os.path.join(self.model_dir, '{}{:%Y%m%dT%H%M}'.
            format(self.config.NAME.lower(), now))
        self.checkpoint_path = os.path.join(self.log_dir,
            'mask_rcnn_{}_*epoch*.pth'.format(self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace('*epoch*', '{:04d}'
            )

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith('mask_rcnn'), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            None
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def load_pre_weights(self, filepath):
        """load pre-trained weights from coco or imagenet
        """
        if os.path.exists(filepath):
            pretrained_dict = torch.load(filepath)
            model_dict = self.state_dict()
            del pretrained_dict['classifier.linear_class.bias']
            del pretrained_dict['classifier.linear_class.weight']
            del pretrained_dict['classifier.linear_bbox.bias']
            del pretrained_dict['classifier.linear_bbox.weight']
            del pretrained_dict['mask.conv5.bias']
            del pretrained_dict['mask.conv5.weight']
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            None
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images, device):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        molded_images, image_metas, windows = self.mold_inputs(images)
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)
            ).float()
        if self.config.GPU_COUNT:
            molded_images = molded_images.to(device)
        with torch.no_grad():
            molded_images = Variable(molded_images)
        detections, mrcnn_mask = self.predict([molded_images, image_metas],
            mode='inference')
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = (self.
                unmold_detections(detections[i], mrcnn_mask[i], image.shape,
                windows[i]))
            results.append({'rois': final_rois, 'class_ids':
                final_class_ids, 'scores': final_scores, 'masks': final_masks})
        return results

    def collate_custom(self, batch):
        """ Convert the input tensors into lists, to enable multi-image batch training
        """
        images = [item[0] for item in batch]
        image_metas = [item[1] for item in batch]
        rpn_match = [item[2] for item in batch]
        rpn_bbox = [item[3] for item in batch]
        gt_class_ids = [item[4] for item in batch]
        gt_boxes = [item[5] for item in batch]
        gt_masks = [item[6] for item in batch]
        return [images, image_metas, rpn_match, rpn_bbox, gt_class_ids,
            gt_boxes, gt_masks]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs,
        BatchSize, steps, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        layer_regex = {'heads':
            '(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)'
            , '3+':
            '(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)'
            , '4+':
            '(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)'
            , '5+':
            '(fpn.C5.*)|(fpn.P5\\_.*)|(fpn.P4\\_.*)|(fpn.P3\\_.*)|(fpn.P2\\_.*)|(rpn.*)|(classifier.*)|(mask.*)'
            , 'all': '.*'}
        if layers in layer_regex.keys():
            layers = layer_regex[layers]
        train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size
            =BatchSize, collate_fn=self.collate_custom, shuffle=True,
            num_workers=1)
        log('\nStarting at epoch {}. LR={}\n'.format(self.epoch, learning_rate)
            )
        log('Checkpoint Path: {}'.format(self.checkpoint_path))
        self.set_trainable(layers)
        self.train()

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(set_bn_eval)
        trainables_wo_bn = [param for name, param in self.named_parameters(
            ) if param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.
            named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([{'params': trainables_wo_bn, 'weight_decay':
            self.config.WEIGHT_DECAY}, {'params': trainables_only_bn}], lr=
            learning_rate, momentum=self.config.LEARNING_MOMENTUM)
        for epoch in range(self.epoch + 1, epochs + 1):
            log('Epoch {}/{}.'.format(epoch, epochs))
            epoch_loss = 0
            step = 0
            for inputs in train_generator:
                optimizer.zero_grad()
                loss = 0
                (images, image_metas_batch, rpn_match_batch, rpn_bbox_batch,
                    gt_class_ids_batch, gt_boxes_batch, gt_masks_batch
                    ) = inputs
                images = torch.stack(images)
                [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(images)
                rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
                layer_outputs = []
                for p in rpn_feature_maps:
                    layer_outputs.append(self.rpn(p))
                outputs = list(zip(*layer_outputs))
                outputs = [torch.cat(list(o), dim=1) for o in outputs]
                (rpn_class_logits_batch, rpn_class_batch, rpn_pred_bbox_batch
                    ) = outputs
                proposal_count = self.config.POST_NMS_ROIS_TRAINING
                for i in range(BatchSize):
                    image_metas = image_metas_batch[i]
                    rpn_class_logits = rpn_class_logits_batch[i]
                    rpn_class = rpn_class_batch[i]
                    rpn_match = rpn_match_batch[i]
                    rpn_bbox = rpn_bbox_batch[i]
                    rpn_pred_bbox = rpn_pred_bbox_batch[i]
                    gt_class_ids = gt_class_ids_batch[i]
                    gt_boxes = gt_boxes_batch[i]
                    gt_masks = gt_masks_batch[i]
                    image_metas = image_metas.unsqueeze(0)
                    rpn_class_logits = rpn_class_logits.unsqueeze(0)
                    rpn_class = rpn_class.unsqueeze(0)
                    rpn_match = rpn_match.unsqueeze(0)
                    rpn_bbox = rpn_bbox.unsqueeze(0)
                    rpn_pred_bbox = rpn_pred_bbox.unsqueeze(0)
                    gt_class_ids = gt_class_ids.unsqueeze(0)
                    gt_boxes = gt_boxes.unsqueeze(0)
                    gt_masks = gt_masks.unsqueeze(0)
                    image_metas = image_metas.numpy()
                    if self.config.GPU_COUNT:
                        images = images
                        rpn_match = rpn_match
                        rpn_bbox = rpn_bbox
                        gt_class_ids = gt_class_ids
                        gt_boxes = gt_boxes
                        gt_masks = gt_masks
                    rpn_rois = proposal_generator([rpn_class, rpn_pred_bbox
                        ], proposal_count=proposal_count, nms_threshold=
                        self.config.RPN_NMS_THRESHOLD, anchors=self.anchors,
                        config=self.config)
                    h, w = self.config.IMAGE_SHAPE[:2]
                    scale = Variable(torch.from_numpy(np.array([h, w, h, w]
                        )).float(), requires_grad=False)
                    if self.config.GPU_COUNT:
                        scale = scale
                    gt_boxes = gt_boxes / scale
                    rois, target_class_ids, target_deltas, target_mask = (
                        detection_target_layer(rpn_rois, gt_class_ids,
                        gt_boxes, gt_masks, self.config))
                    if not rois.size()[0]:
                        mrcnn_class_logits = Variable(torch.FloatTensor())
                        mrcnn_class = Variable(torch.IntTensor())
                        mrcnn_bbox = Variable(torch.FloatTensor())
                        mrcnn_mask = Variable(torch.FloatTensor())
                        if self.config.GPU_COUNT:
                            mrcnn_class_logits = mrcnn_class_logits
                            mrcnn_class = mrcnn_class
                            mrcnn_bbox = mrcnn_bbox
                            mrcnn_mask = mrcnn_mask
                    else:
                        mrcnn_feature_maps = [p2_out[i].unsqueeze(0),
                            p3_out[i].unsqueeze(0), p4_out[i].unsqueeze(0),
                            p5_out[i].unsqueeze(0)]
                        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = (self
                            .classifier(mrcnn_feature_maps, rois))
                        mrcnn_mask = self.mask(mrcnn_feature_maps, rois)
                    (rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss,
                        mrcnn_bbox_loss, mrcnn_mask_loss) = (compute_losses
                        (rpn_match, rpn_bbox, rpn_class_logits,
                        rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                        target_deltas, mrcnn_bbox, target_mask, mrcnn_mask))
                    img_loss = (rpn_class_loss + rpn_bbox_loss +
                        mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss)
                    loss = loss + img_loss
                loss = loss / BatchSize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
                None
                step = step + 1
                if step % steps == 0:
                    None
                    torch.save(self.state_dict(), self.checkpoint_path.
                        format(epoch))
                    break
        self.epoch = epochs

    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]
        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.apply(set_bn_eval)
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        proposal_count = (self.config.POST_NMS_ROIS_TRAINING if mode ==
            'training' else self.config.POST_NMS_ROIS_INFERENCE)
        rpn_rois = proposal_generator([rpn_class, rpn_bbox], proposal_count
            =proposal_count, nms_threshold=self.config.RPN_NMS_THRESHOLD,
            anchors=self.anchors, config=self.config)
        if mode == 'inference':
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(
                mrcnn_feature_maps, rpn_rois)
            detections = detection_layer(self.config, rpn_rois, mrcnn_class,
                mrcnn_bbox, image_metas)
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float
                (), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale
            detection_boxes = detections[:, :4] / scale
            detection_boxes = detection_boxes.unsqueeze(0)
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask]
        elif mode == 'training':
            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float
                (), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale
            gt_boxes = gt_boxes / scale
            rois, target_class_ids, target_deltas, target_mask = (
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes,
                gt_masks, self.config))
            if not rois.size()[0]:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits
                    mrcnn_class = mrcnn_class
                    mrcnn_bbox = mrcnn_bbox
                    mrcnn_mask = mrcnn_mask
            else:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(
                    mrcnn_feature_maps, rois)
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)
            return [rpn_class_logits, rpn_bbox, target_class_ids,
                mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask,
                mrcnn_mask]

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding = utils.resize_image(image,
                min_dim=self.config.IMAGE_MIN_DIM, max_dim=self.config.
                IMAGE_MAX_DIM, padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            image_meta = compose_image_meta(0, image.shape, window, np.
                zeros([self.config.NUM_CLASSES], dtype=np.int32))
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        zero_ix = np.where(detections[:, (4)] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        boxes = detections[:N, :4]
        class_ids = detections[:N, (4)].astype(np.int32)
        scores = detections[:N, (5)]
        masks = mrcnn_mask[(np.arange(N)), :, :, (class_ids)]
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
        exclude_ix = np.where((boxes[:, (2)] - boxes[:, (0)]) * (boxes[:, (
            3)] - boxes[:, (1)]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]
        full_masks = []
        for i in range(N):
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(
            (0,) + masks.shape[1:3])
        return boxes, class_ids, scores, full_masks


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

    def __init__(self, crop_height, crop_width, extrapolation_value=0,
        transform_fpcoor=True):
        super(RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.transform_fpcoor = transform_fpcoor

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
        image_height, image_width = featuremap.size()[2:4]
        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / float(self.crop_width)
            spacing_h = (y2 - y1) / float(self.crop_height)
            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self.crop_width - 1) / float(image_width - 1
                )
            nh = spacing_h * float(self.crop_height - 1) / float(
                image_height - 1)
            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)
        boxes = boxes.detach().contiguous()
        box_ind = box_ind.detach()
        return CropAndResizeFunction(self.crop_height, self.crop_width,
            self.extrapolation_value)(featuremap, boxes, box_ind)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jytime_Mask_RCNN_Pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SamePad2d(*[], **{'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(RPN(*[], **{'anchors_per_location': 4, 'anchor_stride': 1, 'depth': 1}), [torch.rand([4, 1, 4, 4])], {})

