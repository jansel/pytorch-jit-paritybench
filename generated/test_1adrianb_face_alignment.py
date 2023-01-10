import sys
_module = sys.modules[__name__]
del sys
detect_landmarks_in_image = _module
face_alignment = _module
api = _module
detection = _module
blazeface = _module
blazeface_detector = _module
detect = _module
net_blazeface = _module
utils = _module
core = _module
dlib = _module
dlib_detector = _module
folder = _module
folder_detector = _module
sfd = _module
bbox = _module
detect = _module
net_s3fd = _module
sfd_detector = _module
utils = _module
setup = _module
facealignment_test = _module
smoke_test = _module
test_utils = _module

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


import warnings


from enum import IntEnum


import numpy as np


from torch.utils.model_zoo import load_url


import torch.nn.functional as F


import torch.nn as nn


import logging


import math


from torch.hub import download_url_to_file


from torch.hub import HASH_REGEX


import re


class BlazeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=True), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), 'constant', 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(self.convs(h) + x)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


class BlazeFace(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self):
        super(BlazeFace, self).__init__()
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3
        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True), nn.ReLU(inplace=True), BlazeBlock(24, 24), BlazeBlock(24, 28), BlazeBlock(28, 32, stride=2), BlazeBlock(32, 36), BlazeBlock(36, 42), BlazeBlock(42, 48, stride=2), BlazeBlock(48, 56), BlazeBlock(56, 64), BlazeBlock(64, 72), BlazeBlock(72, 80), BlazeBlock(80, 88))
        self.backbone2 = nn.Sequential(BlazeBlock(88, 96, stride=2), BlazeBlock(96, 96), BlazeBlock(96, 96), BlazeBlock(96, 96), BlazeBlock(96, 96))
        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
        self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2), 'constant', 0)
        b = x.shape[0]
        x = self.backbone1(x)
        h = self.backbone2(x)
        c1 = self.classifier_8(x)
        c1 = c1.permute(0, 2, 3, 1)
        c1 = c1.reshape(b, -1, 1)
        c2 = self.classifier_16(h)
        c2 = c2.permute(0, 2, 3, 1)
        c2 = c2.reshape(b, -1, 1)
        c = torch.cat((c1, c2), dim=1)
        r1 = self.regressor_8(x)
        r1 = r1.permute(0, 2, 3, 1)
        r1 = r1.reshape(b, -1, 16)
        r2 = self.regressor_16(h)
        r2 = r2.permute(0, 2, 3, 1)
        r2 = r2.reshape(b, -1, 16)
        r = torch.cat((r1, r2), dim=1)
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path, device=None):
        device = device or self._device()
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=device)
        assert self.anchors.ndimension() == 2
        assert self.anchors.shape[0] == self.num_anchors
        assert self.anchors.shape[1] == 4

    def load_anchors_from_npy(self, arr, device=None):
        device = device or self._device()
        self.anchors = torch.tensor(arr, dtype=torch.float32, device=device)
        assert self.anchors.ndimension() == 2
        assert self.anchors.shape[0] == self.num_anchors
        assert self.anchors.shape[1] == 4

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))
        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))
        assert x.shape[1] == 3
        assert x.shape[2] == 128
        assert x.shape[3] == 128
        x = x
        x = self._preprocess(x)
        with torch.no_grad():
            out = self.__call__(x)
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
            filtered_detections.append(faces)
        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords
        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes
        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        mask = detection_scores >= self.min_score_thresh
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))
        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)
        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]
        boxes[..., 0] = y_center - h / 2.0
        boxes[..., 1] = x_center - w / 2.0
        boxes[..., 2] = y_center + h / 2.0
        boxes[..., 3] = x_center + w / 2.0
        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y
        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.

        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0:
            return []
        output_detections = []
        remaining = torch.argsort(detections[:, 16], descending=True)
        while len(remaining) > 0:
            detection = detections[remaining[0]]
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)
            output_detections.append(weighted_detection)
        return output_detections


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.empty(self.n_channels).fill_(self.scale))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class s3fd(nn.Module):

    def __init__(self):
        super(s3fd, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)
        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1_1(x), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)
        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)
        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)
        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)
        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)
        h = F.relu(self.fc6(h), inplace=True)
        h = F.relu(self.fc7(h), inplace=True)
        ffc7 = h
        h = F.relu(self.conv6_1(h), inplace=True)
        h = F.relu(self.conv6_2(h), inplace=True)
        f6_2 = h
        h = F.relu(self.conv7_1(h), inplace=True)
        h = F.relu(self.conv7_2(h), inplace=True)
        f7_2 = h
        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)
        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)
        chunk = torch.chunk(cls1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)
        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BlazeBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BlazeFace,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (L2Norm,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (s3fd,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_1adrianb_face_alignment(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

