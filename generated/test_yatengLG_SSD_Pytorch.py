import sys
_module = sys.modules[__name__]
del sys
Configs = _module
Dataloader = _module
Dataset_VOC = _module
Transfroms = _module
Transfroms_utils = _module
Data = _module
Demo_detect_one_image = _module
Demo_detect_video = _module
Demo_eval = _module
Demo_train = _module
Model = _module
VGG = _module
vgg16 = _module
base_models = _module
evaler = _module
load_pretrained_weight = _module
ssd_model = _module
Anchors = _module
MultiBoxLoss = _module
PostProcess = _module
Predictor = _module
structs = _module
trainer = _module
Boxs_op = _module
Cal_mean_std = _module
Hash = _module
Utils = _module
utils = _module
visdom_op = _module
voc_cal_ap = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch._six import int_classes as _int_classes


from torch.utils.data import DataLoader


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data import Sampler


from torch.utils.data.dataloader import default_collate


import torch.utils.data


import numpy as np


import torch


from torchvision import transforms


import types


from numpy import random


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import init


from torch.nn import DataParallel


from torch import nn


import time


from math import sqrt


import math


import torchvision


from torch.optim.lr_scheduler import MultiStepLR


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def GetFileMd5(filename):
    if not os.path.isfile(filename):
        return
    myHash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myHash.update(b)
    f.close()
    return myHash.hexdigest()


def add_extras(cfg, i, size=300):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


extras_base = {'300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256], '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]}


vgg_base = {'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512], '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]}


class VGG(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        vgg_config = vgg_base[str(self.cfg.MODEL.INPUT.IMAGE_SIZE)]
        extras_config = extras_base[str(self.cfg.MODEL.INPUT.IMAGE_SIZE)]
        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=self.cfg.MODEL.INPUT.IMAGE_SIZE))
        self.l2_norm = L2Norm(512, scale=20)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)
        features.append(s)
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)
        return tuple(features)

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def load_weights(self):
        url = 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth'
        weight_name = url.split('/')[-1]
        weight_path = self.cfg.FILE.PRETRAIN_WEIGHT_ROOT
        weight_file = os.path.join(weight_path, weight_name)
        if not os.path.exists(weight_file):
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)
            None
            wget.download(url=url, out=weight_file)
            None
        md5 = GetFileMd5(weight_file)
        if md5 == '9fb5a8dfd5f42dc1090365b6179aa659':
            self.load_state_dict(torch.load(weight_file), strict=False)
            None
        else:
            None


class ConvertFromInts(object):

    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Expand(object):

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += int(left), int(top)
        boxes[:, 2:] += int(left), int(top)
        return image, boxes, labels


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> Compose([
        >>>         transforms.CenterCrop(10),
        >>>         transforms.ToTensor(),
        >>>         ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertColor(object):
    """
        H色调用角度度量,取值范围为0°～360°.从红色开始按逆时针方向计算,红色为0°,绿色为120°,蓝色为240°.它们的补色是:黄色为60°,青色为180°,品红为300°;
        S饱和度表示颜色接近光谱色的程度.一种颜色,可以看成是某种光谱色与白色混合的结果.其中光谱色所占的比例愈大，颜色接近光谱色的程度就愈高，颜色的饱和度也就愈高;
        明度表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，此值和物体的透射比或反射比有关。通常取值范围为0%（黑）到100%（白）。
    """

    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomBrightness(object):

    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class RandomContrast(object):

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomHue(object):

    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, (0)] += random.uniform(-self.delta, self.delta)
            image[:, :, (0)][image[:, :, (0)] > 360.0] -= 360.0
            image[:, :, (0)][image[:, :, (0)] < 0.0] += 360.0
        return image, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, (self.swaps)]
        return image


class RandomLightingNoise(object):

    def __init__(self):
        self.perms = (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels


class RandomSaturation(object):

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper must be >= lower.'
        assert self.lower >= 0, 'contrast lower must be non-negative.'

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, (1)] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels


class PhotometricDistort(object):

    def __init__(self):
        self.pd = [RandomContrast(), ConvertColor(current='RGB', transform='HSV'), RandomSaturation(), RandomHue(), ConvertColor(current='HSV', transform='RGB'), RandomContrast()]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class RandomMirror(object):

    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=np.inf)
    return inter[:, (0)] * inter[:, (1)]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = None, (0.1, None), (0.3, None), (0.7, None), (0.9, None), (None, None)

    def __call__(self, image, boxes=None, labels=None):
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap = jaccard_numpy(boxes, rect)
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, (0)]) * (rect[1] < centers[:, (1)])
                m2 = (rect[2] > centers[:, (0)]) * (rect[3] > centers[:, (1)])
                mask = m1 * m2
                if not mask.any():
                    continue
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                current_boxes = boxes[(mask), :].copy()
                current_labels = labels[mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                return current_image, current_boxes, current_labels


class Resize(object):

    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class SubtractMeans(object):

    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToPercentCoords(object):

    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, (0)] /= width
        boxes[:, (2)] /= width
        boxes[:, (1)] /= height
        boxes[:, (3)] /= height
        return image, boxes, labels


class ToTensor(object):

    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class SSDTramsfrom:
    """
    targets_transfroms
    eg:
        transform = SSDTramsfrom(cfg,is_train=True)
    """

    def __init__(self, cfg, is_train):
        if is_train:
            self.transforms = [ConvertFromInts(), PhotometricDistort(), Expand(), RandomSampleCrop(), RandomMirror(), ToPercentCoords(), Resize(cfg.MODEL.INPUT.IMAGE_SIZE), ToTensor()]
        else:
            self.transforms = [Resize(cfg.MODEL.INPUT.IMAGE_SIZE), SubtractMeans(cfg.MODEL.INPUT.PIXEL_MEAN), ToTensor()]

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


def boxes_nms(boxes, scores, nms_thresh, max_count=-1):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(or relative coordinates), shape is (n, 4)
        scores(Tensor): scores, shape is (n, )
        nms_thresh(float): thresh
        max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    Returns:
        indices kept.
    """
    keep = torchvision.ops.nms(boxes, scores, nms_thresh)
    if max_count > 0:
        keep = keep[:max_count]
    return keep


def center_form_to_corner_form(locations):
    return torch.cat([locations[(...), :2] - locations[(...), 2:] / 2, locations[(...), :2] + locations[(...), 2:] / 2], locations.dim() - 1)


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\\_center * center_variance = rac {real\\_center - prior\\_center} {prior\\_hw}$$
        $$exp(predicted\\_hw * size_variance) = rac {real\\_hw} {prior\\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([locations[(...), :2] * center_variance * priors[(...), 2:] + priors[(...), :2], torch.exp(locations[(...), 2:] * size_variance) * priors[(...), 2:]], dim=locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([(boxes[(...), :2] + boxes[(...), 2:]) / 2, boxes[(...), 2:] - boxes[(...), :2]], boxes.dim() - 1)


class priorbox:

    def __init__(self, cfg):
        """
        SSD默认检测框生成器
        :param cfg:
        """
        self.image_size = cfg.MODEL.INPUT.IMAGE_SIZE
        anchor_config = cfg.MODEL.ANCHORS
        self.feature_maps = anchor_config.FEATURE_MAPS
        self.min_sizes = anchor_config.MIN_SIZES
        self.max_sizes = anchor_config.MAX_SIZES
        self.aspect_ratios = anchor_config.ASPECT_RATIOS
        self.clip = anchor_config.CLIP

    def __call__(self):
        """SSD默认检测框生成
            :return
                Tensor(num_priors,boxes)
                其中boxes(x, y, w, h)
                检测框为比例存储,0~1
        """
        priors = []
        for k, (feature_map_w, feature_map_h) in enumerate(self.feature_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    size = self.min_sizes[k]
                    h = w = size / self.image_size
                    priors.append([cx, cy, w, h])
                    size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                    h = w = size / self.image_size
                    priors.append([cx, cy, w, h])
                    size = self.min_sizes[k]
                    h = w = size / self.image_size
                    for ratio in self.aspect_ratios[k]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, w * ratio, h / ratio])
                        priors.append([cx, cy, w / ratio, h * ratio])
        priors = torch.tensor(priors)
        if self.clip:
            priors = center_form_to_corner_form(priors)
            priors.clamp_(max=1, min=0)
            priors = corner_form_to_center_form(priors)
        return priors


class postprocessor:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.MODEL.INPUT.IMAGE_SIZE
        self.height = cfg.MODEL.INPUT.IMAGE_SIZE

    def __call__(self, cls_logits, bbox_pred):
        priors = priorbox(self.cfg)()
        batches_scores = F.softmax(cls_logits, dim=2)
        boxes = convert_locations_to_boxes(bbox_pred, priors, self.cfg.MODEL.ANCHORS.CENTER_VARIANCE, self.cfg.MODEL.ANCHORS.CENTER_VARIANCE)
        batches_boxes = center_form_to_corner_form(boxes)
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            processed_boxes = []
            processed_scores = []
            processed_labels = []
            per_img_scores, per_img_boxes = batches_scores[batch_id], batches_boxes[batch_id]
            for class_id in range(1, per_img_scores.size(1)):
                scores = per_img_scores[:, (class_id)]
                mask = scores > self.cfg.MODEL.TEST.CONFIDENCE_THRESHOLD
                scores = scores[mask]
                if scores.size(0) == 0:
                    continue
                boxes = per_img_boxes[(mask), :]
                boxes[:, 0::2] *= self.width
                boxes[:, 1::2] *= self.height
                keep = boxes_nms(boxes, scores, self.cfg.MODEL.TEST.NMS_THRESHOLD, self.cfg.MODEL.TEST.MAX_PER_CLASS)
                nmsed_boxes = boxes[(keep), :]
                nmsed_labels = torch.tensor([class_id] * keep.size(0), device=device)
                nmsed_scores = scores[keep]
                processed_boxes.append(nmsed_boxes)
                processed_scores.append(nmsed_scores)
                processed_labels.append(nmsed_labels)
            if len(processed_boxes) == 0:
                processed_boxes = torch.empty(0, 4)
                processed_labels = torch.empty(0)
                processed_scores = torch.empty(0)
            else:
                processed_boxes = torch.cat(processed_boxes, 0)
                processed_labels = torch.cat(processed_labels, 0)
                processed_scores = torch.cat(processed_scores, 0)
            if processed_boxes.size(0) > self.cfg.MODEL.TEST.MAX_PER_IMAGE > 0:
                processed_scores, keep = torch.topk(processed_scores, k=self.cfg.MODEL.TEST.MAX_PER_IMAGE)
                processed_boxes = processed_boxes[(keep), :]
                processed_labels = processed_labels[keep]
            results.append([processed_boxes, processed_labels, processed_scores])
        return results


class predictor(nn.Module):
    """
    分类(cls)及回归(reg)网络
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for boxes_per_location, out_channels in zip(cfg.MODEL.ANCHORS.BOXES_PER_LOCATION, cfg.MODEL.ANCHORS.OUT_CHANNELS):
            self.cls_headers.append(self.cls_block(out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.DATA.DATASET.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        对输入的特征图中每个特征点进行分类及回归(不同特征图特征点对应的输出数是不一样的,以检测框数量为准)
        :param features:    # base_model 输出的特征图,这里SSD_VGG_300 为六层特征图
        :return:            # 每个特征点的类别预测与回归预测(输出数量以各自特征点上检测框数量为准)
        """
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())
        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.DATA.DATASET.NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)
        return cls_logits, bbox_pred


def vgg(cfg, pretrained=True):
    None
    model = VGG(cfg)
    if pretrained:
        model.load_weights()
    else:
        model.reset_parameters()
    return model


class SSD(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = vgg(cfg, pretrained=True)
        self.predictor = predictor(cfg)
        self.postprocessor = postprocessor(cfg)
        self.priors = priorbox(self.cfg)()

    def forward(self, images):
        features = self.backbone(images)
        cls_logits, bbox_pred = self.predictor(features)
        return cls_logits, bbox_pred

    def load_pretrained_weight(self, weight_pkl):
        self.load_state_dict(torch.load(weight_pkl))

    def forward_with_postprocess(self, images):
        """
        前向传播并后处理
        :param images:
        :return:
        """
        cls_logits, bbox_pred = self.forward(images)
        detections = self.postprocessor(cls_logits, bbox_pred)
        return detections

    @torch.no_grad()
    def Detect_single_img(self, image, score_threshold=0.7, device='cuda'):
        """
        检测单张照片
        eg:
            image, boxes, labels, scores= net.Detect_single_img(img)
            plt.imshow(image)
            plt.show()

        :param image:           图片,PIL.Image.Image
        :param score_threshold: 阈值
        :param device:          检测时所用设备,默认'cuda'
        :return:                添加回归框的图片(np.array),回归框,标签,分数
        """
        self.eval()
        assert isinstance(image, Image.Image)
        w, h = image.width, image.height
        images_tensor = SSDTramsfrom(self.cfg, is_train=False)(np.array(image))[0].unsqueeze(0)
        self
        images_tensor = images_tensor
        time1 = time.time()
        detections = self.forward_with_postprocess(images_tensor)[0]
        boxes, labels, scores = detections
        boxes, labels, scores = boxes.numpy(), labels.numpy(), scores.numpy()
        boxes[:, 0::2] *= w / self.cfg.MODEL.INPUT.IMAGE_SIZE
        boxes[:, 1::2] *= h / self.cfg.MODEL.INPUT.IMAGE_SIZE
        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        None
        drawn_image = draw_boxes(image=image, boxes=boxes, labels=labels, scores=scores, class_name_map=self.cfg.DATA.DATASET.CLASS_NAME).astype(np.uint8)
        return drawn_image, boxes, labels, scores

    @torch.no_grad()
    def Detect_video(self, video_path, score_threshold=0.5, save_video_path=None, show=True):
        """
        检测视频
        :param video_path:      视频路径  eg: /XXX/aaa.mp4
        :param score_threshold:
        :param save_video_path: 保存路径,不指定则不保存
        :param show:            在检测过程中实时显示,(会存在卡顿现象,受检测效率影响)
        :return:
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if save_video_path:
            out = cv2.VideoWriter(save_video_path, fourcc, cap.get(5), (weight, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                drawn_image, boxes, labels, scores = self.Detect_single_img(image=image, device='cuda:0', score_threshold=score_threshold)
                frame = cv2.cvtColor(np.asarray(drawn_image), cv2.COLOR_RGB2BGR)
                if show:
                    cv2.imshow('frame', frame)
                if save_video_path:
                    out.write(frame)
                if cv2.waitKey(1) & 255 == ord('q'):
                    break
            else:
                break
        cap.release()
        if save_video_path:
            out.release()
        cv2.destroyAllWindows()
        return True


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio
    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


class multiboxloss(nn.Module):

    def __init__(self, neg_pos_ratio):
        """
        SSD损失函数,分为类别损失(使用cross_entropy)
        框体回归损失(使用smooth_l1_loss)
        这里并没有在返回时,采用分别返回的方式返回.便于训练过程中分析处理
        :param neg_pos_ratio:
        """
        super(multiboxloss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """
            计算类别损失和框体回归损失
        Args:
            confidence (batch_size, num_priors, num_classes): 预测类别
            predicted_locations (batch_size, num_priors, 4): 预测位置
            labels (batch_size, num_priors): 所有框的真实类别
            gt_locations (batch_size, num_priors, 4): 所有框真实的位置
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, (0)]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)
        confidence = confidence[(mask), :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[(pos_mask), :].view(-1, 4)
        gt_locations = gt_locations[(pos_mask), :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yatengLG_SSD_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

