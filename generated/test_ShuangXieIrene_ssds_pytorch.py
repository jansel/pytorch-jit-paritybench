import sys
_module = sys.modules[__name__]
del sys
demo = _module
lib = _module
dataset = _module
coco = _module
dataset_factory = _module
voc = _module
voc_eval = _module
layers = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
focal_loss = _module
l2norm = _module
multibox_loss = _module
modeling = _module
model_builder = _module
nets = _module
darknet = _module
mobilenet = _module
resnet = _module
vgg = _module
ssds = _module
fssd = _module
fssd_lite = _module
retina = _module
rfb = _module
rfb_lite = _module
ssd = _module
ssd_lite = _module
yolo = _module
ssds = _module
ssds_train = _module
utils = _module
box_utils = _module
config_parse = _module
dark2pth = _module
data_augment = _module
data_augment_test = _module
eval_utils = _module
fp16_utils = _module
nms = _module
_ext = _module
nms = _module
build = _module
nms_gpu = _module
nms_wrapper = _module
pycocotools = _module
cocoeval = _module
mask = _module
timer = _module
visualize_utils = _module
setup = _module
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


import torch.utils.data as data


import torchvision.transforms as transforms


import numpy as np


import uuid


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.autograd import Function


from torch.autograd import Variable


from math import sqrt as sqrt


from itertools import product as product


import torch.nn.functional as F


import torch.nn.init as init


from collections import namedtuple


import functools


import random


import torch.optim as optim


from torch.optim import lr_scheduler


import math


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


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
    return inter[:, :, (0)] * inter[:, :, (1)]


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
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


class FocalLoss(nn.Module):
    """SSD Weighted Loss Function
    Focal Loss for Dense Object Detection.
        
        Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.
    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                putting more focus on hard, misclassiﬁed examples
        size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(FocalLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors
        self.alpha = Variable(torch.ones(self.num_classes, 1) * cfg.alpha)
        self.gamma = cfg.gamma

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        num_priors = priors.size(0)
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum()
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        loss_l /= num_pos.data.sum()
        loss_c = self.focal_loss(conf_data.view(-1, self.num_classes), conf_t.view(-1, 1))
        return loss_l, loss_c

    def focal_loss(self, inputs, targets):
        """Focal loss.
        mean of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        """
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        loss = batch_loss.mean()
        return loss


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, priors, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class _conv_bn(nn.Module):

    def __init__(self, inp, oup, stride=1):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)


class _conv_block(nn.Module):

    def __init__(self, inp, oup, stride=1, expand_ratio=0.5):
        super(_conv_block, self).__init__()
        depth = int(oup * expand_ratio)
        self.conv = nn.Sequential(nn.Conv2d(inp, depth, 1, 1, bias=False), nn.BatchNorm2d(depth), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(depth, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.conv(x)


class _residual_block(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio=0.5):
        super(_residual_block, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        if self.use_res_connect:
            depth = int(oup * expand_ratio)
            self.conv = nn.Sequential(nn.Conv2d(inp, depth, 1, 1, bias=False), nn.BatchNorm2d(depth), nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(depth, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(0.1, inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(0.1, inplace=True))
        self.depth = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False), nn.BatchNorm2d(oup * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False), nn.BatchNorm2d(oup * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))


class _inverted_residual_bottleneck(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        self.depth = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _basicblock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=1, downsample=None):
        super(_basicblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes * expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * expansion)
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


class _bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None):
        super(_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
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


class FSSD(nn.Module):
    """FSSD: Feature Fusion Single Shot Multibox Detector
    See: https://arxiv.org/pdf/1712.00960.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        features： include to feature layers to fusion feature and build pyramids
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, features, feature_layer, num_classes):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.feature_layer = feature_layer[0][0]
        self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])
        self.norm = nn.BatchNorm2d(int(feature_layer[0][1][-1] / 2) * len(self.transforms), affine=True)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                sources.append(x)
        assert len(self.transforms) == len(sources)
        upsize = sources[0].size()[2], sources[0].size()[3]
        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)
        if phase == 'feature':
            return pyramids
        for x, l, c in zip(pyramids, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FSSDLite(nn.Module):
    """FSSD: Feature Fusion Single Shot Multibox Detector for embeded system
    See: https://arxiv.org/pdf/1712.00960.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        features： include to feature layers to fusion feature and build pyramids
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, features, feature_layer, num_classes):
        super(FSSDLite, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.feature_layer = feature_layer[0][0]
        self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])
        self.norm = nn.BatchNorm2d(int(feature_layer[0][1][-1] / 2) * len(self.transforms), affine=True)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources, transformed, pyramids, loc, conf = [list() for _ in range(5)]
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)
        assert len(self.transforms) == len(sources)
        upsize = sources[0].size()[2], sources[0].size()[3]
        for k, v in enumerate(self.transforms):
            size = None if k == 0 else upsize
            transformed.append(v(sources[k], size))
        x = torch.cat(transformed, 1)
        x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)
        if phase == 'feature':
            return pyramids
        for x, l, c in zip(pyramids, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class Retina(nn.Module):

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(Retina, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras[1])
        self.transforms = nn.ModuleList(extras[0])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = feature_layer[0]

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x, phase='eval'):
        sources, loc, conf = [list() for _ in range(3)]
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)
        for i in range(len(sources))[::-1]:
            if i != len(sources) - 1:
                xx = self.extras[i](self._upsample_add(xx, self.transforms[i](sources[i])))
            else:
                xx = self.transforms[i](sources[i])
            sources[i] = xx
        for i, v in enumerate(self.extras):
            if i >= len(sources):
                x = v(x)
                sources.append(x)
        if phase == 'feature':
            return sources
        for x in sources:
            loc.append(self.loc(x).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class RFB(nn.Module):
    """Receptive Field Block Net for Accurate and Fast Object Detection
    See: https://arxiv.org/pdf/1711.07767.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        norm: norm to add RFB module for previous feature extractor
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, norm, head, feature_layer, num_classes):
        super(RFB, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.norm = nn.ModuleList(norm)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = feature_layer[0]
        self.indicator = 0
        for layer in self.feature_layer:
            if isinstance(layer, int):
                continue
            elif layer == '' or layer == 'S':
                break
            else:
                self.indicator += 1

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources, loc, conf = [list() for _ in range(3)]
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                idx = self.feature_layer.index(k)
                if len(sources) == 0:
                    sources.append(self.norm[idx](x))
                else:
                    x = self.norm[idx](x)
                    sources.append(x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 1:
                sources.append(x)
        if phase == 'feature':
            return sources
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1), BasicConv(inter_planes // 2, inter_planes // 4 * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)), BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False))
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes // 2 * 3, kernel_size=3, stride=1, padding=1), BasicConv(inter_planes // 2 * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False))
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class BasicRFB_a_lite(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a_lite, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)), BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1), BasicConv(inter_planes // 2, inter_planes // 4 * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)), BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False))
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class BasicRFB_lite(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_lite, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes // 2 * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv(inter_planes // 2 * 3, inter_planes // 2 * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)), BasicSepConv(inter_planes // 2 * 3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes // 2 * 3, kernel_size=3, stride=1, padding=1), BasicConv(inter_planes // 2 * 3, inter_planes // 2 * 3, kernel_size=3, stride=stride, padding=1), BasicSepConv(inter_planes // 2 * 3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False))
        self.ConvLinear = BasicConv(3 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x1, x2), 1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out * self.scale + x
        else:
            short = self.shortcut(x)
            out = out * self.scale + short
        out = self.relu(out)
        return out


class RFBLite(nn.Module):
    """Receptive Field Block Net for Accurate and Fast Object Detection for embeded system
    See: https://arxiv.org/pdf/1711.07767.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        norm: norm to add RFB module for previous feature extractor
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(RFBLite, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.norm = BasicRFB_a_lite(feature_layer[1][0], feature_layer[1][0], stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = feature_layer[0]
        self.indicator = 0
        for layer in self.feature_layer:
            if isinstance(layer, int):
                continue
            elif layer == '' or layer == 'S':
                break
            else:
                self.indicator += 1

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources = list()
        loc = list()
        conf = list()
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)
        if phase == 'feature':
            return sources
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = feature_layer[0]

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources, loc, conf = [list() for _ in range(3)]
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        if phase == 'feature':
            return sources
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class SSDLite(nn.Module):
    """Single Shot Multibox Architecture for embeded system
    See: https://arxiv.org/pdf/1512.02325.pdf & 
    https://arxiv.org/pdf/1801.04381.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(SSDLite, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = feature_layer[0]

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources = list()
        loc = list()
        conf = list()
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            sources.append(x)
        if phase == 'feature':
            return sources
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class YOLO(nn.Module):
    """Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.feature_layer = [f for feature in feature_layer[0] for f in feature]
        self.feature_index = list()
        s = -1
        for feature in feature_layer[0]:
            s += len(feature)
            self.feature_index.append(s)

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        cat = dict()
        sources, loc, conf = [list() for _ in range(3)]
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                cat[k] = x
        for k, v in enumerate(self.extras):
            if isinstance(self.feature_layer[k], int):
                x = v(x, cat[self.feature_layer[k]])
            else:
                x = v(x)
            if k in self.feature_index:
                sources.append(x)
        if phase == 'feature':
            return sources
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if phase == 'eval':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


class _router_v2(nn.Module):

    def __init__(self, inp, oup, stride=2):
        super(_router_v2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, 1, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(0.1, inplace=True))
        self.stride = stride

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        B, C, H, W = x2.size()
        s = self.stride
        x2 = x2.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x2 = x2.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x2 = x2.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        x2 = x2.view(B, s * s * C, H // s, W // s)
        return torch.cat((x1, x2), dim=1)


class _router_v3(nn.Module):

    def __init__(self, inp, oup, stride=1, bilinear=True):
        super(_router_v3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, 1, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(0.1, inplace=True))
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(oup, oup, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        return torch.cat((x1, x2), dim=1)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicRFB,
     lambda: ([], {'in_planes': 18, 'out_planes': 4}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
    (BasicRFB_a,
     lambda: ([], {'in_planes': 18, 'out_planes': 4}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
    (BasicRFB_lite,
     lambda: ([], {'in_planes': 18, 'out_planes': 4}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
    (BasicSepConv,
     lambda: ([], {'in_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_basicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_conv_block,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_conv_bn,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_inverted_residual_bottleneck,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_residual_block,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_router_v2,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 2, 2]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_router_v3,
     lambda: ([], {'inp': 4, 'oup': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ShuangXieIrene_ssds_pytorch(_paritybench_base):
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

