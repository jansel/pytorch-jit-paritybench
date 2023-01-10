import sys
_module = sys.modules[__name__]
del sys
convt = _module
data = _module
config = _module
data_augment = _module
wider_lp = _module
detect = _module
layers = _module
prior_box = _module
modules = _module
multibox_loss = _module
models = _module
mobile_net_v3 = _module
mobilenetv3 = _module
net = _module
retina = _module
prepare_data = _module
base_data_adapter = _module
base_provider = _module
dataset_provider = _module
reformat_CCPD = _module
text_list_adapter = _module
train_data_tool = _module
train = _module
utils = _module
box_utils = _module
nms = _module
py_cpu_nms = _module
timer = _module

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


import numpy as np


import torch.backends.cudnn as cudnn


import time


import torchvision


from itertools import product as product


from math import ceil


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision.models._utils as _utils


import torchvision.models as models


import torchvision.models.detection.backbone_utils as backbone_utils


from collections import OrderedDict


import torch.optim as optim


import math


cfg_mnet = {'name': 'mobilenet0.25', 'min_sizes': [[16, 32], [64, 128], [256, 512]], 'steps': [8, 16, 32], 'variance': [0.1, 0.2], 'clip': False, 'loc_weight': 2.0, 'gpu_train': True, 'batch_size': 256, 'ngpu': 4, 'epoch': 100, 'decay1': 190, 'decay2': 220, 'image_size': 640, 'pretrain': True, 'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3}, 'in_channel': 32, 'out_channel': 64}


GPU = cfg_mnet['gpu_train']


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


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


def encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """
    matched = torch.reshape(matched, (matched.size(0), 4, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 4).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= variances[0] * priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy


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


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf
    landm_t[idx] = landm


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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
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
        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = priors.size(0)
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 8)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:12].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t
            conf_t = conf_t
            landm_t = landm_t
        zeros = torch.tensor(0)
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 8)
        landm_t = landm_t[pos_idx1].view(-1, 8)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        pos = conf_t != zeros
        conf_t[pos] = 1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c[pos.view(-1, 1)] = 0
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
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        return loss_l, loss_c, loss_landm


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), Hsigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):

    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):

    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity
        self.conv = nn.Sequential(conv_layer(inp, exp, 1, 1, 0, bias=False), norm_layer(exp), nlin_layer(inplace=True), conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False), norm_layer(exp), SELayer(exp), nlin_layer(inplace=True), conv_layer(exp, oup, 1, 1, 0, bias=False), norm_layer(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False), BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False), BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False), BasicConv(inter_planes, inter_planes // 2 * 3, kernel_size=3, stride=1, padding=1, groups=groups), BasicConv(inter_planes // 2 * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups))
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(conv_layer(inp, oup, 1, 1, 0, bias=False), norm_layer(oup), nlin_layer(inplace=True))


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileNetV3(nn.Module):

    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            mobile_setting = [[3, 16, 16, False, 'RE', 1], [3, 64, 24, False, 'RE', 2], [3, 72, 24, False, 'RE', 1], [5, 72, 40, True, 'RE', 2], [5, 120, 40, True, 'RE', 1], [5, 120, 40, True, 'RE', 1], [3, 240, 80, False, 'HS', 2], [3, 200, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 480, 112, True, 'HS', 1], [3, 672, 112, True, 'HS', 1], [5, 672, 160, True, 'HS', 2], [5, 960, 160, True, 'HS', 1], [5, 960, 160, True, 'HS', 1]]
        elif mode == 'small':
            mobile_setting = [[3, 16, 16, True, 'RE', 2], [3, 72, 24, False, 'RE', 2], [3, 88, 24, False, 'RE', 1], [5, 96, 40, True, 'HS', 2], [5, 240, 40, True, 'HS', 1], [5, 240, 40, True, 'HS', 1], [5, 120, 48, True, 'HS', 1], [5, 144, 48, True, 'HS', 1], [5, 288, 96, True, 'HS', 2], [5, 576, 96, True, 'HS', 1], [5, 576, 96, True, 'HS', 1]]
        else:
            raise NotImplementedError
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))


class SSH(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)
        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class FPN(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        input = list(input.values())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)
        out = [output1, output2, output3]
        return out


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.LeakyReLU(negative_slope=leaky, inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.LeakyReLU(negative_slope=leaky, inplace=True))


class MobileNetV1(nn.Module):

    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(conv_bn(3, 8, 2, leaky=0.1), conv_dw(8, 16, 1), conv_dw(16, 32, 2), conv_dw(32, 32, 1), conv_dw(32, 64, 2), conv_dw(64, 64, 1))
        self.stage2 = nn.Sequential(conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1), conv_dw(128, 128, 1))
        self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 8, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 8)


class Retina(nn.Module):

    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(Retina, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load('./weights/mobilenetV1X0.25_pretrain.tar', map_location=torch.device('cpu'))
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        """
        elif cfg['name'] == 'mobilenetv3':
            from models.mobile_net_v3 import mobilenetv3
            backbone = mobilenetv3() 
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])   
        """
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        if self.phase == 'train':
            output = bbox_regressions, classifications, ldm_regressions
        else:
            output = bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BboxHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (ClassHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LandmarkHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {}),
     True),
    (MobileBottleneck,
     lambda: ([], {'inp': 4, 'oup': 4, 'kernel': 3, 'stride': 1, 'exp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSH,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_gm19900510_Pytorch_Retina_License_Plate(_paritybench_base):
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

